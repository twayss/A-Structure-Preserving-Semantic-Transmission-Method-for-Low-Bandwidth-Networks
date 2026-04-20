import os
import cv2
import csv
import numpy as np
import io
import torch
import lpips
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from tqdm import tqdm
#这里路径以总文件夹为相对路径，建议根据自身实际路径替换
# ================= 核心配置 =================
# 1. 你的 Edge-Link 生成结果 CSV
CSV_PATH = r"./results/inference_v73_Set14/metrics_v73_std_512.csv"


# 2. 你的 Edge-Link 生成图片文件夹 (Recon)
OURS_DIR = r"./results/inference_v73_Set14/recon_std_512"


# 3. 原始高清图片文件夹 (Ground Truth)
GT_DIR = r"./src/dataset/Set14"


# 4. 输出对比图的文件夹
OUTPUT_DIR = r"final_comparison"

# 5. 新的指标汇总 CSV 输出路径
OUTPUT_METRICS_CSV = r"final_comparison_metrics.csv"

# 是否使用 GPU
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===========================================

# 尝试导入 pyiqa，如果没有则提示
try:
    import pyiqa
except ImportError:
    print("❌ 错误: 缺少 pyiqa 库。请运行: pip install pyiqa")
    exit()


def get_image_size_kb(buffer):
    return len(buffer) / 1024.0


# ----------------- 指标计算函数 -----------------
def calc_ssim(img1_np, img2_np):
    """ 计算 SSIM (全参考) """
    return ssim(img1_np, img2_np, channel_axis=2, data_range=255)


def calc_lpips(loss_fn, img1_np, img2_np):
    """ 计算 LPIPS (全参考, 越低越好) """
    # LPIPS 需要 [-1, 1]
    t1 = torch.from_numpy(img1_np).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
    t2 = torch.from_numpy(img2_np).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
    t1 = t1.unsqueeze(0).to(DEVICE)
    t2 = t2.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = loss_fn(t1, t2)
    return score.item()


def calc_niqe(niqe_fn, img_np):
    """
    计算 NIQE (无参考, 越低越好)
    img_np: HxWx3 RGB, 0-255
    """
    # PyIQA 通常需要 [0, 1] 的 RGB Tensor
    t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    t = t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = niqe_fn(t)
    return score.item()


# ------------------------------------------------

def compress_to_target_kb(img_pil, fmt, target_kb):
    """暴力搜索，找到最接近目标大小的 Quality"""
    best_buffer = None
    closest_diff = float('inf')
    best_q = 0
    # 动态调整搜索范围
    for q in range(1, 101):
        buf = io.BytesIO()
        try:
            img_pil.save(buf, format=fmt, quality=q)
        except:
            continue
        size = get_image_size_kb(buf.getvalue())
        diff = abs(size - target_kb)
        if diff < closest_diff:
            closest_diff = diff
            best_buffer = buf
            best_q = q
    return best_buffer, best_q, get_image_size_kb(best_buffer.getvalue())


def add_label(img, text_lines, color=(0, 255, 0)):
    """支持多行文本标签"""
    h, w = img.shape[:2]
    # 根据行数调整黑色背景高度
    line_height = 22
    header_height = line_height * len(text_lines) + 8

    cv2.rectangle(img, (0, 0), (w, header_height), (0, 0, 0), -1)

    y = 18
    for line in text_lines:
        cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += line_height
    return img


def main():
    print(f"🔥 Loading Metrics on {DEVICE}...")
    # 1. 加载 LPIPS
    loss_fn_lpips = lpips.LPIPS(net='alex').to(DEVICE)

    # 2. 加载 NIQE (无参考指标)
    # create_metric 会自动下载预训练权重
    print("⏳ Loading NIQE model (might download weights)...")
    niqe_fn = pyiqa.create_metric('niqe', device=DEVICE, as_loss=False)

    if not os.path.exists(CSV_PATH):
        print(f"❌ Input CSV not found: {CSV_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['Filename'] == 'AVERAGE': continue
            tasks.append(row)

    print(f"🚀 Starting fair comparison with NIQE for {len(tasks)} images...")

    # CSV Header 增加 NIQE 列
    csv_header = [
        "Filename", "Target_KB",
        "Ours_SSIM", "Ours_LPIPS", "Ours_NIQE",
        "JPEG_SSIM", "JPEG_LPIPS", "JPEG_NIQE", "JPEG_Q",
        "WebP_SSIM", "WebP_LPIPS", "WebP_NIQE", "WebP_Q"
    ]

    with open(OUTPUT_METRICS_CSV, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(csv_header)

        for row in tqdm(tasks):
            filename = row['Filename']
            target_kb = float(row['Payload (KB)'])

            # ================= 准备数据 =================
            gt_path = os.path.join(GT_DIR, filename)
            if not os.path.exists(gt_path): continue

            # 读取 GT (转为 512x512)
            gt_bgra = cv2.imread(gt_path)
            gt_rgb = cv2.cvtColor(gt_bgra, cv2.COLOR_BGR2RGB)
            gt_pil = Image.fromarray(gt_rgb).resize((512, 512), Image.LANCZOS)
            gt_rgb_512 = np.array(gt_pil)
            gt_bgr_512 = cv2.cvtColor(gt_rgb_512, cv2.COLOR_RGB2BGR)

            # 读取 Ours
            our_path = os.path.join(OURS_DIR, filename)
            if not os.path.exists(our_path): continue
            our_bgr = cv2.imread(our_path)
            our_rgb = cv2.cvtColor(our_bgr, cv2.COLOR_BGR2RGB)
            if our_bgr.shape != gt_bgr_512.shape:
                our_bgr = cv2.resize(our_bgr, (512, 512))
                our_rgb = cv2.resize(our_rgb, (512, 512))

            # 生成 JPEG
            jpg_buf, jpg_q, jpg_size = compress_to_target_kb(gt_pil, 'JPEG', target_kb)
            jpg_rgb = np.array(Image.open(jpg_buf))
            jpg_bgr = cv2.cvtColor(jpg_rgb, cv2.COLOR_RGB2BGR)
            if jpg_rgb.shape != gt_rgb_512.shape:
                jpg_rgb = cv2.resize(jpg_rgb, (512, 512))
                jpg_bgr = cv2.resize(jpg_bgr, (512, 512))

            # 生成 WebP
            webp_buf, webp_q, webp_size = compress_to_target_kb(gt_pil, 'WEBP', target_kb)
            webp_rgb = np.array(Image.open(webp_buf))
            webp_bgr = cv2.cvtColor(webp_rgb, cv2.COLOR_RGB2BGR)
            if webp_rgb.shape != gt_rgb_512.shape:
                webp_rgb = cv2.resize(webp_rgb, (512, 512))
                webp_bgr = cv2.resize(webp_bgr, (512, 512))

            # ================= 计算指标 =================
            # 1. Ours
            s_ours = calc_ssim(gt_rgb_512, our_rgb)
            l_ours = calc_lpips(loss_fn_lpips, gt_rgb_512, our_rgb)
            n_ours = calc_niqe(niqe_fn, our_rgb)  # No-Reference

            # 2. JPEG
            s_jpg = calc_ssim(gt_rgb_512, jpg_rgb)
            l_jpg = calc_lpips(loss_fn_lpips, gt_rgb_512, jpg_rgb)
            n_jpg = calc_niqe(niqe_fn, jpg_rgb)

            # 3. WebP
            s_webp = calc_ssim(gt_rgb_512, webp_rgb)
            l_webp = calc_lpips(loss_fn_lpips, gt_rgb_512, webp_rgb)
            n_webp = calc_niqe(niqe_fn, webp_rgb)

            # 4. GT NIQE (Optional, just for reference)
            n_gt = calc_niqe(niqe_fn, gt_rgb_512)

            # 写入 CSV
            writer.writerow([
                filename, f"{target_kb:.2f}",
                f"{s_ours:.4f}", f"{l_ours:.4f}", f"{n_ours:.4f}",
                f"{s_jpg:.4f}", f"{l_jpg:.4f}", f"{n_jpg:.4f}", jpg_q,
                f"{s_webp:.4f}", f"{l_webp:.4f}", f"{n_webp:.4f}", webp_q
            ])

            # ================= 生成对比图 =================
            h, w = gt_bgr_512.shape[:2]
            canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

            # GT
            t_gt = add_label(gt_bgr_512.copy(),
                             ["Ground Truth", f"NIQE: {n_gt:.3f} (Reference)"],
                             (255, 255, 255))

            # Ours (Green)
            ours_label = [
                f"Ours: {target_kb:.2f}KB",
                f"SSIM:{s_ours:.3f} LPIPS:{l_ours:.3f}",
                f"NIQE:{n_ours:.3f} (Naturalness)"
            ]
            t_ours = add_label(our_bgr.copy(), ours_label, (0, 255, 0))

            # JPEG (Red)
            jpg_label = [
                f"JPEG (Q={jpg_q}): {jpg_size:.2f}KB",
                f"SSIM:{s_jpg:.3f} LPIPS:{l_jpg:.3f}",
                f"NIQE:{n_jpg:.3f}"
            ]
            t_jpg = add_label(jpg_bgr.copy(), jpg_label, (0, 0, 255))

            # WebP (Cyan)
            webp_label = [
                f"WebP (Q={webp_q}): {webp_size:.2f}KB",
                f"SSIM:{s_webp:.3f} LPIPS:{l_webp:.3f}",
                f"NIQE:{n_webp:.3f}"
            ]
            t_webp = add_label(webp_bgr.copy(), webp_label, (0, 255, 255))

            canvas[0:h, 0:w] = t_gt
            canvas[0:h, w:w * 2] = t_ours
            canvas[h:h * 2, 0:w] = t_jpg
            canvas[h:h * 2, w:w * 2] = t_webp

            save_name = f"Compare_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), canvas)

    print(f"\n✅ All Done! Metrics saved to: {OUTPUT_METRICS_CSV}")
    print(f"🖼️  Comparisons with NIQE saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
