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
CSV_PATH = r"./results/inference_v73_DIV2K/metrics_v73_native.csv"
OURS_DIR = r"./results/inference_v73_DIV2K/recon_native"
GT_DIR = r"./src/dataset/DIV2K_valid_HR"

# 输出文件夹和 CSV
OUTPUT_DIR = r"final_comparison_DIV_SMART_CROP"
OUTPUT_METRICS_CSV = r"final_comparison_DIV_metrics_smart_crop.csv"

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# LPIPS 显存保护（全图计算时如果太大则缩放，Crop 计算时不缩放）
LIMIT_LPIPS_RES = True
# Crop 的大小 (用于计算局部指标和展示)
CROP_SIZE = 256
# ===========================================

try:
    import pyiqa
except ImportError:
    print("❌ 错误: 缺少 pyiqa 库。请运行: pip install pyiqa")
    exit()


def get_image_size_kb(buffer):
    return len(buffer) / 1024.0


# ----------------- 指标计算 -----------------
def calc_ssim(img1_np, img2_np):
    return ssim(img1_np, img2_np, channel_axis=2, data_range=255)


def calc_lpips(loss_fn, img1_np, img2_np, use_limit=False):
    t1 = torch.from_numpy(img1_np).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0
    t2 = torch.from_numpy(img2_np).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0

    if use_limit and LIMIT_LPIPS_RES:
        _, h, w = t1.shape
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            import torch.nn.functional as F
            t1 = F.interpolate(t1.unsqueeze(0), size=(new_h, new_w), mode='bilinear').squeeze(0)
            t2 = F.interpolate(t2.unsqueeze(0), size=(new_h, new_w), mode='bilinear').squeeze(0)

    t1 = t1.unsqueeze(0).to(DEVICE)
    t2 = t2.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = loss_fn(t1, t2)
    return score.item()


def calc_niqe(niqe_fn, img_np, use_limit=False):
    t = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0
    if use_limit and LIMIT_LPIPS_RES:
        _, h, w = t.shape
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            import torch.nn.functional as F
            t = F.interpolate(t.unsqueeze(0), size=(new_h, new_w), mode='bilinear').squeeze(0)
    t = t.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        score = niqe_fn(t)
    return score.item()


# ----------------- 压缩逻辑 -----------------
def compress_to_target_kb_strict(img_pil, fmt, target_kb):
    """
    🔥 严格公平模式 (Resolution Crushing)
    """
    original_w, original_h = img_pil.size

    # 阶段 1: 原分辨率
    best_buffer = None
    closest_diff = float('inf')
    best_q = 1

    for q in range(1, 101, 2):
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

        if size > target_kb * 1.5: break

    final_size = get_image_size_kb(best_buffer.getvalue())

    # 阶段 2: 熔断降级 (如果 Q=1 也压不住)
    is_crushed = False
    downscale_factor = 1.0

    if final_size > target_kb * 1.1:
        current_w, current_h = original_w, original_h
        panic_q = 5  # 给一点点画质，靠降分辨率来换体积

        while True:
            current_w = int(current_w * 0.9)
            current_h = int(current_h * 0.9)
            if current_w < 32 or current_h < 32: break

            current_img = img_pil.resize((current_w, current_h), Image.LANCZOS)
            buf = io.BytesIO()
            current_img.save(buf, format=fmt, quality=panic_q)
            size = get_image_size_kb(buf.getvalue())

            if size <= target_kb * 1.05:
                best_buffer = buf
                best_q = panic_q
                final_size = size
                is_crushed = True
                downscale_factor = current_w / original_w
                break

    return best_buffer, best_q, final_size, is_crushed, downscale_factor


# ----------------- 可视化辅助 -----------------
def add_label(img, text_lines, color=(0, 255, 0)):
    h, w = img.shape[:2]
    scale = max(w, h) / 1000.0
    font_scale = max(0.8, 1.0 * scale)
    thickness = max(2, int(2 * scale))
    line_height = int(40 * scale)

    header_height = line_height * len(text_lines) + int(20 * scale)
    cv2.rectangle(img, (0, 0), (w, header_height), (0, 0, 0), -1)

    y = int(30 * scale)
    for line in text_lines:
        cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
        y += line_height
    return img


def get_crop_region(img_rgb, crop_size=256):
    """
    智能寻找主体区域：纹理梯度 + 中心偏置
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # 1. 纹理密度 (Sobel)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)

    # 2. 中心偏置 (Center Bias)
    Y, X = np.ogrid[:h, :w]
    center_y, center_x = h / 2, w / 2
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    center_weight = 1.0 - (dist_from_center / max_dist)
    center_weight = np.power(center_weight, 2)

    # 3. 结合
    saliency_map = magnitude * center_weight

    # 4. 平滑找最大值
    kernel_size = min(h, w) // 16  # 不需要太平滑，定位准一点
    if kernel_size % 2 == 0: kernel_size += 1
    density = cv2.GaussianBlur(saliency_map, (kernel_size, kernel_size), 0)

    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(density)
    cy, cx = maxLoc[1], maxLoc[0]

    # 5. 计算坐标
    half = crop_size // 2
    y1 = max(0, cy - half)
    y2 = min(h, y1 + crop_size)
    x1 = max(0, cx - half)
    x2 = min(w, x1 + crop_size)

    # 边界修正
    if y2 - y1 < crop_size:
        if y1 == 0:
            y2 = min(h, crop_size)
        else:
            y1 = max(0, h - crop_size)
    if x2 - x1 < crop_size:
        if x1 == 0:
            x2 = min(w, crop_size)
        else:
            x1 = max(0, w - crop_size)

    return int(y1), int(y2), int(x1), int(x2), int(cy), int(cx)


def main():
    print(f"🔥 Loading Metrics on {DEVICE}...")
    loss_fn_lpips = lpips.LPIPS(net='alex').to(DEVICE)
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

    print(f"🚀 Starting SMART comparison for {len(tasks)} images...")

    # 更新 CSV Header，加入 Crop 指标
    csv_header = [
        "Filename", "Target_KB", "Resolution",
        # Full Image Metrics
        "Ours_SSIM", "Ours_LPIPS", "Ours_NIQE",
        "JPEG_SSIM", "JPEG_LPIPS", "JPEG_NIQE", "JPEG_Scale",
        "WebP_SSIM", "WebP_LPIPS", "WebP_NIQE", "WebP_Scale",
        # Crop Metrics (Local Detail)
        "Ours_Crop_SSIM", "Ours_Crop_LPIPS", "Ours_Crop_NIQE",
        "JPEG_Crop_SSIM", "JPEG_Crop_LPIPS", "JPEG_Crop_NIQE",
        "WebP_Crop_SSIM", "WebP_Crop_LPIPS", "WebP_Crop_NIQE"
    ]

    with open(OUTPUT_METRICS_CSV, 'w', newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(csv_header)

        for row in tqdm(tasks):
            filename = row['Filename']
            target_kb = float(row['Payload (KB)'])

            # 1. Load Ours & GT
            our_path = os.path.join(OURS_DIR, filename)
            if not os.path.exists(our_path): continue
            our_bgr = cv2.imread(our_path)
            h, w = our_bgr.shape[:2]
            our_rgb = cv2.cvtColor(our_bgr, cv2.COLOR_BGR2RGB)

            gt_path = os.path.join(GT_DIR, filename)
            if not os.path.exists(gt_path): continue
            gt_bgr_raw = cv2.imread(gt_path)
            if gt_bgr_raw.shape[:2] != (h, w):
                gt_bgr = cv2.resize(gt_bgr_raw, (w, h), interpolation=cv2.INTER_LANCZOS4)
            else:
                gt_bgr = gt_bgr_raw
            gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
            gt_pil = Image.fromarray(gt_rgb)

            # 2. Strict Compression
            # JPEG
            jpg_buf, jpg_q, jpg_size, jpg_crushed, jpg_scale = compress_to_target_kb_strict(gt_pil, 'JPEG', target_kb)
            jpg_rgb_raw = np.array(Image.open(jpg_buf))
            if jpg_rgb_raw.shape[:2] != (h, w):
                jpg_rgb = cv2.resize(jpg_rgb_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                jpg_rgb = jpg_rgb_raw

            # WebP
            webp_buf, webp_q, webp_size, webp_crushed, webp_scale = compress_to_target_kb_strict(gt_pil, 'WEBP',
                                                                                                 target_kb)
            webp_rgb_raw = np.array(Image.open(webp_buf))
            if webp_rgb_raw.shape[:2] != (h, w):
                webp_rgb = cv2.resize(webp_rgb_raw, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                webp_rgb = webp_rgb_raw

            # ================= 3. 智能寻找 Crop 区域 =================
            # 使用 GT 来寻找最复杂的纹理区域 (作为 Ground Truth 的纹理参考)
            y1, y2, x1, x2, cy, cx = get_crop_region(gt_rgb, CROP_SIZE)

            # 提取 Crops (未放大的原始像素，用于计算指标)
            crop_gt = gt_rgb[y1:y2, x1:x2]
            crop_ours = our_rgb[y1:y2, x1:x2]
            crop_jpg = jpg_rgb[y1:y2, x1:x2]
            crop_webp = webp_rgb[y1:y2, x1:x2]

            # ================= 4. 计算指标 (全图 + 局部) =================
            # Full Image
            s_ours = calc_ssim(gt_rgb, our_rgb)
            l_ours = calc_lpips(loss_fn_lpips, gt_rgb, our_rgb, use_limit=True)
            n_ours = calc_niqe(niqe_fn, our_rgb, use_limit=True)

            s_jpg = calc_ssim(gt_rgb, jpg_rgb)
            l_jpg = calc_lpips(loss_fn_lpips, gt_rgb, jpg_rgb, use_limit=True)
            n_jpg = calc_niqe(niqe_fn, jpg_rgb, use_limit=True)

            s_webp = calc_ssim(gt_rgb, webp_rgb)
            l_webp = calc_lpips(loss_fn_lpips, gt_rgb, webp_rgb, use_limit=True)
            n_webp = calc_niqe(niqe_fn, webp_rgb, use_limit=True)

            # Crop Metrics (不使用 Limit，因为图很小)
            cs_ours = calc_ssim(crop_gt, crop_ours)
            cl_ours = calc_lpips(loss_fn_lpips, crop_gt, crop_ours, use_limit=False)
            cn_ours = calc_niqe(niqe_fn, crop_ours, use_limit=False)

            cs_jpg = calc_ssim(crop_gt, crop_jpg)
            cl_jpg = calc_lpips(loss_fn_lpips, crop_gt, crop_jpg, use_limit=False)
            cn_jpg = calc_niqe(niqe_fn, crop_jpg, use_limit=False)

            cs_webp = calc_ssim(crop_gt, crop_webp)
            cl_webp = calc_lpips(loss_fn_lpips, crop_gt, crop_webp, use_limit=False)
            cn_webp = calc_niqe(niqe_fn, crop_webp, use_limit=False)

            writer.writerow([
                filename, f"{target_kb:.2f}", f"{w}x{h}",
                # Full
                f"{s_ours:.4f}", f"{l_ours:.4f}", f"{n_ours:.4f}",
                f"{s_jpg:.4f}", f"{l_jpg:.4f}", f"{n_jpg:.4f}", f"{jpg_scale:.2f}",
                f"{s_webp:.4f}", f"{l_webp:.4f}", f"{n_webp:.4f}", f"{webp_scale:.2f}",
                # Crop
                f"{cs_ours:.4f}", f"{cl_ours:.4f}", f"{cn_ours:.4f}",
                f"{cs_jpg:.4f}", f"{cl_jpg:.4f}", f"{cn_jpg:.4f}",
                f"{cs_webp:.4f}", f"{cl_webp:.4f}", f"{cn_webp:.4f}"
            ])

            # ================= 5. 可视化 (PIP 画中画) =================
            def overlay_pip(full_img_rgb, crop_img_rgb, label_lines, color):
                # 1. 准备全图 (带标签)
                full_bgr = cv2.cvtColor(full_img_rgb, cv2.COLOR_RGB2BGR)
                full_bgr = add_label(full_bgr, label_lines, color)

                # 2. 准备放大图 (2倍放大)
                # 使用 Nearest Neighbor 放大，诚实展示马赛克
                zoom_h, zoom_w = CROP_SIZE * 2, CROP_SIZE * 2
                crop_zoom = cv2.resize(crop_img_rgb, (zoom_w, zoom_h), interpolation=cv2.INTER_NEAREST)
                crop_zoom_bgr = cv2.cvtColor(crop_zoom, cv2.COLOR_RGB2BGR)

                # 给放大图加个显眼的边框
                cv2.rectangle(crop_zoom_bgr, (0, 0), (zoom_w - 1, zoom_h - 1), (255, 255, 255), 4)
                cv2.rectangle(crop_zoom_bgr, (0, 0), (zoom_w - 1, zoom_h - 1), color, 2)

                # 3. 贴到右下角
                fh, fw = full_bgr.shape[:2]
                pad = 20
                y_off = fh - zoom_h - pad
                x_off = fw - zoom_w - pad

                if y_off > 0 and x_off > 0:
                    full_bgr[y_off:y_off + zoom_h, x_off:x_off + zoom_w] = crop_zoom_bgr

                # 4. 在原图上画出 Crop 的位置 (红色方框)
                cv2.rectangle(full_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3)

                return full_bgr

            # 准备四张图
            t_gt = overlay_pip(gt_rgb, crop_gt, ["Ground Truth"], (255, 255, 255))

            t_ours = overlay_pip(our_rgb, crop_ours, [
                f"Ours: {target_kb:.2f}KB",
                f"Full LPIPS:{l_ours:.3f}",
                f"Crop LPIPS:{cl_ours:.3f} (Focus)"
            ], (0, 255, 0))

            jpg_note = f" (Resized {jpg_scale * 100:.0f}%)" if jpg_crushed else ""
            t_jpg = overlay_pip(jpg_rgb, crop_jpg, [
                f"JPEG: {jpg_size:.2f}KB{jpg_note}",
                f"Full LPIPS:{l_jpg:.3f}",
                f"Crop LPIPS:{cl_jpg:.3f}"
            ], (0, 0, 255))

            webp_note = f" (Resized {webp_scale * 100:.0f}%)" if webp_crushed else ""
            t_webp = overlay_pip(webp_rgb, crop_webp, [
                f"WebP: {webp_size:.2f}KB{webp_note}",
                f"Full LPIPS:{l_webp:.3f}",
                f"Crop LPIPS:{cl_webp:.3f}"
            ], (0, 255, 255))

            # 拼合大图
            canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            canvas[0:h, 0:w] = t_gt
            canvas[0:h, w:w * 2] = t_ours
            canvas[h:h * 2, 0:w] = t_jpg
            canvas[h:h * 2, w:w * 2] = t_webp

            save_name = f"Compare_{filename}"
            cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), canvas)

    print(f"\n✅ All Done! Metrics saved to: {OUTPUT_METRICS_CSV}")
    print(f"🖼️  Comparisons with Smart Crop saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
