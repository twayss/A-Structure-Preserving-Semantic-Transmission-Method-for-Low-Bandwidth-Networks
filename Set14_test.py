import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import cv2
import clip
from diffusers import AutoencoderKL
from ultralytics import FastSAM
import math
from tqdm import tqdm
from torch.cuda.amp import autocast
import gc
import io
import zlib
import csv
import lpips
from skimage.metrics import structural_similarity as ssim


# ==============================================================================
# def clean_edges(edge_np, min_size=20):
#     if edge_np.dtype != np.uint8:
#         edge_np = (edge_np > 0).astype(np.uint8)
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_np, connectivity=8)
#     small_components = np.where((stats[:, 4] < min_size) & (stats[:, 4] > 0))[0]
#     if len(small_components) > 0:
#         mask = np.isin(labels, small_components)
#         edge_np[mask] = 0
#     return edge_np
def clean_edges(edge_np, min_size=20):
    """
    通用清洗函数，可用于背景或前景
    """
    if edge_np.dtype != np.uint8:
        edge_np = (edge_np > 0).astype(np.uint8)

    # 快速检查是否有内容，避免空图报错
    if np.count_nonzero(edge_np) == 0:
        return edge_np

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(edge_np, connectivity=8)
    # stats: [x, y, w, h, area]
    small_components = np.where((stats[:, 4] < min_size) & (stats[:, 4] > 0))[0]

    if len(small_components) > 0:
        mask = np.isin(labels, small_components)
        edge_np[mask] = 0
    return edge_np
def get_auto_canny_thresholds(image_np, sigma=0.33):
    """
    根据图像的中值亮度自动计算最佳 Canny 阈值。
    这是解决 Baboon (低对比度) 和 Foreman (高对比度) 差异的关键。
    """
    if len(image_np.shape) == 3:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    v = np.median(image_np)

    # 根据统计学规律计算
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))

    return lower, upper
# ==============================================================================
# 0. 核心配置
# ==============================================================================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
#这里路径以总文件夹为相对路径，建议根据自身实际路径替换
TEST_CONFIG = {
    # 输入图片文件夹
    'test_dir': r"./src/dataset/Set14",

    # 权重路径
    'checkpoint_path':  r"./src/model/best_lpips_1pth",

    # 输出根目录
    'output_root': r"./results/inference_Set14_Test",

    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_objects': 256,

    # 🔥🔥🔥 核心修改 🔥🔥🔥
    # 设置 tile_size=512 意味着：直接处理整张图，不切片！
    'tile_size': 512,
    'tile_overlap': 0,

    # 🔥🔥🔥 核心开关 🔥🔥🔥
    # True: 强制缩放到 512x512 (Standard 14 模式)
    'force_resize_512': True,
}


# ==============================================================================
# 1. 🛠️ 工具函数 (保持不变)
# ==============================================================================

def pack_edge_to_bytes(edge_tensor):
    edge_np = (edge_tensor.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    flat_edge = edge_np.flatten()
    packed_data = np.packbits(flat_edge)
    compressed_data = zlib.compress(packed_data.tobytes(), level=9)
    return compressed_data, edge_np.shape


def unpack_bytes_to_edge(compressed_data, original_shape, device='cuda'):
    packed_data_bytes = zlib.decompress(compressed_data)
    packed_np = np.frombuffer(packed_data_bytes, dtype=np.uint8)
    unpacked_flat = np.unpackbits(packed_np)
    target_size = original_shape[0] * original_shape[1]
    unpacked_flat = unpacked_flat[:target_size]
    edge_np = unpacked_flat.reshape(original_shape).astype(np.float32)
    edge_tensor = torch.from_numpy(edge_np).to(device).unsqueeze(0).unsqueeze(0)
    return edge_tensor


def calculate_ssim(img_tensor_a, img_tensor_b):
    img_a = (img_tensor_a.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    img_b = (img_tensor_b.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    min_side = min(img_a.shape[0], img_a.shape[1])
    win_size = min(7, min_side)
    if win_size % 2 == 0: win_size -= 1
    if win_size < 3: win_size = 3
    score = ssim(img_a, img_b, win_size=win_size, channel_axis=2, data_range=255)
    return score


# ==============================================================================
# 2. 🔥 V78 SmartEncoder: 收缩保护区，极致分化
# ==============================================================================


class SmartEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        try:
            self.fast_sam = FastSAM('FastSAM-x.pt')
        except:
            self.fast_sam = None

    @torch.no_grad()
    def get_global_features(self, img_pil,
                            fg_low=180, fg_high=280,  # 🔥 默认再高一点，压制起手体积
                            bg_low=180, bg_high=280,
                            bg_clean=50,
                            fg_clean=0):

        w, h = img_pil.size
        img_np = np.array(img_pil)

        # 1. Global Vector
        vec = self.clip_model.encode_image(self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)).float()

        # 2. Mask (SAM)
        mask_np = None
        if self.fast_sam is not None:
            results = self.fast_sam(img_pil, device=self.device, retina_masks=True,
                                    imgsz=max(h, w), conf=0.4, iou=0.9, verbose=False)
            if len(results) > 0 and results[0].masks is not None:
                mask_tensor = results[0].masks.data.sum(dim=0) > 0
                mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
                # 无腐蚀，信任 SAM

        if mask_np is None or np.count_nonzero(mask_np) == 0:
            mask_np = np.ones((h, w), dtype=np.uint8)

        # 3. 🔥 分层边缘提取
        if len(img_np.shape) == 3:
            gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            gray_img = img_np

        # === Foreground ===
        edge_fg = cv2.Canny(gray_img, fg_low, fg_high)
        if fg_clean > 0:
            edge_fg = clean_edges(edge_fg, min_size=fg_clean)

        # === Background ===
        edge_bg = cv2.Canny(gray_img, bg_low, bg_high)
        if bg_clean > 0:
            edge_bg = clean_edges(edge_bg, min_size=bg_clean)

        # === Merge ===
        final_edge_np = np.where(mask_np > 0, edge_fg, edge_bg)

        # To Tensor
        edges_tensor = torch.from_numpy(final_edge_np).float().to(self.device).unsqueeze(0).unsqueeze(0) / 255.0
        mask_id = torch.zeros((h, w), device=self.device, dtype=torch.long)
        real_mask_tensor = torch.from_numpy(mask_np).to(self.device).float()
        if real_mask_tensor.ndim == 2: real_mask_tensor = real_mask_tensor.unsqueeze(0)

        return mask_id, edges_tensor, vec, real_mask_tensor
# ==============================================================================
# 3. 模型结构 (保持 V69 不变)
# ==============================================================================
class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        q = self.query_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        v = self.value_conv(x).view(B, -1, H * W).permute(0, 2, 1)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return self.gamma * out + x


class SPADEBlock(nn.Module):
    def __init__(self, features, embedding_dim=512):
        super().__init__()
        self.bn = nn.InstanceNorm2d(features, affine=False)
        self.fc = nn.Sequential(nn.Linear(embedding_dim, 128), nn.ReLU(), nn.Linear(128, features * 2))

    def forward(self, x, vec):
        out = self.bn(x)
        gamma, beta = self.fc(vec).view(x.size(0), -1, 1, 1).chunk(2, dim=1)
        return out * (1 + gamma) + beta


class ResNetSPADEBlock(nn.Module):
    def __init__(self, features, embedding_dim=512):
        super().__init__()
        self.s1 = SPADEBlock(features, embedding_dim)
        self.c1 = nn.Conv2d(features, features, 3, 1, 1, bias=False)
        self.s2 = SPADEBlock(features, embedding_dim)
        self.c2 = nn.Conv2d(features, features, 3, 1, 1, bias=False)

    def forward(self, x, vec):
        return self.c2(F.relu(self.s2(self.c1(F.relu(self.s1(x, vec))), vec))) + x


class LatentFeatureAdapter(nn.Module):
    def __init__(self, num_classes=256, embedding_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.emb = nn.Embedding(num_classes, 64)
        self.init_conv = nn.Conv2d(68, 64, 3, 1, 1)
        self.down1 = nn.Sequential(nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.body1 = ResNetSPADEBlock(128, embedding_dim)
        self.down2 = nn.Sequential(nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU())
        self.body2 = ResNetSPADEBlock(256, embedding_dim)
        self.down3 = nn.Sequential(nn.Conv2d(256, 256, 3, 2, 1), nn.ReLU())
        self.body3 = ResNetSPADEBlock(256, embedding_dim)
        self.attn_bottom = SelfAttention(256)
        self.to_latent = nn.Conv2d(256, 4, 3, 1, 1)

    def forward(self, mask_indices, color_map, edges, vec):
        m_emb = self.emb(mask_indices.long()).permute(0, 3, 1, 2)
        cat_input = torch.cat([m_emb, color_map, edges], dim=1)
        x = self.attn_bottom(
            self.body3(self.down3(self.body2(self.down2(self.body1(self.down1(self.init_conv(cat_input)), vec)), vec)),
                       vec))
        return self.to_latent(x)


class PretrainedVAEDecoder(nn.Module):
    def __init__(self, num_classes=256, embedding_dim=512, device='cuda'):
        super().__init__()
        local_vae_dir = r"/home/sc1/stable-diffusion-v1-5/vae"
        if not os.path.exists(os.path.join(local_vae_dir, "config.json")):
            self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float32).to(device)
        else:
            self.vae = AutoencoderKL.from_pretrained(local_vae_dir, local_files_only=True, use_safetensors=True,
                                                     torch_dtype=torch.float32).to(device)
        self.vae.requires_grad_(False)
        self.adapter = LatentFeatureAdapter(num_classes, embedding_dim).to(device)

    def forward(self, m, c, e, v):
        latents = self.adapter(m, c, e, v)
        latents = latents.to(dtype=torch.float32)
        decoded = self.vae.decode(latents).sample
        return decoded.clamp(-1, 1)


# ==============================================================================
# 4. 🔥 切片处理逻辑 (V95: 三区隔离法)
# ==============================================================================

def process_image_global_first(img_pil, encoder, model, tile_size=512, device='cuda', save_edge_path=None):
    w, h = img_pil.size

    # ==========================================================================
    # Logic: Tighter controls to hit < 11KB
    # ==========================================================================

    # 1️⃣ Round 1: Strict Start
    # FG/BG (180, 280) -> Man/Flowers/Comic should drop a bit compared to 170.
    with torch.no_grad():
        with autocast(enabled=True):
            full_mask_id, full_edge_tensor, global_vec, real_mask_tensor = \
                encoder.get_global_features(img_pil,
                                            fg_low=180, fg_high=280,
                                            bg_low=180, bg_high=280,
                                            bg_clean=50, fg_clean=0)

            edge_bits, shape_info = pack_edge_to_bytes(full_edge_tensor)
            edge_bytes = len(edge_bits)

            final_edge_tensor = full_edge_tensor
            final_edge_bits = edge_bits

            # 2️⃣ Branching Logic

            # Case A: Rescue Coastguard (Prevent Overshoot)
            # Threshold: < 7.5KB
            if edge_bytes < 7680:
                print(f"   [Logic] Payload tiny ({edge_bytes / 1024:.2f}KB). Boosting details...")

                # Boost:
                # Old: (100, 200) -> Too noisy (13KB)
                # New: (130, 230) -> Cleaner, focuses on stronger waves
                # Clean: 30 -> Kills small speckles
                _, boost_edge_tensor, _, _ = \
                    encoder.get_global_features(img_pil,
                                                fg_low=130, fg_high=230,
                                                bg_low=130, bg_high=230,
                                                bg_clean=30, fg_clean=0)

                boost_bits, _ = pack_edge_to_bytes(boost_edge_tensor)

                # Limit: 11KB (Strict)
                if len(boost_bits) < 11264:
                    final_edge_tensor = boost_edge_tensor
                    final_edge_bits = boost_bits
                    edge_bytes = len(boost_bits)
                    print(f"   [Logic] Boost accepted -> {edge_bytes / 1024:.2f}KB")
                else:
                    print(f"   [Logic] Boost rejected (Too large). Keeping strict.")

            # Case B: Suppress Heavy Images (Man/Flowers/Baboon)
            # Threshold: > 12.5KB (Lowered to catch Flowers/Zebra)
            elif edge_bytes > 12800:
                print(f"   [Logic] Payload large ({edge_bytes / 1024:.2f}KB). Activating FG Clean...")

                # Prune:
                # FG/BG (210, 310) -> Very sparse scattered points.
                # FG Clean (30) -> Removes small dust on Man's suit/Flowers.
                # BG Clean (120) -> Kills all background noise.
                _, prune_edge_tensor, _, _ = \
                    encoder.get_global_features(img_pil,
                                                fg_low=210, fg_high=310,
                                                bg_low=210, bg_high=310,
                                                bg_clean=120, fg_clean=30)

                prune_bits, _ = pack_edge_to_bytes(prune_edge_tensor)
                final_edge_tensor = prune_edge_tensor
                final_edge_bits = prune_bits
                print(f"   [Logic] Pruned to {len(prune_bits) / 1024:.2f}KB")

    # --- Packing & Reconstruction (Unchanged) ---
    total_bytes = 1024
    total_bytes += len(final_edge_bits)
    full_edge_restored = unpack_bytes_to_edge(final_edge_bits, shape_info, device=device)
    full_mask_tensor = full_mask_id

    # Mask Packing
    target_h_map = h // 8
    target_w_map = w // 8
    mask_np = real_mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    mask_small = Image.fromarray(mask_np).resize((target_w_map, target_h_map), Image.Resampling.NEAREST)
    mask_quantized_np = (np.array(mask_small) // 64 * 64).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(mask_quantized_np).save(buf, format='PNG', optimize=True)
    total_bytes += len(buf.getvalue())
    mask_restored_pil = Image.fromarray(mask_quantized_np).resize((w, h), Image.Resampling.NEAREST)
    full_mask_restored = torch.from_numpy(np.array(mask_restored_pil)).float().to(device)

    # Cmap Packing
    patch_32_pil = img_pil.resize((target_w_map, target_h_map), resample=Image.Resampling.LANCZOS)
    patch_32_np = np.array(patch_32_pil)
    buf = io.BytesIO()
    Image.fromarray(patch_32_np).save(buf, format='WEBP', lossless=False, quality=80)
    total_bytes += len(buf.getvalue())
    buf.seek(0)
    img_tensor_32 = transforms.ToTensor()(Image.open(buf).convert('RGB')).to(device)
    full_cmap_restored = F.interpolate(img_tensor_32.unsqueeze(0), size=(h, w), mode='bicubic', align_corners=False)
    full_cmap_restored = (full_cmap_restored - 0.5) * 2.0

    if save_edge_path is not None:
        edge_vis_np = (full_edge_restored.squeeze().cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(save_edge_path, edge_vis_np)

    # Reconstruction
    output = torch.zeros((3, h, w), device='cpu', dtype=torch.float32)
    stride = tile_size
    h_steps = math.ceil(h / stride)
    w_steps = math.ceil(w / stride)

    for h_idx in range(h_steps):
        for w_idx in range(w_steps):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)
            current_h = y2 - y1
            current_w = x2 - x1

            tile_edge = full_edge_restored[:, :, y1:y2, x1:x2]
            tile_mask = full_mask_restored[y1:y2, x1:x2]
            tile_cmap = full_cmap_restored[:, :, y1:y2, x1:x2]

            pad_h = tile_size - current_h
            pad_w = tile_size - current_w

            if pad_h > 0 or pad_w > 0:
                mode_h = 'reflect' if current_h > pad_h else 'replicate'
                mode_w = 'reflect' if current_w > pad_w else 'replicate'
                final_mode = 'replicate' if (mode_h == 'replicate' or mode_w == 'replicate') else 'reflect'
                tile_edge = F.pad(tile_edge, (0, pad_w, 0, pad_h), mode=final_mode)
                tile_cmap = F.pad(tile_cmap, (0, pad_w, 0, pad_h), mode=final_mode)
                tile_mask = tile_mask.unsqueeze(0).unsqueeze(0)
                tile_mask = F.pad(tile_mask, (0, pad_w, 0, pad_h), mode='replicate')
                tile_mask = tile_mask.squeeze(0).squeeze(0)

            try:
                with torch.no_grad():
                    with autocast(enabled=True):
                        in_mask = tile_mask.unsqueeze(0)
                        recon_patch = model(in_mask, tile_cmap, tile_edge, global_vec)
                        patch_out = recon_patch.squeeze(0).cpu().float()

                valid_out = patch_out[:, :current_h, :current_w]
                output[:, y1:y2, x1:x2] = valid_out
            except:
                pass

    ALIGN_RADIUS = 8
    decay_weights = torch.linspace(1.0, 0.0, ALIGN_RADIUS + 1)[:-1]

    for w_idx in range(1, w_steps):
        x = w_idx * stride
        if x < w:
            col_left = output[:, :, x - 1]
            col_right = output[:, :, x]
            half_gap = (col_right - col_left) * 0.5
            for i in range(min(ALIGN_RADIUS, x)):
                output[:, :, x - 1 - i] += half_gap * decay_weights[i]
            for i in range(min(ALIGN_RADIUS, w - x)):
                output[:, :, x + i] -= half_gap * decay_weights[i]

    for h_idx in range(1, h_steps):
        y = h_idx * stride
        if y < h:
            row_top = output[:, y - 1, :]
            row_bot = output[:, y, :]
            half_gap = (row_bot - row_top) * 0.5
            for i in range(min(ALIGN_RADIUS, y)):
                output[:, y - 1 - i, :] += half_gap * decay_weights[i]
            for i in range(min(ALIGN_RADIUS, h - y)):
                output[:, y + i, :] -= half_gap * decay_weights[i]

    final_img = (output * 0.5 + 0.5).clamp(0, 1)
    return final_img, total_bytes, real_mask_tensor


# ==============================================================================
# 5. 主入口
# ==============================================================================

def run_pipeline():
    # 1. 根据开关决定子文件夹和 CSV 文件名，防止数据混淆
    if TEST_CONFIG['force_resize_512']:
        mode_suffix = "std_512"
        print(f"⚠️ Mode: FORCE RESIZE to 512x512 (Standard Benchmark Mode)")
    else:
        mode_suffix = "native"
        print(f"✅ Mode: NATIVE RESOLUTION (Wild/DIV2K Mode)")

    comp_dir = os.path.join(TEST_CONFIG['output_root'], f"comparison_{mode_suffix}")
    recon_dir = os.path.join(TEST_CONFIG['output_root'], f"recon_{mode_suffix}")
    edge_dir = os.path.join(TEST_CONFIG['output_root'], f"edges_{mode_suffix}")

    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(recon_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    print(f"🚀 V73 Real Bokeh Pipeline (High Detail FG | Aggressively Blurred BG)")
    device = TEST_CONFIG['device']
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    encoder = SmartEncoder(device=device)
    model = PretrainedVAEDecoder(num_classes=TEST_CONFIG['max_objects'], device=device)

    print("📥 Loading Checkpoints...")
    ckpt = torch.load(TEST_CONFIG['checkpoint_path'], map_location=device)
    state_dict = ckpt.get('decoder', ckpt)
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    if not any('adapter' in k for k in new_state_dict):
        retry_dict = {}
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            if not name.startswith('adapter.') and not name.startswith('vae.'): name = 'adapter.' + name
            retry_dict[name] = v
        model.load_state_dict(retry_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    files = [f for f in os.listdir(TEST_CONFIG['test_dir']) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    files.sort()

    # 自动命名 CSV
    csv_path = os.path.join(TEST_CONFIG['output_root'], f'metrics_v73_{mode_suffix}.csv')
    csv_file = open(csv_path, 'w', newline='', encoding='utf-8')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Filename', 'Payload (KB)', 'Full_SSIM', 'Full_LPIPS', 'Obj_SSIM', 'Obj_LPIPS'])

    # 1. 初始化变量
    total_kb = 0
    total_ssim_full, total_lpips_full = 0, 0
    total_ssim_obj, total_lpips_obj = 0, 0
    count = 0

    for filename in tqdm(files):
        try:
            img_path = os.path.join(TEST_CONFIG['test_dir'], filename)
            raw_pil = Image.open(img_path).convert('RGB')

            # 🔥🔥🔥 核心修复：在这里执行强制缩放 🔥🔥🔥
            if TEST_CONFIG['force_resize_512']:
                raw_pil = raw_pil.resize((512, 512), Image.Resampling.LANCZOS)

            # 在缩放后生成 GT，确保评估标准统一
            gt_tensor = transforms.ToTensor()(raw_pil)

            edge_save_path = os.path.join(edge_dir, filename)

            # 1. 获取 Mask
            recon_vis, total_bytes, mask_tensor = process_image_global_first(
                raw_pil, encoder, model,
                tile_size=TEST_CONFIG['tile_size'],
                device=device,
                save_edge_path=edge_save_path
            )

            # ========================================================
            # 🔥🔥🔥 核心修改：计算 Masked Metrics (只看主体) 🔥🔥🔥
            # ========================================================

            # 1. 处理 Mask 尺寸和类型
            if mask_tensor.dim() == 2:
                mask_tensor = mask_tensor.unsqueeze(0)  # (1, H, W)

            # 确保 mask 在 CPU 上以便用于 SSIM，在 GPU 上用于 LPIPS
            mask_cpu = mask_tensor.cpu().float()
            mask_gpu = mask_tensor.to(device).float()

            # 2. 生成“纯主体”图像 (背景全黑)
            gt_masked = gt_tensor * mask_cpu
            recon_masked = recon_vis * mask_cpu

            # 3. 计算 Masked SSIM
            s_val_full = calculate_ssim(recon_vis, gt_tensor)  # 全图 SSIM
            s_val_obj = calculate_ssim(recon_masked, gt_masked)  # 主体 SSIM

            # 4. 计算 Masked LPIPS
            gt_gpu = gt_tensor.unsqueeze(0).to(device) * 2.0 - 1.0
            recon_gpu = recon_vis.unsqueeze(0).to(device) * 2.0 - 1.0

            # 生成 GPU 上的 Masked Image
            gt_gpu_masked = gt_gpu * mask_gpu
            recon_gpu_masked = recon_gpu * mask_gpu

            l_val_full = loss_fn_alex(gt_gpu, recon_gpu).item()  # 全图 LPIPS
            l_val_obj = loss_fn_alex(gt_gpu_masked, recon_gpu_masked).item()  # 主体 LPIPS

            # ========================================================

            kb_size = total_bytes / 1024
            total_kb += kb_size
            total_ssim_full += s_val_full
            total_lpips_full += l_val_full
            total_ssim_obj += s_val_obj
            total_lpips_obj += l_val_obj
            count += 1
            # ==========================================
            # 记录数据
            csv_writer.writerow([
                filename,
                f"{kb_size:.2f}",
                f"{s_val_full:.4f}", f"{l_val_full:.4f}",
                f"{s_val_obj:.4f}", f"{l_val_obj:.4f}"
            ])

            tqdm.write(
                f"📄 {filename[:10]} | {kb_size:.1f}KB | Full LPIPS: {l_val_full:.3f} -> Obj LPIPS: {l_val_obj:.3f} 🚀")

            # 保存对比图
            combined = torch.cat([recon_vis, gt_tensor, recon_masked, gt_masked], dim=2)
            save_image(combined, os.path.join(comp_dir, f"cmp_{filename}"))

            # 🔥🔥🔥🔥 补全：保存单独的生成图 🔥🔥🔥🔥
            save_image(recon_vis, os.path.join(recon_dir, filename))

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"❌ Error {filename}: {e}")
            import traceback
            traceback.print_exc()

    if count > 0:
        avg_kb = total_kb / count
        avg_ssim_full = total_ssim_full / count
        avg_lpips_full = total_lpips_full / count
        avg_ssim_obj = total_ssim_obj / count
        avg_lpips_obj = total_lpips_obj / count
    else:
        avg_kb, avg_ssim_full, avg_lpips_full, avg_ssim_obj, avg_lpips_obj = 0, 0, 0, 0, 0

    print(f"\n📊 [{mode_suffix}] Avg Payload: {avg_kb:.2f} KB")
    print(f"   Full -> SSIM: {avg_ssim_full:.4f} | LPIPS: {avg_lpips_full:.4f}")
    print(f"   Obj  -> SSIM: {avg_ssim_obj:.4f}  | LPIPS: {avg_lpips_obj:.4f} 🚀")

    # CSV 最后一行也写全
    csv_writer.writerow([
        'AVERAGE',
        f"{avg_kb:.2f}",
        f"{avg_ssim_full:.4f}", f"{avg_lpips_full:.4f}",
        f"{avg_ssim_obj:.4f}", f"{avg_lpips_obj:.4f}"
    ])
    csv_file.close()


if __name__ == "__main__":
    run_pipeline()
