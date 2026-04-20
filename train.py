import os

# ==============================================================================
# 0. 环境调优
# ==============================================================================
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.utils import save_image
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import csv
import re
import clip
import torch.nn.utils as utils

# [LPIPS]
try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("⚠️ 缺少 lpips 库，将跳过 LPIPS 计算。")

try:
    from torchmetrics.functional import structural_similarity_index_measure as ssim_metric
except ImportError:
    try:
        from torchmetrics.functional import ssim as ssim_metric
    except ImportError:
        print("⚠️ 无法导入 SSIM，请检查 torchmetrics 版本")

try:
    from diffusers import AutoencoderKL
    from ultralytics import FastSAM
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
    print("❌ 缺少必要库，请检查安装。")
    exit(1)

# ==============================================================================
# ⚙️ 全局配置
# ==============================================================================
MAX_OBJECTS = 256


def get_color_palette():
    np.random.seed(42)
    return np.random.randint(0, 255, size=(MAX_OBJECTS, 3)).flatten().tolist()


# ==============================================================================
# 1. 判别器
# ==============================================================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.block0 = nn.Sequential(utils.spectral_norm(nn.Conv2d(in_channels, 64, 4, 2, 1)), nn.LeakyReLU(0.2, True))
        self.block1 = nn.Sequential(utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.InstanceNorm2d(128),
                                    nn.LeakyReLU(0.2, True))
        self.block2 = nn.Sequential(utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.InstanceNorm2d(256),
                                    nn.LeakyReLU(0.2, True))
        self.block3 = nn.Sequential(utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), nn.InstanceNorm2d(512),
                                    nn.LeakyReLU(0.2, True))
        self.last_conv = utils.spectral_norm(nn.Conv2d(512, 1, 3, 1, 1))

    def forward(self, x):
        f0 = self.block0(x)
        f1 = self.block1(f0)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        out = self.last_conv(f3)
        return out, [f0, f1, f2, f3]


# ==============================================================================
# 2. SmartEncoder
# ==============================================================================
class SmartEncoder(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        print("📥 [SmartEncoder] 初始化模型中...")

        self.fast_sam = FastSAM('FastSAM-x.pt')
        self.sam_checkpoint = "sam_vit_b_01ec64.pth"
        try:
            if not os.path.exists(self.sam_checkpoint): raise FileNotFoundError
            self.sam = sam_model_registry["vit_b"](checkpoint=self.sam_checkpoint)
        except:
            print("⚠️ 切换到 MobileSAM (vit_t)...")
            self.sam_checkpoint = "mobile_sam.pt"
            self.sam = sam_model_registry["vit_t"](checkpoint=self.sam_checkpoint)

        self.sam.to(device=device)
        self.sam_predictor = SamPredictor(self.sam)
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)

    @torch.no_grad()
    def forward(self, img_pil):
        w, h = img_pil.size
        # 1. 准备基础数据
        img_tensor = transforms.ToTensor()(img_pil).to(self.device)
        mask_id = torch.zeros((h, w), device=self.device, dtype=torch.long)

        # 🔥🔥🔥 核心修改：Low-Res GT + 量化模拟 🔥🔥🔥
        # A. 下采样到 32x32 (模拟通信压缩)
        low_res = F.interpolate(img_tensor.unsqueeze(0), size=(32, 32), mode='bilinear', align_corners=False)
        # B. ⚡️ 模拟 int8 量化精度 (0~1 范围)
        low_res = (low_res * 255.0).round() / 255.0
        # C. 上采样回原图尺寸 (得到 0~1 范围的渐变图)
        color_map = F.interpolate(low_res, size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        # 🔥🔥🔥 修改结束 🔥🔥🔥

        # === SAM 仅用于生成 Mask ID ===
        fast_res = self.fast_sam(img_pil, device=self.device, retina_masks=False, imgsz=512,
                                 conf=0.15, iou=0.7, verbose=False, agnostic_nms=False)
        boxes = None
        if len(fast_res) > 0 and fast_res[0].boxes is not None:
            boxes = fast_res[0].boxes.xyxy

        if boxes is not None and len(boxes) > 0:
            img_np = np.array(img_pil)
            self.sam_predictor.set_image(img_np)
            transformed_boxes = self.sam_predictor.transform.apply_boxes_torch(boxes, img_np.shape[:2])
            masks, _, _ = self.sam_predictor.predict_torch(
                point_coords=None, point_labels=None, boxes=transformed_boxes, multimask_output=False
            )
            m = masks.squeeze(1)
            areas = m.sum(dim=(1, 2))
            sorted_indices = torch.argsort(areas, descending=True)

            for i in range(len(sorted_indices)):
                real_idx = sorted_indices[i]
                mask_bool = m[real_idx]
                if mask_bool.sum() > 0:
                    current_id = (i % (MAX_OBJECTS - 1)) + 1
                    mask_id[mask_bool] = current_id
                    # 🛑 这里绝不能用 avg_color 覆盖 color_map！

        # === 边缘和向量提取 ===
        gray_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
        edges = torch.from_numpy(cv2.Canny(gray_img, 100, 200)).float().to(self.device).unsqueeze(0) / 255.0
        vec = self.clip_model.encode_image(self.clip_preprocess(img_pil).unsqueeze(0).to(self.device)).float()

        # ✅ 最后统一归一化到 [-1, 1]
        color_map = (color_map - 0.5) * 2.0

        return mask_id, color_map, edges, vec


# ==============================================================================
# 3. 网络结构
# ==============================================================================
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
    def __init__(self, num_classes=MAX_OBJECTS, embedding_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.emb = nn.Embedding(num_classes, 64)
        self.init_conv = nn.Conv2d(68, 64, 3, 1, 1)  # 64(Emb) + 3(Color) + 1(Edge)
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


class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W * H)
        out = torch.bmm(self.value_conv(x).view(B, -1, W * H),
                        self.softmax(torch.bmm(proj_query, proj_key)).permute(0, 2, 1))
        return self.gamma * out.view(B, C, H, W) + x


class PretrainedVAEDecoder(nn.Module):
    def __init__(self, num_classes=MAX_OBJECTS, embedding_dim=512, device='cuda'):
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16).to(device)
        self.vae.requires_grad_(False)
        self.vae.enable_slicing()
        self.adapter = LatentFeatureAdapter(num_classes, embedding_dim).to(device)

    def forward(self, m, c, e, v):
        latents = self.adapter(m, c, e, v)
        return self.vae.decode(latents).sample.clamp(-1, 1)


# ==============================================================================
# 4. Loss
# ==============================================================================
class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg19(weights='DEFAULT').features
        self.slice1 = nn.Sequential(*list(vgg.children())[:4]).eval().to(device)
        self.slice2 = nn.Sequential(*list(vgg.children())[4:9]).eval().to(device)
        self.slice3 = nn.Sequential(*list(vgg.children())[9:35]).eval().to(device)
        for p in self.parameters(): p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def gram_matrix(self, input):
        a, b, c, d = input.size()
        features = input.view(a, b, c * d)
        G = torch.bmm(features, features.transpose(1, 2))
        return G.div(b * c * d)

    def forward(self, x, y):
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float();
            y = y.float()
            if torch.isnan(x).any(): x = torch.nan_to_num(x)
            if torch.isnan(y).any(): y = torch.nan_to_num(y)
            x = (x - self.mean) / self.std
            y = (y - self.mean) / self.std
            x_h1, y_h1 = self.slice1(x), self.slice1(y)
            x_h2, y_h2 = self.slice2(x_h1), self.slice2(y_h1)
            x_h3, y_h3 = self.slice3(x_h2), self.slice3(y_h2)
            loss_content = F.l1_loss(x_h1, y_h1) * 0.1 + F.l1_loss(x_h2, y_h2) * 0.25 + F.l1_loss(x_h3, y_h3) * 1.0
            loss_style = F.l1_loss(self.gram_matrix(x_h1), self.gram_matrix(y_h1)) * 0.1 + \
                         F.l1_loss(self.gram_matrix(x_h2), self.gram_matrix(y_h2)) * 0.2 + \
                         F.l1_loss(self.gram_matrix(x_h3), self.gram_matrix(y_h3)) * 0.5
            return loss_content + (loss_style * 30.0)


class EdgeLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        k_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        k_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
        self.k_x = k_x;
        self.k_y = k_y

    def get_gradients(self, img):
        if img.shape[1] == 3: img = img.mean(dim=1, keepdim=True)
        gx = F.conv2d(img, self.k_x, padding=1)
        gy = F.conv2d(img, self.k_y, padding=1)
        return torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)

    def forward(self, pred, target):
        return F.l1_loss(self.get_gradients(pred), self.get_gradients(target))


class SSIMLoss(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()

    def forward(self, pred, target):
        pred_norm = (pred * 0.5 + 0.5).clamp(0, 1)
        target_norm = (target * 0.5 + 0.5).clamp(0, 1)
        ssim_val = ssim_metric(pred_norm, target_norm, data_range=1.0)
        return 1.0 - ssim_val


# ==============================================================================
# 5. Trainer
# ==============================================================================
class ProjectTrainer:
    def __init__(self, config, resume_checkpoint=None):
        self.cfg = config
        self.device = config['device']
        self.ckpt_dir = os.path.join(config['result_dir'], 'checkpoints2')
        for d in [config['result_dir'], config['process_dir'], self.ckpt_dir]: os.makedirs(d, exist_ok=True)
        self.log_file = os.path.join(config['result_dir'], 'training_log.csv')
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w') as f:
                f.write("Epoch,G_Loss,D_Loss,SSIM,LPIPS\n")

        self.decoder = PretrainedVAEDecoder(num_classes=MAX_OBJECTS, device=self.device)
        self.netD = Discriminator().to(self.device)

        self.optG = optim.Adam(self.decoder.adapter.parameters(), lr=1e-6, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.netD.parameters(), lr=2e-6, betas=(0.5, 0.999))
        self.scaler = GradScaler()
        self.schedulerG = optim.lr_scheduler.CosineAnnealingLR(self.optG, T_max=400, eta_min=1e-7)
        self.schedulerD = optim.lr_scheduler.CosineAnnealingLR(self.optD, T_max=850, eta_min=1e-7)
        # self.schedulerG = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optG, T_0=50, T_mult=1, eta_min=1e-7)
        # self.schedulerD = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optD, T_0=50, T_mult=1, eta_min=1e-7)

        self.start_epoch, self.best_lpips = 0, 1.0
        self.l1_loss, self.vgg_loss, self.gan_loss = nn.L1Loss(), VGGLoss(self.device), nn.MSELoss()
        self.edge_loss = EdgeLoss(self.device)
        self.ssim_loss = SSIMLoss(self.device)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=False).to(
            self.device) if LPIPS_AVAILABLE else None
        self.palette = get_color_palette()

        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"💉 准备移植旧权重: {resume_checkpoint}")
            ckpt = torch.load(resume_checkpoint, map_location=self.device)
            old_state = ckpt.get('decoder', {})
            new_state = self.decoder.state_dict()
            filtered_state = {k: v for k, v in old_state.items() if k in new_state and v.shape == new_state[k].shape}
            print(f"✅ 成功继承层数: {len(filtered_state)} / {len(new_state)}")
            if len(filtered_state) > 0:
                new_state.update(filtered_state)
                self.decoder.load_state_dict(new_state, strict=False)
                if 'netD' in ckpt:
                    self.netD.load_state_dict(ckpt['netD'])
                    print("✅ 判别器 (D) 权重已恢复 (避免训练崩盘)")
                else:
                    print("⚠️ 警告: 未找到判别器权重，D 将被重置！(如果是新训练任务可忽略)")

            if 'epoch' in ckpt:
                self.start_epoch = ckpt['epoch'] + 1
                print(f"📅 进度恢复成功: 将从 Epoch {self.start_epoch} 继续训练！")
            else:
                self.start_epoch = 0

    def get_std_mean(self, x):
        B, C, H, W = x.shape
        return x.view(B, C, -1).std(dim=2), x.view(B, C, -1).mean(dim=2)

    def run_epoch(self, loader, is_train=True, ep=0):
        if is_train:
            self.decoder.train();
            self.netD.train()
        else:
            self.decoder.eval();
            self.netD.eval()

        m_lossG, m_lossD, m_ssim, m_lpips = 0, 0, 0, 0
        last_d_loss = 0.0
        sample_data = None
        target_idx = ep % len(loader) if len(loader) > 0 else 0

        if not is_train: torch.cuda.empty_cache()
        context = torch.enable_grad() if is_train else torch.no_grad()

        with context:
            for i, (mask, cmap, edge, vec, gt) in enumerate(loader):
                mask, cmap, edge, vec, gt = [x.to(self.device) for x in [mask, cmap, edge, vec, gt]]

                should_train_d = is_train and (i % 5 == 0)  # D 频率

                if should_train_d:
                    self.netD.train()
                    self.optD.zero_grad()
                    with autocast(enabled=True):
                        with torch.no_grad():
                            vec_noisy = vec + torch.randn_like(vec) * 0.05
                            recon_d = self.decoder(mask, cmap, edge, vec_noisy)
                        pred_real_d, _ = self.netD(gt)
                        loss_real = self.gan_loss(pred_real_d.float(),
                                                  torch.tensor(0.9).expand_as(pred_real_d).to(self.device))
                        pred_fake_d, _ = self.netD(recon_d.detach())
                        loss_fake = self.gan_loss(pred_fake_d.float(),
                                                  torch.tensor(0.1).expand_as(pred_fake_d).to(self.device))
                        loss_D_total = (loss_real + loss_fake) * 0.5
                    self.scaler.scale(loss_D_total).backward()
                    self.scaler.step(self.optD)
                    self.scaler.update()
                    last_d_loss = loss_D_total.item()
                    m_lossD += last_d_loss
                else:
                    m_lossD += last_d_loss

                if is_train: self.optG.zero_grad()
                with autocast(enabled=True):
                    vec_noisy = vec + torch.randn_like(vec) * 0.05 if is_train else vec
                    recon = self.decoder(mask, cmap, edge, vec_noisy)
                    recon_f, gt_f = recon.float(), gt.float()

                    if is_train: self.netD.eval()
                    pred_fake, feat_fake = self.netD(recon)
                    with torch.no_grad():
                        _, feat_real = self.netD(gt)
                    if is_train: self.netD.train()

                    loss_fm = 0
                    for f_f, f_r in zip(feat_fake, feat_real): loss_fm += F.l1_loss(f_f, f_r.detach())
                    loss_fm = loss_fm * 2.0
                    loss_gan = self.gan_loss(pred_fake.float(), torch.ones_like(pred_fake).float()) * 20.0
                    loss_vgg = self.vgg_loss(recon_f, gt_f) * 1.0
                    loss_lpips = self.lpips_metric(recon_f, gt_f) * 190.0 if LPIPS_AVAILABLE else 0
                    loss_color_total = (
                                F.l1_loss(F.interpolate(recon_f, (32, 32)), F.interpolate(gt_f, (32, 32))) * 0.8 +
                                F.l1_loss(F.interpolate(recon_f, (64, 64)), F.interpolate(gt_f, (64, 64))) * 0.2)
                    r_std, r_mean = self.get_std_mean(recon_f)
                    g_std, g_mean = self.get_std_mean(gt_f)
                    loss_contrast = (F.mse_loss(r_std, g_std) + F.mse_loss(r_mean, g_mean)) * 8.0
                    loss_edge = self.edge_loss(recon_f, gt_f) * 4.0

                    losses = {"GAN": loss_gan, "FM": loss_fm, "VGG": loss_vgg, "LPIPS": loss_lpips,
                              "Color": loss_color_total, "Contrast": loss_contrast, "Edge": loss_edge}
                    for name, val in losses.items():
                        if torch.isnan(val): print(f"🚨 Loss [{name}] is NaN!")
                    loss_G_total = loss_gan + loss_fm + loss_vgg + loss_lpips + loss_color_total + loss_contrast + loss_edge

                if is_train:
                    self.scaler.scale(loss_G_total).backward()
                    self.scaler.unscale_(self.optG)
                    torch.nn.utils.clip_grad_norm_(self.decoder.adapter.parameters(), 1.0)
                    self.scaler.step(self.optG)
                    self.scaler.update()
                    m_lossG += loss_G_total.item()

                if not is_train:
                    m_lossG += loss_G_total.item()
                    if self.lpips_metric: m_lpips += self.lpips_metric(recon_f, gt_f).item()
                    try:
                        recon_vis = (recon_f * 0.5 + 0.5).clamp(0, 1)
                        gt_vis = (gt_f * 0.5 + 0.5).clamp(0, 1)
                        m_ssim += ssim_metric(recon_vis, gt_vis, data_range=1.0).item()
                    except:
                        pass

                if not is_train and i == target_idx:
                    latent = self.decoder.adapter(mask[0:1], cmap[0:1], edge[0:1], vec[0:1])[0]
                    # 🔥 更新：把 cmap[0] 也传出去，用于画图
                    sample_data = (mask[0], edge[0], recon[0].float(), gt[0], latent.float(), cmap[0])

        n = len(loader) if len(loader) > 0 else 1
        return m_lossG / n, m_lossD / n, m_ssim / n, m_lpips / n, sample_data

    def fit(self, t_loader, v_loader, epochs):
        print(f"🚀 开始训练 (Embedding 增强版) | Batch Size: {self.cfg['batch_size']}")
        for ep in range(self.start_epoch, epochs):
            t_lg, t_ld, _, _, _ = self.run_epoch(t_loader, True, ep)
            v_lg, v_ld, v_s, v_p, sample = self.run_epoch(v_loader, False, ep)
            print(f"📊 Ep {ep} | G: {t_lg:.3f} | D: {t_ld:.3f} | SSIM: {v_s:.3f} | LPIPS: {v_p:.4f}")
            try:
                with open(self.log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([ep, t_lg, t_ld, v_s, v_p])
            except Exception as e:
                print(f"⚠️ 写入日志失败: {e}")
            self.schedulerG.step();
            self.schedulerD.step()

            if ep % 1 == 0:
                print(f"📉 Current LR: {self.optG.param_groups[0]['lr']:.2e}")
                state = {'decoder': self.decoder.state_dict(), 'netD': self.netD.state_dict(), 'epoch': ep}
                torch.save(state, os.path.join(self.ckpt_dir, "latest.pth"))
                if LPIPS_AVAILABLE and v_p < self.best_lpips:
                    self.best_lpips = v_p;
                    torch.save(state, os.path.join(self.ckpt_dir, "best_lpips.pth"))
            # *sample 自动解包 6 个参数
            if sample: self.save_art(ep, *sample)

    def save_art(self, ep, mask, edge, recon, gt, latent, cmap):  # 👈 新增 cmap 参数
        # 1. Mask 可视化
        m_pil = Image.fromarray(mask.cpu().numpy().astype(np.uint8)).convert("P")
        m_pil.putpalette(self.palette)
        m_vis = F.interpolate(transforms.ToTensor()(m_pil.convert("RGB")).unsqueeze(0), (256, 256),
                              mode='nearest').squeeze(0).to(self.device)
        # 2. Edge 可视化
        e_vis = F.interpolate(edge.unsqueeze(0).expand(-1, 3, -1, -1), (256, 256), mode='bilinear').squeeze(0)

        # 3. 🔥 Color Map 可视化 (直观展示低清输入)
        # cmap 是 [-1, 1] 的，转成 [0, 1]
        c_vis = (cmap * 0.5 + 0.5).clamp(0, 1)
        # 它是 SmartEncoder 插值后的 256x256，这里直接用即可，不需要再插值

        # 4. 结果和原图
        recon_vis = (recon * 0.5 + 0.5).clamp(0, 1)
        gt_vis = (gt * 0.5 + 0.5).clamp(0, 1)

        # 📸 拼接 5 张图：Mask | Edge | ColorMap | Recon | GT
        save_image(torch.cat([m_vis, e_vis, c_vis, recon_vis, gt_vis], dim=2),
                   os.path.join(self.cfg['process_dir'], f'epoch_{ep}_cmp.jpg'))


# ==============================================================================
# 6. Dataset
# ==============================================================================
class LoRACommunicationDataset(Dataset):
    def __init__(self, hr_dir, encoder, img_size=256, mode="Train"):
        self.img_size = img_size
        self.transform = transforms.RandomCrop(img_size) if mode == "Train" else transforms.CenterCrop(img_size)
        self.files = [os.path.join(hr_dir, f) for f in os.listdir(hr_dir) if
                      f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        self.encoder = encoder

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            img_pil = self.transform(Image.open(self.files[idx]).convert('RGB'))
            m, cmap, e, v = self.encoder(img_pil)
            gt_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])(img_pil)
            return m.cpu(), cmap.cpu(), e.cpu(), v.cpu().squeeze(0), gt_tensor
        except:
            return self.__getitem__(np.random.randint(0, len(self.files)))


# ==============================================================================
# 7. 🛠️ 传输数据量核算函数 (单独调用)
# ==============================================================================
def analyze_payload_size(encoder, dataset, save_dir, sample_count=50):
    """
    随机抽取 sample_count 张图，计算平均传输大小、最大值、最小值
    """
    print(f"\n📊 [Payload Statistics] 正在核算数据集传输量 (采样数: {sample_count})...")
    os.makedirs(save_dir, exist_ok=True)

    # 用于统计
    stats = {
        'mask': [], 'color': [], 'edge': [], 'vec': [], 'total': []
    }

    # 随机采样索引
    indices = np.random.choice(len(dataset), size=min(len(dataset), sample_count), replace=False)

    # 临时文件路径 (循环覆盖，不占空间)
    tmp_mask = os.path.join(save_dir, "temp_mask.png")
    tmp_color = os.path.join(save_dir, "temp_color.jpg")
    tmp_edge = os.path.join(save_dir, "temp_edge.png")
    tmp_vec = os.path.join(save_dir, "temp_vec.npy")

    for i, idx in enumerate(indices):
        # 打印进度条
        print(f"\r   Processing: {i + 1}/{len(indices)}", end="")

        try:
            img_pil = dataset.transform(Image.open(dataset.files[idx]).convert('RGB'))
            with torch.no_grad():
                mask_id, color_map, edges, vec = encoder(img_pil)

            # 1. Mask
            mask_np = mask_id.cpu().numpy().astype(np.uint8)
            Image.fromarray(mask_np).save(tmp_mask, optimize=True)
            s_mask = os.path.getsize(tmp_mask) / 1024

            # 2. Color (32x32)
            cmap_32 = F.interpolate(color_map.unsqueeze(0), size=(32, 32), mode='bilinear',
                                    align_corners=False).squeeze(0)
            cmap_np = ((cmap_32.permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
            Image.fromarray(cmap_np).save(tmp_color, quality=80)
            s_color = os.path.getsize(tmp_color) / 1024

            # 3. Edge (1-bit Binary)
            edge_raw = (edges.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            _, edge_bin = cv2.threshold(edge_raw, 100, 255, cv2.THRESH_BINARY)
            Image.fromarray(edge_bin).convert('1').save(tmp_edge, optimize=True)
            s_edge = os.path.getsize(tmp_edge) / 1024

            # 4. Vector
            vec_np = vec.cpu().numpy().astype(np.float16)
            np.save(tmp_vec, vec_np)
            s_vec = os.path.getsize(tmp_vec) / 1024

            # 记录
            stats['mask'].append(s_mask)
            stats['color'].append(s_color)
            stats['edge'].append(s_edge)
            stats['vec'].append(s_vec)
            stats['total'].append(s_mask + s_color + s_edge + s_vec)

        except Exception as e:
            print(f" (跳过坏图: {e})", end="")

    print("\n\n" + "=" * 50)
    print(f"📦 Payload 统计报告 (基于 {len(stats['total'])} 张样本)")
    print("=" * 50)

    def print_stat(name, data):
        avg = np.mean(data)
        std = np.std(data)
        max_v = np.max(data)
        min_v = np.min(data)
        print(f"   {name:<15}: Avg {avg:5.2f} KB | Std ±{std:4.2f} | Range [{min_v:4.2f} - {max_v:4.2f}]")

    print_stat("🖼️  Mask", stats['mask'])
    print_stat("✏️  Edge", stats['edge'])
    print_stat("🎨 Color", stats['color'])
    print_stat("🧠 Vector", stats['vec'])
    print("-" * 50)
    print_stat("💰 TOTAL", stats['total'])
    print("=" * 50 + "\n")

# ==============================================================================
# Main
# ==============================================================================
if __name__ == '__main__':
    # ⚠️ 请确保这里指向你的旧权重文件
    RESUME_PATH = r"./src/model/latest.pth"
    #这里路径以总文件夹为相对路径，建议根据自身实际路径替换
    CONFIG = {
        'train_dir': r"./src/dataset/DIV2K_train_HR",
        'test_dir': r"./src/dataset/DIV2K_valid_HR",
        'result_dir': r"./results",#存放结果的文件夹
        'process_dir': r"./results/process",
        'img_size': 256,
        'batch_size': 8,
        'epochs': 1000,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'resume_path': r"./results/checkpoints2/latest.pth"#新的模型权重文件
    }

    if not os.path.exists(CONFIG['train_dir']): exit(1)

    global_encoder = SmartEncoder(device=CONFIG['device'])
    t_set = LoRACommunicationDataset(CONFIG['train_dir'], global_encoder, CONFIG['img_size'], "Train")
    v_set = LoRACommunicationDataset(CONFIG['test_dir'], global_encoder, CONFIG['img_size'], "Test")
    analyze_payload_size(global_encoder, t_set,
                         os.path.join(CONFIG['result_dir'], "payload_check"),
                         sample_count=50)
    # 🔥🔥🔥 1. 在训练前，先核算一下数据大小 (保存到 payload_check 文件夹) 🔥🔥🔥
    analyze_payload_size(global_encoder, t_set, os.path.join(CONFIG['result_dir'], "payload_check"))

    t_loader = DataLoader(t_set, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    v_loader = DataLoader(v_set, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    ProjectTrainer(CONFIG, resume_checkpoint=CONFIG['RESUME_PATH']).fit(t_loader, v_loader, CONFIG['epochs'])
    #resume_checkpoint可根据需要设置none或者继承旧权重
