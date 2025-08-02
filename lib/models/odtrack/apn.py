# In lib/models/odtrack/apn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

# Import DoRA components
from ..dora import vision_transformer as vits
from ..dora.utils import MultiCropWrapper
from torchvision.transforms.functional import crop, resize, gaussian_blur


class DoRA_APN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.MODEL.APN.DORA_PATCH_SIZE
        # --- START OF MODIFICATION: Read cropping scheme from config ---
        self.cropping_scheme = getattr(cfg.MODEL.APN, "CROPPING_SCHEME", "center_of_mass")
        # --- END OF MODIFICATION ---

        dora_backbone = vits.__dict__[cfg.MODEL.APN.DORA_ARCH](patch_size=self.patch_size, num_classes=0)
        dino_head = vits.DINOHead(dora_backbone.embed_dim, out_dim=65536, use_bn=False, norm_last_layer=True)
        self.teacher = MultiCropWrapper(dora_backbone, dino_head)
        self._load_dora_weights(self.teacher, cfg.MODEL.APN.DORA_WEIGHTS_PATH)
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def _load_dora_weights(self, model, path_to_weights):
        print(f"Loading DoRA weights from: {path_to_weights}")
        state_dict = torch.load(path_to_weights, map_location="cpu")
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"DoRA weights loaded with msg: {msg}")

    # --- START OF MODIFICATION: New Helper Function for Center of Mass Cropping ---
    def _crop_center_of_mass(self, track_map_prob, H_t, W_t):
        B, H_s, W_s = track_map_prob.shape
        device = track_map_prob.device

        # Create coordinate grids
        x_coords = torch.linspace(0, W_s - 1, W_s, device=device)
        y_coords = torch.linspace(0, H_s - 1, H_s, device=device)
        yy_grid, xx_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')

        # Calculate Center of Mass (CoM)
        com_y = torch.sum(track_map_prob * yy_grid.unsqueeze(0), dim=[1, 2])
        com_x = torch.sum(track_map_prob * xx_grid.unsqueeze(0), dim=[1, 2])

        # Calculate Standard Deviation around CoM
        std_y = torch.sqrt(torch.sum(track_map_prob * (yy_grid.unsqueeze(0) - com_y.view(B, 1, 1))**2, dim=[1, 2]))
        std_x = torch.sqrt(torch.sum(track_map_prob * (xx_grid.unsqueeze(0) - com_x.view(B, 1, 1))**2, dim=[1, 2]))

        # Define window size based on std dev (e.g., 3 standard deviations wide)
        k = 2
        win_h = torch.round(k * std_y).long().clamp(min=H_t, max=H_s)
        win_w = torch.round(k * std_x).long().clamp(min=W_t, max=W_s)
        
        # Determine top-left corner of the box
        best_y = torch.round(com_y - win_h / 2).long().clamp(min=0)
        best_x = torch.round(com_x - win_w / 2).long().clamp(min=0)

        # Ensure the box doesn't go out of bounds
        best_y = torch.min(best_y, H_s - win_h)
        best_x = torch.min(best_x, W_s - win_w)

        # Calculate confidence by summing probability inside the determined box
        integral_map = torch.cumsum(torch.cumsum(track_map_prob, dim=-1), dim=-2)
        confidence = torch.zeros(B, device=device)
        for b in range(B):
            y1, x1 = best_y[b], best_x[b]
            y2, x2 = y1 + win_h[b], x1 + win_w[b]
            # Handle edge case where window is size 0
            if y1 >= y2 or x1 >= x2:
                confidence[b] = 0
                continue
            sum_val = integral_map[b, y2-1, x2-1]
            if y1 > 0: sum_val -= integral_map[b, y1-1, x2-1]
            if x1 > 0: sum_val -= integral_map[b, y2-1, x1-1]
            if y1 > 0 and x1 > 0: sum_val += integral_map[b, y1-1, x1-1]
            confidence[b] = sum_val

        return best_x, best_y, win_w, win_h, confidence
    # --- END OF NEW HELPER FUNCTION ---

    # --- START OF MODIFICATION: Helper Function for Sliding Window Cropping ---
    def _crop_sliding_window(self, track_map_prob, H_t, W_t):
        B, H_s, W_s = track_map_prob.shape
        win_h, win_w = H_t, W_t
        
        integral_map = torch.cumsum(torch.cumsum(track_map_prob, dim=-1), dim=-2)
        padded_integral_map = F.pad(integral_map, (1, 0, 1, 0))

        D = padded_integral_map[:, win_h:, win_w:]
        C = padded_integral_map[:, win_h:, :-win_w]
        B_val = padded_integral_map[:, :-win_h, win_w:]
        A = padded_integral_map[:, :-win_h, :-win_w]
        
        sum_map = D - B_val - C + A
        
        flat_sum_map = sum_map.view(B, -1)
        confidence, flat_idx = torch.max(flat_sum_map, dim=1)
        
        best_y = flat_idx // sum_map.shape[2]
        best_x = flat_idx % sum_map.shape[2]

        return best_x, best_y, torch.tensor([win_w]*B, device=track_map_prob.device), torch.tensor([win_h]*B, device=track_map_prob.device), confidence
    # --- END OF NEW HELPER FUNCTION ---

    def forward(self, template_imgs: List[torch.Tensor],
                search_frame: torch.Tensor,
                template_annos: List[torch.Tensor],
                return_viz=False):
        with torch.no_grad():
            B = search_frame.shape[0]
            all_prototypes, H_t, W_t = [], 0, 0
            for i in range(len(template_imgs)):
                template_img, template_anno = template_imgs[i], template_annos[i]

                _, template_patches, _, _, _ = self.teacher.backbone(template_img)
                template_patches = template_patches[:, 1:, :]

                H_t, W_t = template_img.shape[-2] // self.patch_size, template_img.shape[-1] // self.patch_size
                
                patch_mask = torch.zeros(B, H_t * W_t, 1, device=template_img.device)
                for b_idx in range(B):
                    gt_box = template_anno[b_idx]
                    x1, y1 = int(gt_box[0] * W_t), int(gt_box[1] * H_t)
                    x2, y2 = int((gt_box[0] + gt_box[2]) * W_t), int((gt_box[1] + gt_box[3]) * H_t)
                    for row in range(max(0, y1), min(H_t, y2)):
                        for col in range(max(0, x1), min(W_t, x2)):
                            patch_mask[b_idx, row * W_t + col, 0] = 1.0

                masked_features = template_patches * patch_mask
                num_patches_in_box = patch_mask.sum(dim=1)
                mean_masked_features = masked_features.sum(dim=1) / (num_patches_in_box + 1e-6)
                all_prototypes.append(mean_masked_features)

            stacked_prototypes = torch.stack(all_prototypes, dim=1)
            averaged_prototype = torch.mean(stacked_prototypes, dim=1, keepdim=True)

            _, _, _, _, search_key = self.teacher.backbone(search_frame)
            search_key_patches = search_key[:, :, 1:, :] # Remove CLS token for attention map

            # --- START OF MODIFICATION: Stable Head Dimension Calculation ---
            _, _, D = averaged_prototype.shape
            # Get num_heads directly from the loaded DoRA model architecture
            num_heads = self.teacher.backbone.num_heads
            head_dim = D // num_heads
            assert D % num_heads == 0, "Embedding dim not divisible by num_heads"
            # --- END OF MODIFICATION ---
            
            proto_reshaped = averaged_prototype.view(B, 1, num_heads, head_dim).permute(0,2,1,3)
            track_attn = (proto_reshaped @ search_key_patches.transpose(-2,-1)) * (head_dim ** -0.5)
            track_mask_raw = track_attn.mean(dim=1)
            
            H_s, W_s = search_frame.shape[-2] // self.patch_size, search_frame.shape[-1] // self.patch_size
            track_mask_raw_reshaped = track_mask_raw.view(B, 1, H_s, W_s)
            track_mask_blurred = gaussian_blur(track_mask_raw_reshaped, kernel_size=5)
            track_map_prob_1d = track_mask_blurred.view(B, -1).softmax(dim=-1)
            track_map_prob = track_map_prob_1d.view(B, H_s, W_s)

            # --- START OF MODIFICATION: Switchable Cropping Logic ---
            if self.cropping_scheme == 'sliding_window':
                best_x, best_y, win_w, win_h, confidence = self._crop_sliding_window(track_map_prob, H_t, W_t)
            else: # Default to 'center_of_mass'
                best_x, best_y, win_w, win_h, confidence = self._crop_center_of_mass(track_map_prob, H_t, W_t)
            # --- END OF MODIFICATION ---

            predicted_templates = []
            for b_idx in range(B):
                xmin_pix = best_x[b_idx].item() * self.patch_size
                ymin_pix = best_y[b_idx].item() * self.patch_size
                win_w_pix = win_w[b_idx].item() * self.patch_size
                win_h_pix = win_h[b_idx].item() * self.patch_size

                predicted_crop = crop(search_frame[b_idx], ymin_pix, xmin_pix, win_h_pix, win_w_pix)
                predicted_template = resize(predicted_crop, (self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE))
                predicted_templates.append(predicted_template)

            final_predicted_template = torch.stack(predicted_templates)

            if return_viz:
                upsampled_mask = F.interpolate(track_map_prob.unsqueeze(1), size=search_frame.shape[-2:], mode='bilinear')
                viz_data = {'upsampled_mask': upsampled_mask}
                return final_predicted_template, confidence, viz_data
            
            return final_predicted_template, confidence