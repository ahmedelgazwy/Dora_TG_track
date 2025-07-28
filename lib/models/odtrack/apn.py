# In lib/models/odtrack/apn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# Import DoRA components
from ..dora import vision_transformer as vits
from ..dora.utils import MultiCropWrapper
from ..dora.sinkhorn_knopp import SinkhornKnopp
from einops import rearrange
from torchvision.transforms.functional import crop, resize


class DoRA_APN(nn.Module):
    """
    Appearance Prediction Network based on DoRA.
    Uses a pre-trained, frozen DoRA teacher model to track an object
    from a template image into a search frame.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_size = cfg.MODEL.APN.DORA_PATCH_SIZE
        self.num_queries = cfg.MODEL.APN.NUM_OBJECT_QUERIES

        dora_backbone = vits.__dict__[cfg.MODEL.APN.DORA_ARCH](patch_size=self.patch_size, num_classes=0)
        dino_head = vits.DINOHead(dora_backbone.embed_dim, out_dim=65536, use_bn=False, norm_last_layer=True)
        self.teacher = MultiCropWrapper(dora_backbone, dino_head)

        self._load_dora_weights(self.teacher, cfg.MODEL.APN.DORA_WEIGHTS_PATH)

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        self.sinkhorn_knopp = SinkhornKnopp(num_iters_sk=3, epsilon_sk=0.05)

    def _load_dora_weights(self, model, path_to_weights):
        print(f"Loading DoRA weights from: {path_to_weights}")
        state_dict = torch.load(path_to_weights, map_location="cpu")
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"DoRA weights loaded with msg: {msg}")

    # --- START OF MODIFICATION ---
    def forward(self, last_template_img, search_frame, last_template_anno, return_viz=False):
    # --- END OF MODIFICATION ---
        """
        Predicts the appearance of the object from `last_template_img` in the `search_frame`.
        """
        with torch.no_grad():
            _, template_patches, template_attn, template_query, _ = self.teacher.backbone(last_template_img)
            
            H_t, W_t = last_template_img.shape[-2] // self.patch_size, last_template_img.shape[-1] // self.patch_size
            
            cls_attn_heads = template_attn[:, :, 0, 1:]
            
            num_heads_to_use = min(self.num_queries, cls_attn_heads.shape[1])
            chosen_heads = random.sample(range(cls_attn_heads.shape[1]), num_heads_to_use)
            random_attn_heads = cls_attn_heads[:, chosen_heads, :]

            template_query_patches = template_query[:, :, 1:, :]
            template_query_patches = rearrange(template_query_patches, 'b h n d -> b n (h d)')

            obj_prototypes = torch.einsum('bkn, bnd -> bkd', random_attn_heads, template_query_patches)

            template_patches_for_assignment = template_patches[:, 1:, :]
            normalized_features = template_patches_for_assignment / template_patches_for_assignment.norm(dim=-1, keepdim=True)
            
            assignment = (normalized_features @ obj_prototypes.transpose(-2, -1))
            assignment_opt = self.sinkhorn_knopp(assignment) # Shape: (B, NumPatches, k)
            
            # --- START OF MODIFICATION ---
            # Use the GT bounding box to create a patch mask for robust object selection
            patch_mask = torch.zeros(last_template_img.shape[0], H_t * W_t, 1, device=last_template_img.device)
            for i in range(last_template_img.shape[0]):
                # Convert normalized bbox [cx, cy, w, h] to patch coordinates
                gt_box = last_template_anno[i]
                x_patch = gt_box[0] * W_t
                y_patch = gt_box[1] * H_t
                w_patch = gt_box[2] * W_t
                h_patch = gt_box[3] * H_t
                # Get patch indices within the box
                x1, y1 = int(x_patch - w_patch / 2), int(y_patch - h_patch / 2)
                x2, y2 = int(x_patch + w_patch / 2), int(y_patch + h_patch / 2)
                for row in range(max(0, y1), min(H_t, y2)):
                    for col in range(max(0, x1), min(W_t, x2)):
                        patch_mask[i, row * W_t + col, 0] = 1
            
            # Find the object prototype 'k' that has the highest correspondence with the masked patches
            masked_assignment_sum = (assignment_opt * patch_mask).sum(dim=1) # Shape: (B, k)
            _, target_object_idx = torch.max(masked_assignment_sum, dim=1)
            # --- END OF MODIFICATION ---

            target_assignment = assignment_opt[torch.arange(assignment_opt.shape[0]), :, target_object_idx].unsqueeze(1)
            refined_target_prototype = target_assignment @ normalized_features

            _, _, _, _, search_key = self.teacher.backbone(search_frame)
            search_key_patches = search_key[:, :, 1:, :]

            B, _, D = refined_target_prototype.shape
            num_heads = search_key_patches.shape[1]
            head_dim = D // num_heads
            
            proto_reshaped = refined_target_prototype.view(B, 1, num_heads, head_dim).permute(0,2,1,3)
            track_attn = (proto_reshaped @ search_key_patches.transpose(-2,-1)) * (head_dim ** -0.5)
            track_mask = track_attn.mean(dim=1).softmax(dim=-1)

            H_s, W_s = search_frame.shape[-2] // self.patch_size, search_frame.shape[-1] // self.patch_size
            track_mask_reshaped = track_mask.view(-1, 1, H_s, W_s)
            
            upsampled_mask = F.interpolate(track_mask_reshaped, size=search_frame.shape[-2:], mode='bilinear')
            
            predicted_templates = []
            for i in range(upsampled_mask.shape[0]):
                single_mask = upsampled_mask[i, 0]
                threshold = torch.quantile(single_mask, 0.8)
                binary_mask = (single_mask > threshold).cpu()
                
                rows = torch.any(binary_mask, axis=1)
                cols = torch.any(binary_mask, axis=0)
                if not (torch.any(rows) and torch.any(cols)):
                    predicted_templates.append(resize(last_template_img[i], self.cfg.DATA.TEMPLATE.SIZE))
                    continue

                ymin, ymax = torch.where(rows)[0][[0, -1]]
                xmin, xmax = torch.where(cols)[0][[0, -1]]
                
                predicted_crop = crop(search_frame[i], ymin, xmin, ymax - ymin, xmax - xmin)
                predicted_template = resize(predicted_crop, (self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE))
                predicted_templates.append(predicted_template)

            final_predicted_template = torch.stack(predicted_templates)

            if return_viz:
                viz_data = {'upsampled_mask': upsampled_mask}
                return final_predicted_template, viz_data
            
            return final_predicted_template