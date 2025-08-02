# In lib/models/odtrack/odtrack.py

import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
# --- START OF MODIFICATION ---
from torchvision.transforms.functional import resize, gaussian_blur
# --- END OF MODIFICATION ---

from lib.models.layers.head import build_box_head
from lib.models.odtrack.vit import vit_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_xywh_to_xyxy
from .apn import DoRA_APN


class ODTrack(nn.Module):
    def __init__(self, transformer, box_head, cfg, aux_loss=False, head_type="CORNER", token_len=1):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        if getattr(cfg.MODEL, "APN", None) and cfg.MODEL.APN.USE:
            self.apn = DoRA_APN(cfg)
            self.confidence_threshold = getattr(cfg.MODEL.APN, "CONFIDENCE_THRESHOLD", 0.5)
            self.apn_prototype_strategy = getattr(cfg.MODEL.APN, "PROTOTYPE_STRATEGY", "average")
            # --- START OF MODIFICATION ---
            self.background_suppression = getattr(cfg.MODEL.APN, "BACKGROUND_SUPPRESSION", False)
            # --- END OF MODIFICATION ---
            self.max_templates = getattr(cfg.DATA.TEMPLATE, "MAX_TOTAL_TEMPLATES", 3)
        else:
            self.apn = None
        self.template_size_apn = cfg.DATA.TEMPLATE.SIZE
        self.search_size_apn = cfg.DATA.SEARCH.SIZE
        self.head_type = head_type
        if head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

    # --- START OF MODIFICATION: New Helper Function ---
    def _suppress_background(self, image: torch.Tensor, box: torch.Tensor):
        """Blurs the background of an image outside the given bounding box."""
        b, c, h, w = image.shape
        # Create a blurred version of the entire image
        blurred_image = gaussian_blur(image, kernel_size=21)

        # Create a binary mask for the foreground (the box)
        fg_mask = torch.zeros_like(image)
        # Convert normalized xywh to pixel coordinates
        box_pix = box_xywh_to_xyxy(box) * torch.tensor([w, h, w, h], device=image.device)
        for i in range(b):
            x1, y1, x2, y2 = box_pix[i].long()
            fg_mask[i, :, y1:y2, x1:x2] = 1

        # Combine the sharp foreground with the blurry background
        return image * fg_mask + blurred_image * (1 - fg_mask)
    # --- END OF MODIFICATION ---
        
    def forward(self, template: List[torch.Tensor],
                search: List[torch.Tensor],
                template_history: List[torch.Tensor] = None,
                run_apn_viz=False,
                search_images_raw: List[torch.Tensor] = None,
                template_anno_history: List[torch.Tensor] = None
                ):
        
        predicted_templates_list = []
        confidence_list = []
        apn_viz_data_list = []

        if self.apn is not None and template_history is not None and len(template_history) > 0 and search_images_raw is not None:
            templates_for_apn = []
            annos_for_apn = []

            if self.apn_prototype_strategy == 'first':
                templates_for_apn = [template_history[0]]
                annos_for_apn = [template_anno_history[0]]
            elif self.apn_prototype_strategy == 'last':
                templates_for_apn = [template_history[-1]]
                annos_for_apn = [template_anno_history[-1]]
            else:
                templates_for_apn = template_history
                annos_for_apn = template_anno_history

            # --- START OF MODIFICATION: Apply background suppression ---
            if self.background_suppression:
                suppressed_templates = []
                for img, anno in zip(templates_for_apn, annos_for_apn):
                    suppressed_templates.append(self._suppress_background(img, anno))
                apn_templates_to_process = suppressed_templates
            else:
                apn_templates_to_process = templates_for_apn
            # --- END OF MODIFICATION ---
            
            apn_templates_resized = [resize(t, [self.template_size_apn, self.template_size_apn]) for t in apn_templates_to_process]

            for i in range(len(search_images_raw)):
                current_raw_search_frame_orig = search_images_raw[i]
                current_raw_search_frame = resize(current_raw_search_frame_orig, [self.search_size_apn, self.search_size_apn])
                
                apn_viz_data = None
                apn_args = (apn_templates_resized, current_raw_search_frame, annos_for_apn)

                if run_apn_viz and i == 0:
                    predicted_template, confidence, apn_viz_data = self.apn(*apn_args, return_viz=True)
                else:
                    predicted_template, confidence = self.apn(*apn_args, return_viz=False)

                predicted_templates_list.append(predicted_template)
                confidence_list.append(confidence)
                apn_viz_data_list.append(apn_viz_data)
        
        # ... (rest of the forward pass remains unchanged) ...
        out_dict = []
        for i in range(len(search)):
            template_for_backbone = list(template)
            
            if self.apn is not None and len(predicted_templates_list) > i:
                confidence = confidence_list[i]
                
                if confidence.mean() > self.confidence_threshold:
                    predicted_template = predicted_templates_list[i]
                    template_for_backbone.append(predicted_template)
                    
                    num_gt_templates = len(template)
                    if len(template_for_backbone) > self.max_templates:
                        template_for_backbone.pop(num_gt_templates)
            
            x, aux_dict = self.backbone(z=template_for_backbone, x=search[i])
            feat_last = x
            
            enc_opt = feat_last[:, -self.feat_len_s:]
            
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
            
            out = self.forward_head(opt, None)
            out.update(aux_dict)
            
            if len(predicted_templates_list) > i:
                out['predicted_template'] = predicted_templates_list[i]
                out['apn_confidence'] = confidence_list[i]
            if len(apn_viz_data_list) > i and apn_viz_data_list[i] is not None:
                out.update(apn_viz_data_list[i])
            out['templates_used'] = template_for_backbone
            
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)
        score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
        outputs_coord = bbox
        outputs_coord_new = outputs_coord.view(bs, Nq, 4)
        out = {'pred_boxes': outputs_coord_new, 'score_map': score_map_ctr,
               'size_map': size_map, 'offset_map': offset_map}
        return out

def build_odtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
    backbone.finetune_track(cfg=cfg, patch_start_index=1)
    box_head = build_box_head(cfg, backbone.embed_dim)
    model = ODTrack(backbone, box_head, cfg=cfg, head_type=cfg.MODEL.HEAD.TYPE)
    return model