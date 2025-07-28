# In lib/models/odtrack/odtrack.py

import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from torchvision.transforms.functional import resize

from lib.models.layers.head import build_box_head
from lib.models.odtrack.vit import vit_base_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from .apn import DoRA_APN


class ODTrack(nn.Module):
    def __init__(self, transformer, box_head, cfg, aux_loss=False, head_type="CORNER", token_len=1):
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        if getattr(cfg.MODEL, "APN", None) and cfg.MODEL.APN.USE:
            self.apn = DoRA_APN(cfg)
        else:
            self.apn = None
        self.template_size_apn = cfg.DATA.TEMPLATE.SIZE
        self.search_size_apn = cfg.DATA.SEARCH.SIZE
        self.head_type = head_type
        if head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)
        
    def forward(self, template: List[torch.Tensor],
                search: List[torch.Tensor],
                template_history: List[torch.Tensor] = None,
                run_apn_viz=False,
                search_images_raw: List[torch.Tensor] = None,
                # --- START OF MODIFICATION ---
                template_anno_history: List[torch.Tensor] = None
                # --- END OF MODIFICATION ---
                ):
        
        predicted_templates_list = []
        apn_viz_data_list = []

        if self.apn is not None and template_history is not None and len(template_history) > 0 and search_images_raw is not None:
            last_template_img_orig = template_history[-1]
            # --- START OF MODIFICATION ---
            # Get the annotation for the last template
            last_template_anno = template_anno_history[-1]
            # --- END OF MODIFICATION ---
            
            last_template_img = resize(last_template_img_orig, [self.template_size_apn, self.template_size_apn])

            for i in range(len(search_images_raw)):
                current_raw_search_frame_orig = search_images_raw[i]
                current_raw_search_frame = resize(current_raw_search_frame_orig, [self.search_size_apn, self.search_size_apn])
                
                apn_viz_data = None
                
                # --- START OF MODIFICATION ---
                # Pass the annotation to the APN
                apn_args = (last_template_img, current_raw_search_frame, last_template_anno)
                # --- END OF MODIFICATION ---

                if run_apn_viz and i == 0:
                    predicted_template, apn_viz_data = self.apn(*apn_args, return_viz=True)
                else:
                    predicted_template = self.apn(*apn_args, return_viz=False)

                predicted_templates_list.append(predicted_template)
                apn_viz_data_list.append(apn_viz_data)
        
        out_dict = []
        for i in range(len(search)):
            if self.apn is not None and len(predicted_templates_list) > i:
                template_for_backbone = [predicted_templates_list[i]]
            else:
                template_for_backbone = template

            # Note: ce_template_mask and other similar args are omitted for clarity
            x, aux_dict = self.backbone(z=template_for_backbone, x=search[i])
            feat_last = x
            
            enc_opt = feat_last[:, -self.feat_len_s:]
            
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
            
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            
            if len(predicted_templates_list) > i:
                out['predicted_template'] = predicted_templates_list[i]
            if len(apn_viz_data_list) > i and apn_viz_data_list[i] is not None:
                out.update(apn_viz_data_list[i])
            
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
    # This function remains unchanged.
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