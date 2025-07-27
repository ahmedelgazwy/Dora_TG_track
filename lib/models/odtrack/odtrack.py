# In lib/models/odtrack/odtrack.py

import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.odtrack.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.odtrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
# --- START OF MODIFICATION ---
from .apn import DoRA_APN
# --- END OF MODIFICATION ---


class ODTrack(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, cfg, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        # --- START OF MODIFICATION ---
        # Instantiate the DoRA-based Appearance Prediction Network (APN) if enabled
        if getattr(cfg.MODEL, "APN", None) and cfg.MODEL.APN.USE:
            self.apn = DoRA_APN(cfg)
        else:
            self.apn = None
        # --- END OF MODIFICATION ---

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        self.track_query = None
        self.token_len = token_len

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                # --- START OF MODIFICATION ---
                template_history: List[torch.Tensor] = None, # Accepts template history
                run_apn_viz=False # New flag to control visualization
                # --- END OF MODIFICATION ---
                ):
        assert isinstance(search, list), "The type of search is not List"
        apn_viz_data = None
        predicted_template = None
        # --- START OF MODIFICATION ---
        # If APN is used, predict the template for the current frame.
        # Otherwise, use the latest template from history (original behavior).
        if self.apn is not None and template_history is not None and len(template_history) > 0:
            # The DoRA APN needs the last known template and the current search frame
            last_template_img = template_history[-1]
            current_search_frame = search[0] # Assuming single search frame
            if run_apn_viz:
                predicted_template, apn_viz_data = self.apn(last_template_img, current_search_frame, return_viz=True)
            else:
                predicted_template = self.apn(last_template_img, current_search_frame, return_viz=False)
            
            
            # The backbone expects a list of templates. We use only the predicted one.
            template_for_backbone = [predicted_template]
        else:
            # Fallback to original behavior if APN is not used
            predicted_template = None 
             
            template_for_backbone = template
        # --- END OF MODIFICATION ---

        out_dict = []
        for i in range(len(search)):
            # --- START OF MODIFICATION ---
            # Use the template determined above (either predicted or from input)
            x, aux_dict = self.backbone(z=template_for_backbone, x=search[i],
            # --- END OF MODIFICATION ---
                                        ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len)
            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
            if self.backbone.add_cls_token:
                self.track_query = (x[:, :self.token_len].clone()).detach() # stop grad  (B, N, C)
                
            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            
            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x
            
            # --- START OF MODIFICATION ---
            # The predicted template is no longer needed for loss calculation,
            # but we can keep it for debugging/visualization if needed.
            if i==0:
                if predicted_template is not None:
                    out['predicted_template'] = predicted_template
                if apn_viz_data is not None:
                    out.update(apn_viz_data)
                # --- END OF MODIFICATION ---
            
            out_dict.append(out)
            
        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError


def build_odtrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )
    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = ODTrack(
        backbone,
        box_head,
        cfg=cfg,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )

    return model