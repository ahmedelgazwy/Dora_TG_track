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
from .apn import DoRA_APN


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
        if getattr(cfg.MODEL, "APN", None) and cfg.MODEL.APN.USE:
            self.apn = DoRA_APN(cfg)
        else:
            self.apn = None

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
                template_history: List[torch.Tensor] = None,
                run_apn_viz=False,
                raw_search_frames_for_apn=None
                ):
        
        # --- START OF RECURRENT LOGIC ---
        if self.apn is not None and raw_search_frames_for_apn is not None:
            # Step 1: Predict all future templates in parallel
            last_template_img = template_history[-1]
            if run_apn_viz:
                all_predicted_templates, apn_viz_data = self.apn(last_template_img, raw_search_frames_for_apn, return_viz=True)
            else:
                all_predicted_templates = self.apn(last_template_img, raw_search_frames_for_apn, return_viz=False)
            
            # Initialize dynamic template list for the backbone
            dynamic_template_list = template_history
            
            out_dict_list = []
            # Step 2: Loop through search frames sequentially for backbone processing
            for i in range(len(search)):
                # Get the predicted template for the current frame
                current_predicted_template = all_predicted_templates[:, i, :, :, :]
                
                # Append the new prediction to the dynamic context
                dynamic_template_list.append(current_predicted_template)

                # Run the backbone with the updated template list
                x, aux_dict = self.backbone(z=dynamic_template_list, x=search[i],
                                            ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len)
                
                # Process backbone output and run head
                feat_last = x[-1] if isinstance(x, list) else x
                enc_opt = feat_last[:, -self.feat_len_s:]
                if self.backbone.add_cls_token:
                    self.track_query = (x[:, :self.token_len].clone()).detach()
                
                att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))
                opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
                
                out = self.forward_head(opt, None)
                out.update(aux_dict)
                
                # Add visualization data for this specific frame
                if run_apn_viz:
                    out['predicted_template'] = current_predicted_template
                    out['upsampled_mask'] = apn_viz_data['upsampled_masks'][:, i, :, :, :]

                out_dict_list.append(out)
            
            return out_dict_list

        else:
            # Fallback to original behavior if APN is not used
            out_dict = []
            for i in range(len(search)):
                x, aux_dict = self.backbone(z=template, x=search[i],
                                            ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,
                                            return_last_attn=return_last_attn, track_query=self.track_query, token_len=self.token_len)
                feat_last = x[-1] if isinstance(x, list) else x
                enc_opt = feat_last[:, -self.feat_len_s:]
                if self.backbone.add_cls_token:
                    self.track_query = (x[:, :self.token_len].clone()).detach()
                
                att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))
                opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()
                out = self.forward_head(opt, None)
                out.update(aux_dict)
                out_dict.append(out)
            return out_dict
        # --- END OF RECURRENT LOGIC ---

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