# In lib/models/odtrack/apn.py

import os
import sys
import time
import math
import random
import datetime
import subprocess
from collections import defaultdict, deque

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from PIL import ImageFilter, ImageOps
from einops import rearrange, reduce, repeat

from torch.nn import functional as F
# from attmask import AttMask
import random


import torch
import torch.nn as nn
from lib.models.layers.patch_embed import PatchEmbed
# Import DoRA components
from ..dora import vision_transformer as vits
from ..dora.utils import MultiCropWrapper
from ..dora.sinkhorn_knopp import SinkhornKnopp
from einops import rearrange
from torchvision.transforms.functional import crop, resize
class AppearancePredictionNetwork(nn.Module):
    """
    Appearance Prediction Network (APN).
    Takes a sequence of historical templates and predicts the next one.
    This is a Transformer-based Encoder-Decoder architecture.
    """
    def __init__(self, cfg):
        super().__init__()

        # --- Hyperparameters from config ---
        self.embed_dim = cfg.MODEL.APN.EMBED_DIM
        self.template_size = cfg.DATA.TEMPLATE.SIZE
        self.num_history = cfg.DATA.TEMPLATE.NUM_HISTORY
        self.patch_size = cfg.MODEL.APN.PATCH_SIZE
        self.num_patches = (self.template_size // self.patch_size) ** 2

        # --- Components ---
        
        # 1. Image to Patch Embedding
        self.patch_embed = PatchEmbed(img_size=self.template_size, patch_size=self.patch_size,
                                      in_chans=3, embed_dim=self.embed_dim)

        # 2. Temporal Positional Encoding (learnable)
        # Encodes the position of each template in the historical sequence
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_history * self.num_patches, self.embed_dim))

        # 3. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=cfg.MODEL.APN.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=cfg.MODEL.APN.DROPOUT,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.MODEL.APN.NUM_ENCODER_LAYERS)

        # 4. Decoder Query Tokens (learnable)
        # These tokens act as the input to the decoder, asking it to generate the patches for the new template
        self.decoder_query_tokens = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        
        # 5. Generative Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=cfg.MODEL.APN.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=cfg.MODEL.APN.DROPOUT,
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.MODEL.APN.NUM_DECODER_LAYERS)

        # 6. Patch to Image Decoder (using Transposed Convolutions)
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.embed_dim // 4, 3, kernel_size=2, stride=2),
            nn.Sigmoid() # To ensure output pixel values are between 0 and 1
        )
        
    def forward(self, template_history: list):
        # Input: list of tensors, each of shape (B, C, H, W)
        B = template_history[0].shape[0]
        
        # Stack history into a single tensor and flatten batch and time dimensions
        # (T, B, C, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W)
        x = torch.stack(template_history, dim=1)
        x = x.flatten(0, 1)

        # 1. Get patch embeddings
        # (B*T, C, H, W) -> (B*T, num_patches, embed_dim)
        patch_embeds = self.patch_embed(x)

        # 2. Reshape and add temporal positional encoding
        # (B*T, num_patches, embed_dim) -> (B, T*num_patches, embed_dim)
        patch_embeds = patch_embeds.view(B, -1, self.embed_dim)
        patch_embeds += self.temporal_pos_embed

        # 3. Pass through the encoder
        memory = self.encoder(patch_embeds)
        
        # 4. Prepare decoder queries
        decoder_queries = self.decoder_query_tokens.expand(B, -1, -1)
        
        # 5. Pass through the decoder
        # Output shape: (B, num_patches, embed_dim)
        predicted_patches = self.decoder(tgt=decoder_queries, memory=memory)

        # 6. Reconstruct the image from patches
        # (B, num_patches, embed_dim) -> (B, embed_dim, sqrt(num_patches), sqrt(num_patches))
        feat_h = feat_w = int(self.num_patches ** 0.5)
        predicted_patches = predicted_patches.permute(0, 2, 1).contiguous().view(B, self.embed_dim, feat_h, feat_w)
        
        # Upsample to the final image
        predicted_template = self.image_decoder(predicted_patches)
        
        return predicted_template

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

        # 1. Instantiate DoRA teacher model
        # The DINOHead is required by the MultiCropWrapper but its output is not directly used here
        dora_backbone = vits.__dict__[cfg.MODEL.APN.DORA_ARCH](patch_size=self.patch_size, num_classes=0)
        dino_head = vits.DINOHead(dora_backbone.embed_dim, out_dim=65536, use_bn=False, norm_last_layer=True)
        self.teacher = MultiCropWrapper(dora_backbone, dino_head)

        # 2. Load pre-trained weights
        self._load_dora_weights(self.teacher, cfg.MODEL.APN.DORA_WEIGHTS_PATH)

        # 3. Freeze the entire teacher model
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

        # 4. Instantiate Sinkhorn-Knopp for object refinement
        self.sinkhorn_knopp = SinkhornKnopp(num_iters_sk=3, epsilon_sk=0.05)

    def _load_dora_weights(self, model, path_to_weights):
        """Helper function to load DoRA checkpoint."""
        print(f"Loading DoRA weights from: {path_to_weights}")
        state_dict = torch.load(path_to_weights, map_location="cpu")
        # DoRA checkpoints often store the teacher weights under the 'teacher' key
        if 'teacher' in state_dict:
            state_dict = state_dict['teacher']
        # Remove 'module.' prefix if saved with DDP
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"DoRA weights loaded with msg: {msg}")

    def forward(self, last_template_img, batched_raw_search_frames, return_viz=False):
        """
        Predicts templates for a batch of search frames in parallel.
        Args:
            last_template_img (Tensor): (B, C, H_t, W_t)
            batched_raw_search_frames (Tensor): (B, NumSearch, C, H_s, W_s)
        """
        with torch.no_grad():
            B, Ns, C, Hs, Ws = batched_raw_search_frames.shape
            
            # Step 1: Discover object prototype from the single template image (once per batch)
            _, template_patches, template_attn, template_query, _ = self.teacher.backbone(last_template_img)
            
            H_t, W_t = last_template_img.shape[-2] // self.patch_size, last_template_img.shape[-1] // self.patch_size
            
            # Use attention from CLS token to all patches
            cls_attn_heads = template_attn[:, :, 0, 1:]
            
            # Heuristically choose attention heads. Random selection is a good start.
            num_heads_to_use = min(self.num_queries, cls_attn_heads.shape[1])
            chosen_heads = random.sample(range(cls_attn_heads.shape[1]), num_heads_to_use)
            random_attn_heads = cls_attn_heads[:, chosen_heads, :]

            template_query_patches = rearrange(template_query[:, :, 1:, :], 'b h n d -> b n (h d)')
            obj_prototypes = torch.einsum('bkn, bnd -> bkd', random_attn_heads, template_query_patches)

            # Refine prototypes using Sinkhorn-Knopp
            normalized_features = template_patches[:, 1:, :] / template_patches[:, 1:, :].norm(dim=-1, keepdim=True)
            assignment = (normalized_features @ obj_prototypes.transpose(-2, -1))
            assignment_opt = self.sinkhorn_knopp(assignment) # (B, num_patches, k)

            # --- Step 2: Select the target object (heuristic: the one most focused on the center) ---
            center_patch_idx = (H_t // 2) * W_t + (W_t // 2)
            # Find which object query (k) has the max value for the center patch column
            _, target_object_idx = torch.max(assignment_opt[:, center_patch_idx, :], dim=1)
            target_assignment = assignment_opt[torch.arange(B), :, target_object_idx].unsqueeze(1)
            refined_target_prototype = target_assignment @ normalized_features

            # Step 2: Process all search frames in a single large batch
            search_frames_flat = batched_raw_search_frames.view(B * Ns, C, Hs, Ws)
            _, _, _, _, search_key_flat = self.teacher.backbone(search_frames_flat)
            search_key_patches_flat = search_key_flat[:, :, 1:, :]

            # Step 3: Batched Cross-Attention
            # Repeat the prototype for each search frame in the sequence
            target_prototype_repeated = refined_target_prototype.repeat_interleave(Ns, dim=0)
            
            _, _, D = target_prototype_repeated.shape
            num_heads = search_key_patches_flat.shape[1]
            head_dim = D // num_heads
            
            proto_reshaped = target_prototype_repeated.view(B * Ns, 1, num_heads, head_dim).permute(0, 2, 1, 3)
            track_attn = (proto_reshaped @ search_key_patches_flat.transpose(-2, -1)) * (head_dim ** -0.5)
            track_mask_flat = track_attn.mean(dim=1).softmax(dim=-1)

            # Step 4: Batched Cropping and Resizing
            H_s_p, W_s_p = Hs // self.patch_size, Ws // self.patch_size
            track_mask_reshaped = track_mask_flat.view(B * Ns, 1, H_s_p, W_s_p)
            upsampled_mask_flat = F.interpolate(track_mask_reshaped, size=(Hs, Ws), mode='bilinear')
            
            all_predicted_templates = []
            for i in range(B * Ns):
                single_mask = upsampled_mask_flat[i, 0]
                threshold = torch.quantile(single_mask, 0.9)
                binary_mask = (single_mask > threshold).cpu()
                
                # Get bounding box from the binary mask
                rows = torch.any(binary_mask, axis=1)
                cols = torch.any(binary_mask, axis=0)
                if not (torch.any(rows) and torch.any(cols)):
                    # Fallback: use a resized version of the original input template
                    original_template_idx = i // Ns
                    fallback_template = resize(last_template_img[original_template_idx], (self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE))
                    all_predicted_templates.append(fallback_template)
                    continue

                ymin, ymax = torch.where(rows)[0][[0, -1]]
                xmin, xmax = torch.where(cols)[0][[0, -1]]
                
                predicted_crop = crop(search_frames_flat[i], ymin, xmin, ymax - ymin, xmax - xmin)
                predicted_template = resize(predicted_crop, (self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE))
                all_predicted_templates.append(predicted_template)

            # Reshape back to (B, Ns, C, H_t, W_t)
            final_predicted_templates = torch.stack(all_predicted_templates).view(B, Ns, C, self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE)

            if return_viz:
                # Reshape masks for visualization as well
                upsampled_masks = upsampled_mask_flat.view(B, Ns, 1, Hs, Ws)
                viz_data = {'upsampled_masks': upsampled_masks} # Note the plural 'masks'
                return final_predicted_templates, viz_data
            
            return final_predicted_templates