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

    def forward(self, last_template_img, search_frame, return_viz=False):
        """
        Predicts the appearance of the object from `last_template_img` in the `search_frame`.
        """
        with torch.no_grad():
            # The DoRA teacher expects a list of images (for multi-crop), so we wrap our tensors in lists
            # We also only care about the backbone output, not the DINO head, so we call it directly
            
            # --- Step 1: Discover object prototypes from the template image ---
            _, template_patches, template_attn, template_query, _ = self.teacher.backbone(last_template_img)
            
            H_t, W_t = last_template_img.shape[-2] // self.patch_size, last_template_img.shape[-1] // self.patch_size
            num_template_patches = H_t * W_t
            
            # Use attention from CLS token to all patches
            cls_attn_heads = template_attn[:, :, 0, 1:]
            
            # Heuristically choose attention heads. Random selection is a good start.
            num_heads_to_use = min(self.num_queries, cls_attn_heads.shape[1])
            chosen_heads = random.sample(range(cls_attn_heads.shape[1]), num_heads_to_use)
            random_attn_heads = cls_attn_heads[:, chosen_heads, :]

            # Extract corresponding query vectors for the template
            template_query_patches = template_query[:, :, 1:, :] # (B, num_heads, num_patches, head_dim)
            template_query_patches = rearrange(template_query_patches, 'b h n d -> b n (h d)') # (B, num_patches, embed_dim)

            # Create initial object prototypes
            obj_prototypes = torch.einsum('bkn, bnd -> bkd', random_attn_heads, template_query_patches)

            # Refine prototypes using Sinkhorn-Knopp
            normalized_features = template_patches[:, 1:, :] / template_patches[:, 1:, :].norm(dim=-1, keepdim=True)
            assignment = (normalized_features @ obj_prototypes.transpose(-2, -1))
            assignment_opt = self.sinkhorn_knopp(assignment) # (B, num_patches, k)

            # --- Step 2: Select the target object (heuristic: the one most focused on the center) ---
            center_patch_idx = (H_t // 2) * W_t + (W_t // 2)
            # Find which object query (k) has the max value for the center patch column
            _, target_object_idx = torch.max(assignment_opt[:, center_patch_idx, :], dim=1)

            # Use the refined assignment to create a clean prototype for the target object
            target_assignment = assignment_opt[torch.arange(assignment_opt.shape[0]), :, target_object_idx].unsqueeze(1) # (B, 1, num_patches)
            refined_target_prototype = target_assignment @ normalized_features # (B, 1, embed_dim)

            # --- Step 3: Track the selected object in the search frame ---
            _, _, _, _, search_key = self.teacher.backbone(search_frame)
            search_key_patches = search_key[:, :, 1:, :] # (B, num_heads, num_search_patches, head_dim)

            # Perform cross-attention: (target_prototype) x (search_key)
            # We need to reshape the prototype to match head dimension for attention
            B, _, D = refined_target_prototype.shape
            num_heads = search_key_patches.shape[1]
            head_dim = D // num_heads
            
            proto_reshaped = refined_target_prototype.view(B, 1, num_heads, head_dim).permute(0,2,1,3) # (B, num_heads, 1, head_dim)
            
            track_attn = (proto_reshaped @ search_key_patches.transpose(-2,-1)) * (head_dim ** -0.5) # (B, num_heads, 1, num_search_patches)
            
            # Average attention across heads and apply softmax
            track_mask = track_attn.mean(dim=1).softmax(dim=-1) # (B, 1, num_search_patches)

            # --- Step 4: Generate the predicted template by cropping ---
            H_s, W_s = search_frame.shape[-2] // self.patch_size, search_frame.shape[-1] // self.patch_size
            track_mask_reshaped = track_mask.view(-1, 1, H_s, W_s)
            
            # Upsample mask to the size of the search frame
            upsampled_mask = F.interpolate(track_mask_reshaped, size=search_frame.shape[-2:], mode='bilinear')
            
            # Binarize the mask and find bounding box
            predicted_templates = []
            for i in range(upsampled_mask.shape[0]):
                single_mask = upsampled_mask[i, 0]
                threshold = single_mask.mean() # Simple thresholding
                binary_mask = (single_mask > threshold).cpu()
                
                # Get bounding box from the binary mask
                rows = torch.any(binary_mask, axis=1)
                cols = torch.any(binary_mask, axis=0)
                if not (torch.any(rows) and torch.any(cols)): # Handle empty mask case
                    predicted_templates.append(resize(last_template_img[i], self.cfg.DATA.TEMPLATE.SIZE))
                    continue

                ymin, ymax = torch.where(rows)[0][[0, -1]]
                xmin, xmax = torch.where(cols)[0][[0, -1]]
                
                # Crop the search frame and resize to template size
                predicted_crop = crop(search_frame[i], ymin, xmin, ymax - ymin, xmax - xmin)
                predicted_template = resize(predicted_crop, (self.cfg.DATA.TEMPLATE.SIZE, self.cfg.DATA.TEMPLATE.SIZE))
                predicted_templates.append(predicted_template)
            final_predicted_template = torch.stack(predicted_templates)
            if return_viz:
                viz_data = {'upsampled_mask': upsampled_mask}
                print('RETURNING VIS DATA')
                return final_predicted_template, viz_data
            
            return final_predicted_template
            
