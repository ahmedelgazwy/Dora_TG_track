# In lib/models/odtrack/apn.py

import torch
import torch.nn as nn
from lib.models.layers.patch_embed import PatchEmbed

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
        # --- START OF MODIFICATION ---
        # Get dropout value from config for additional layers
        self.dropout_rate = cfg.MODEL.APN.DROPOUT
        # --- END OF MODIFICATION ---

        # --- Components ---
        
        # 1. Image to Patch Embedding
        self.patch_embed = PatchEmbed(img_size=self.template_size, patch_size=self.patch_size,
                                      in_chans=3, embed_dim=self.embed_dim)

        # --- START OF MODIFICATION ---
        # Add dropout after patch embedding
        self.patch_drop = nn.Dropout(self.dropout_rate)
        # --- END OF MODIFICATION ---

        # 2. Temporal Positional Encoding (learnable)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_history * self.num_patches, self.embed_dim))

        # 3. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=cfg.MODEL.APN.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout_rate, # Use the dropout rate here
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.MODEL.APN.NUM_ENCODER_LAYERS)

        # 4. Decoder Query Tokens (learnable)
        self.decoder_query_tokens = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        
        # 5. Generative Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=cfg.MODEL.APN.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout_rate, # Use the dropout rate here
            activation='relu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.MODEL.APN.NUM_DECODER_LAYERS)

        # 6. Patch to Image Decoder
        # --- START OF MODIFICATION ---
        # Add dropout in the image reconstruction head
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.1), # Added 2D dropout
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.embed_dim // 4, self.embed_dim // 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.embed_dim // 8, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        # --- END OF MODIFICATION ---
        
    def forward(self, template_history: list):
        B = template_history[0].shape[0]
        x = torch.stack(template_history, dim=1)
        x = x.flatten(0, 1)

        # print("x", x.shape)

        patch_embeds = self.patch_embed(x)
        patch_embeds = self.patch_drop(patch_embeds) # Apply dropout
        patch_embeds = patch_embeds.reshape(B, -1, self.embed_dim)
        patch_embeds += self.temporal_pos_embed
        # print("patch_embeds", patch_embeds.shape)

        memory = self.encoder(patch_embeds)
        # print("memory", memory.shape)
        
        decoder_queries = self.decoder_query_tokens.expand(B, -1, -1)
        predicted_patches = self.decoder(tgt=decoder_queries, memory=memory)
        # print("predicted_patches", predicted_patches.shape)

        feat_h = feat_w = int(self.num_patches ** 0.5)
        predicted_patches = predicted_patches.permute(0, 2, 1).contiguous().view(B, self.embed_dim, feat_h, feat_w)
        
        predicted_template = self.image_decoder(predicted_patches)
        
        return predicted_template