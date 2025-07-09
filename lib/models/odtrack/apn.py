# In lib/models/odtrack/apn.py

import torch
import torch.nn as nn
from lib.models.layers.patch_embed import PatchEmbed

class AppearancePredictionNetwork(nn.Module):
    """
    Appearance Prediction Network (APN).
    Takes a sequence of historical templates and predicts the next one.
    This is a Transformer-based Encoder-Decoder architecture with Layer Normalization.
    """
    def __init__(self, cfg):
        super().__init__()

        # --- Hyperparameters from config ---
        self.embed_dim = cfg.MODEL.APN.EMBED_DIM
        self.template_size = cfg.DATA.TEMPLATE.SIZE
        self.num_history = cfg.DATA.TEMPLATE.NUM_HISTORY
        self.patch_size = cfg.MODEL.APN.PATCH_SIZE
        self.num_patches = (self.template_size // self.patch_size) ** 2
        self.dropout_rate = cfg.MODEL.APN.DROPOUT

        # --- Components ---
        
        # 1. Image to Patch Embedding
        self.patch_embed = PatchEmbed(img_size=self.template_size, patch_size=self.patch_size,
                                      in_chans=3, embed_dim=self.embed_dim)

        self.patch_drop = nn.Dropout(self.dropout_rate)

        # 2. Temporal Positional Encoding (learnable)
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_history * self.num_patches, self.embed_dim))

        # --- START OF MODIFICATION ---
        # Add LayerNorm layers for stabilization
        self.encoder_input_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_output_norm = nn.LayerNorm(self.embed_dim)
        self.decoder_output_norm = nn.LayerNorm(self.embed_dim)
        # --- END OF MODIFICATION ---

        # 3. Temporal Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=cfg.MODEL.APN.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout_rate,
            activation='relu',
            batch_first=True,
            norm_first=True  # Use pre-norm for better stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.MODEL.APN.NUM_ENCODER_LAYERS, norm=self.encoder_output_norm) # Pass final norm layer

        # 4. Decoder Query Tokens (learnable)
        self.decoder_query_tokens = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
        
        # 5. Generative Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=cfg.MODEL.APN.NUM_HEADS,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout_rate,
            activation='relu',
            batch_first=True,
            norm_first=True # Use pre-norm for better stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=cfg.MODEL.APN.NUM_DECODER_LAYERS, norm=self.decoder_output_norm) # Pass final norm layer

        # 6. Patch to Image Decoder
        self.image_decoder = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(),
            # Adjusted the architecture to better match the patch size (16x16 -> 128x128 requires 3 doublings if starting from 16x16, 4 if starting from 8x8)
            nn.ConvTranspose2d(self.embed_dim // 4, self.embed_dim // 8, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(self.embed_dim // 8, 3, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
        
    def forward(self, template_history: list):
        B = template_history[0].shape[0]
        x = torch.stack(template_history, dim=1)
        x = x.flatten(0, 1)

        patch_embeds = self.patch_embed(x)
        patch_embeds = self.patch_drop(patch_embeds)
        patch_embeds = patch_embeds.reshape(B, -1, self.embed_dim)
        patch_embeds += self.temporal_pos_embed
        
        # --- START OF MODIFICATION ---
        # Apply LayerNorm before the encoder
        encoder_input = self.encoder_input_norm(patch_embeds)
        memory = self.encoder(encoder_input)
        # The final norm for the encoder is now handled inside the nn.TransformerEncoder module
        # --- END OF MODIFICATION ---
        
        decoder_queries = self.decoder_query_tokens.expand(B, -1, -1)
        
        # --- START OF MODIFICATION ---
        predicted_patches = self.decoder(tgt=decoder_queries, memory=memory)
        # The final norm for the decoder is now handled inside the nn.TransformerDecoder module
        # --- END OF MODIFICATION ---

        feat_h = feat_w = int(self.num_patches ** 0.5)
        predicted_patches_norm = predicted_patches # The output of the decoder is already normed
        predicted_patches_reshaped = predicted_patches_norm.permute(0, 2, 1).contiguous().view(B, self.embed_dim, feat_h, feat_w)
        
        predicted_template = self.image_decoder(predicted_patches_reshaped)
        
        return predicted_template