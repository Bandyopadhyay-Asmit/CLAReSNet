"""
CLAReSNet: Convolutional Latent Attention Residual Spectral Network
====================================================================

A deep learning architecture for hyperspectral image classification combining:
- CNN-based spatial feature extraction
- Multi-scale spectral attention with adaptive latent tokens
- Transformer-style encoder layers with RNN modules
- Enhanced attention mechanisms (CBAM)

Author: Asmit Bandyopadhyay
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
from scipy.spatial.distance import pdist

# =============================================================================
# SECTION 1: CONFIGURATION & HYPERPARAMETERS
# =============================================================================

@dataclass
class ModelConfig:
    """
    Configuration dataclass for CLAReSNet model architecture.
    
    Attributes:
        emb_dim (int): Embedding dimension for feature representations. Default: 256
        base_c (int): Base number of channels in CNN layers. Default: 64
        n_encoder_layers (int): Number of spectral encoder layers. Default: 4
        use_cbam (bool): Whether to use CBAM attention in CNN. Default: True
        use_multiscale (bool): Whether to use multi-scale convolutions. Default: True
        use_checkpoint (bool): Whether to use gradient checkpointing. Default: False
        dropout (float): Dropout rate for regularization. Default: 0.4
    """
    emb_dim: int = 256
    base_c: int = 64
    n_encoder_layers: int = 4
    use_cbam: bool = True
    use_multiscale: bool = True
    use_checkpoint: bool = False
    dropout: float = 0.4


# =============================================================================
# SECTION 2: POSITIONAL ENCODING
# =============================================================================

class SpectralPositionalEncoding(nn.Module):
    """
    Hybrid positional encoding specifically designed for hyperspectral sequences.
    
    Combines sinusoidal (Transformer-style) and learnable positional encodings
    to capture both fixed spectral relationships and data-adaptive patterns.
    
    Args:
        emb_dim (int): Embedding dimension
        encoding_type (str): Type of encoding - 'sinusoidal', 'learnable', or 'hybrid'
        max_bands (int): Maximum number of spectral bands to support
    
    Input Shape:
        x: (B, num_bands, emb_dim) - Batch of spectral sequences
        wavelengths: Optional wavelength information (not used in current implementation)
    
    Output Shape:
        (B, num_bands, emb_dim) - Input with positional encoding added
    """
    
    def __init__(self, emb_dim: int, encoding_type: str = "hybrid", max_bands: int = 512):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoding_type = encoding_type
        
        if encoding_type == "learnable":
            # Learnable positional embeddings for each band position
            # Shape: (max_bands, emb_dim)
            self.pos_embedding = nn.Parameter(torch.randn(max_bands, emb_dim) * 0.02)
            
        elif encoding_type == "hybrid":
            # Combination of sinusoidal (fixed) + learnable (adaptive)
            # Sinusoidal part: (1, max_bands, emb_dim // 2)
            self.register_buffer('sinusoidal_pe', 
                               self._create_sinusoidal_encoding(max_bands, emb_dim // 2))
            # Learnable part: (max_bands, emb_dim // 2)
            self.learnable_pe = nn.Parameter(torch.randn(max_bands, emb_dim // 2) * 0.02)
            
        else:  # "sinusoidal" - original Transformer approach
            # Shape: (1, max_bands, emb_dim)
            self.register_buffer('pe', 
                               self._create_sinusoidal_encoding(max_bands, emb_dim))
    
    def _create_sinusoidal_encoding(self, max_len: int, emb_dim: int) -> torch.Tensor:
        """
        Create standard sinusoidal positional encoding from 'Attention is All You Need'.
        
        Args:
            max_len (int): Maximum sequence length
            emb_dim (int): Embedding dimension
        
        Returns:
            torch.Tensor: Sinusoidal encoding of shape (1, max_len, emb_dim)
        
        Formula:
            PE(pos, 2i) = sin(pos / 10000^(2i/emb_dim))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/emb_dim))
        """
        pe = torch.zeros(max_len, emb_dim)  # (max_len, emb_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2).float() * 
                            (-math.log(10000.0) / emb_dim))  # (emb_dim/2,)
        
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        
        return pe.unsqueeze(0)  # (1, max_len, emb_dim)
    
    def forward(self, x: torch.Tensor, wavelengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input sequence.
        
        Args:
            x: Input tensor of shape (B, num_bands, emb_dim)
            wavelengths: Optional wavelength information (reserved for future use)
        
        Returns:
            torch.Tensor: Input with positional encoding added, shape (B, num_bands, emb_dim)
        """
        B, num_bands, D = x.shape
        
        if self.encoding_type == "learnable":
            # Slice and expand: (num_bands, emb_dim) -> (B, num_bands, emb_dim)
            pos_enc = self.pos_embedding[:num_bands].unsqueeze(0).expand(B, -1, -1)
                
        elif self.encoding_type == "hybrid":
            # Sinusoidal: (1, num_bands, emb_dim//2) -> (B, num_bands, emb_dim//2)
            sin_pe = self.sinusoidal_pe[:, :num_bands, :]
            # Learnable: (num_bands, emb_dim//2) -> (B, num_bands, emb_dim//2)
            learnable_pe = self.learnable_pe[:num_bands].unsqueeze(0).expand(B, -1, -1)
            # Concatenate: (B, num_bands, emb_dim)
            pos_enc = torch.cat([sin_pe.expand(B, -1, -1), learnable_pe], dim=-1)
            
        else:  # sinusoidal
            # (1, num_bands, emb_dim) broadcast to (B, num_bands, emb_dim)
            pos_enc = self.pe[:, :num_bands, :]
            
        return x + pos_enc


# =============================================================================
# SECTION 3: MULTI-SCALE SPECTRAL LATENT ATTENTION
# =============================================================================

class MultiScaleSpectralLatentAttention(nn.Module):
    """
    Multi-scale spectral attention mechanism with adaptive latent token compression.
    
    This module implements a sophisticated attention mechanism inspired by Perceiver
    architecture, with multi-scale processing for capturing spectral features at
    different resolutions.
    
    Architecture:
        1. Input-to-Latent Cross-Attention (Encoding): Compress input to latent space
        2. Latent-to-Latent Self-Attention (Processing): Process in compressed space
        3. Latent FFN: Further refinement of latent representations
        4. Latent-to-Output Cross-Attention (Decoding): Expand back to original space
        5. Multi-scale Fusion: Combine outputs from different scales
    
    Args:
        emb_dim (int): Embedding dimension
        scales (List[int]): Downsampling factors for multi-scale processing [1, 2, 4]
        min_tokens (int): Minimum number of latent tokens
        max_tokens (int): Maximum number of latent tokens
        base_tokens (int): Base number of tokens for scaling calculation
        base_length (int): Base sequence length for scaling calculation
        max_len (int): Maximum input sequence length
    
    Input Shape:
        x: (B, T, emb_dim) where B=batch, T=sequence length (spectral bands)
    
    Output Shape:
        (B, T, emb_dim) - Attended features at original resolution
        Optional: attention weights if return_att=True
    """
    
    def __init__(self, emb_dim: int, scales: List[int] = [1, 2, 4], 
                 min_tokens: int = 8, max_tokens: int = 64, 
                 base_tokens: int = 16, base_length: int = 16, max_len: int = 5000):
        super().__init__()
        self.scales = scales
        self.emb_dim = emb_dim
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.base_tokens = base_tokens
        self.base_length = base_length
        
        # Positional encoding for input sequences
        # Applies to input: (B, T, emb_dim) -> (B, T, emb_dim)
        self.pos_encoding = SpectralPositionalEncoding(
            emb_dim, encoding_type="hybrid", max_bands=max_len
        )
        
        # Pool of latent tokens - dynamically sliced based on sequence length
        # Shape: (max_tokens, emb_dim)
        self.latent_token_pool = nn.Parameter(torch.randn(max_tokens, emb_dim) * 0.02)
        
        # Input-to-latent cross-attention (encoding phase) for each scale
        # Query: latent tokens (B, num_latents, emb_dim)
        # Key/Value: input sequence (B, T//scale, emb_dim)
        # Output: (B, num_latents, emb_dim)
        self.input_to_latent = nn.ModuleList([
            nn.MultiheadAttention(emb_dim, num_heads=8, dropout=0.1, batch_first=True)
            for _ in scales
        ])
        
        # Latent-to-latent self-attention (processing phase)
        # All: (B, num_latents, emb_dim)
        self.latent_self_attn = nn.ModuleList([
            nn.MultiheadAttention(emb_dim, num_heads=8, dropout=0.1, batch_first=True)
            for _ in scales
        ])
        
        # Latent-to-output cross-attention (decoding phase)
        # Query: downsampled input (B, T//scale, emb_dim)
        # Key/Value: latent tokens (B, num_latents, emb_dim)
        # Output: (B, T//scale, emb_dim)
        self.latent_to_output = nn.ModuleList([
            nn.MultiheadAttention(emb_dim, num_heads=8, dropout=0.1, batch_first=True)
            for _ in scales
        ])
        
        # Layer normalizations for each phase and scale
        self.input_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in scales])
        self.latent_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in scales])
        self.output_norms = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in scales])
        
        # FFN for latent processing at each scale
        # (B, num_latents, emb_dim) -> (B, num_latents, emb_dim*2) -> (B, num_latents, emb_dim)
        self.latent_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(emb_dim, emb_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(emb_dim * 2, emb_dim),
                nn.Dropout(0.1)
            ) for _ in scales
        ])
        
        # Scale fusion mechanism
        # Learnable weights for combining scales
        self.scale_weights = nn.Parameter(torch.ones(len(scales)) / len(scales))
        # Fusion network: (B, T, emb_dim * num_scales) -> (B, T, emb_dim)
        self.scale_fusion = nn.Sequential(
            nn.Linear(emb_dim * len(scales), emb_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        
        # Final output processing
        self.final_norm = nn.LayerNorm(emb_dim)
        self.dropout = nn.Dropout(0.1)

    def _get_num_latents(self, T: int) -> int:
        """
        Compute number of latent tokens based on input sequence length.
        Longer sequences get more latent tokens for better compression.
        
        Formula: base_tokens * log2(T / base_length)
        
        Args:
            T (int): Input sequence length
        
        Returns:
            int: Number of latent tokens, clamped to [min_tokens, max_tokens]
        
        Example:
            T=16: 16 tokens (base case)
            T=32: 24 tokens
            T=64: 32 tokens
        """
        num_latents = int(self.base_tokens * math.log2(max(T, self.base_length) / self.base_length))
        return min(max(num_latents, self.min_tokens), self.max_tokens)

    def _process_scale(self, x_pos: torch.Tensor, scale_idx: int, scale: int, 
                      num_latents: int, return_att: bool = False):
        """
        Process input through latent attention at a specific scale.
        
        Args:
            x_pos: Input with positional encoding, shape (B, T, emb_dim)
            scale_idx: Index of scale in self.scales list
            scale: Downsampling factor (1, 2, 4, etc.)
            num_latents: Number of latent tokens to use
            return_att: Whether to return attention weights
        
        Returns:
            output: Attended features at original resolution (B, T, emb_dim)
            attn_dict: Dictionary of attention weights (if return_att=True)
        
        Processing Steps:
            1. Downsample input by scale factor
            2. Encode: input -> latent space via cross-attention
            3. Process: refine in latent space via self-attention
            4. Enhance: latent FFN
            5. Decode: latent -> output via cross-attention
            6. Upsample back to original resolution
        """
        B, T, D = x_pos.shape
        
        # Select subset of latent tokens based on sequence length
        # Shape: (B, num_latents, emb_dim)
        latents = self.latent_token_pool[:num_latents].unsqueeze(0).expand(B, -1, -1)
        
        # Prepare input at the appropriate scale
        if scale == 1 or T < scale:
            x_scale = x_pos  # (B, T, emb_dim)
        else:
            # Downsample input for multi-scale processing
            # Shape: (B, T//scale, emb_dim)
            x_scale = x_pos[:, ::scale, :]
        
        # Phase 1: Input-to-Latent Cross-Attention (Encoding)
        # Compress input sequence to latent space
        # Query: (B, num_latents, emb_dim), K/V: (B, T//scale, emb_dim)
        # Output: (B, num_latents, emb_dim)
        latents_encoded, input_att = self.input_to_latent[scale_idx](
            query=latents, key=x_scale, value=x_scale, need_weights=return_att
        )
        latents = self.input_norms[scale_idx](latents + latents_encoded)
        
        # Phase 2: Latent-to-Latent Self-Attention (Processing)
        # Process information in compressed latent space
        # All: (B, num_latents, emb_dim)
        latents_processed, self_att = self.latent_self_attn[scale_idx](
            latents, latents, latents, need_weights=return_att
        )
        latents = self.latent_norms[scale_idx](latents + latents_processed)
        
        # Phase 3: Latent FFN (Further processing)
        # (B, num_latents, emb_dim) -> (B, num_latents, emb_dim)
        latents_ffn = self.latent_ffns[scale_idx](latents)
        latents = latents + latents_ffn
        
        # Phase 4: Latent-to-Output Cross-Attention (Decoding)
        # Decode from latent space back to sequence space
        # Query: (B, T//scale, emb_dim), K/V: (B, num_latents, emb_dim)
        # Output: (B, T//scale, emb_dim)
        output, output_att = self.latent_to_output[scale_idx](
            query=x_scale, key=latents, value=latents, need_weights=return_att
        )
        output = self.output_norms[scale_idx](x_scale + output)
        
        # Upsample back to original temporal resolution if needed
        if scale > 1 and T >= scale:
            # (B, T//scale, emb_dim) -> (B, emb_dim, T//scale) -> (B, emb_dim, T) -> (B, T, emb_dim)
            output = F.interpolate(
                output.transpose(1, 2), 
                size=T, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
        
        if return_att:
            # Package attention weights
            attn_dict = {
                'input_att': input_att if input_att is not None else None,
                'self_att': self_att if self_att is not None else None,
                'output_att': output_att if output_att is not None else None
            }
            # Average over heads if attention weights are available
            if self_att is not None:
                if self_att.dim() == 4:  # Expected: (B, num_heads, num_latents, num_latents)
                    attn_dict['self_att'] = self_att.mean(dim=1)  # (B, num_latents, num_latents)
                else:
                    print(f"Warning: Unexpected self_att shape {self_att.shape}")
            return output, attn_dict
            
        return output

    def forward(self, x: torch.Tensor, return_att: bool = False):
        """
        Forward pass through multi-scale latent attention.
        
        Args:
            x: Input tensor of shape (B, T, emb_dim)
            return_att: Whether to return attention weights
        
        Returns:
            output: Attended features (B, T, emb_dim)
            scale_attns: List of attention dicts for each scale (if return_att=True)
        
        Processing Flow:
            Input (B, T, emb_dim)
            ↓ Add positional encoding
            ├→ Scale 1 (full resolution) → (B, T, emb_dim)
            ├→ Scale 2 (half resolution) → (B, T, emb_dim)
            ├→ Scale 4 (quarter resolution) → (B, T, emb_dim)
            ↓ Concatenate → (B, T, emb_dim * 3)
            ↓ Fusion network → (B, T, emb_dim)
            ↓ Residual + Norm
            Output (B, T, emb_dim)
        """
        B, T, D = x.shape
        
        # Compute number of latent tokens based on sequence length
        num_latents = self._get_num_latents(T)
        
        # Add positional encoding: (B, T, emb_dim) -> (B, T, emb_dim)
        x_pos = self.pos_encoding(x)
        
        # Process each scale through latent attention
        scale_outputs = []
        scale_attns = [] if return_att else None
        
        for i, scale in enumerate(self.scales):
            if return_att:
                output, attn = self._process_scale(x_pos, i, scale, num_latents, return_att=True)
                scale_attns.append(attn)
            else:
                output = self._process_scale(x_pos, i, scale, num_latents)
            scale_outputs.append(output)  # Each: (B, T, emb_dim)
        
        # Fuse multi-scale outputs
        if len(scale_outputs) > 1:
            # Concatenate: [(B, T, emb_dim)] * 3 -> (B, T, emb_dim * 3)
            concatenated = torch.cat(scale_outputs, dim=-1)
            # Fusion: (B, T, emb_dim * 3) -> (B, T, emb_dim)
            fused = self.scale_fusion(concatenated)
        else:
            fused = scale_outputs[0]
        
        # Final processing with residual connection
        output = self.dropout(fused)
        output = self.final_norm(output + x_pos)  # (B, T, emb_dim)
        
        if return_att:
            return output, scale_attns
        return output


# =============================================================================
# SECTION 4: ATTENTION MECHANISMS (CBAM)
# =============================================================================

class EnhancedSEBlock(nn.Module):
    """
    Enhanced Squeeze-and-Excitation Block with dual pooling strategies.
    
    Combines both average and max pooling to capture different aspects of
    channel-wise feature importance.
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for bottleneck layer
    
    Input Shape:
        (B, C, H, W) - Feature map
    
    Output Shape:
        (B, C, H, W) - Recalibrated feature map
    
    Architecture:
        Input (B, C, H, W)
        ├→ AvgPool (B, C, 1, 1) → (B, C)
        ├→ MaxPool (B, C, 1, 1) → (B, C)
        ↓ Concat → (B, 2*C)
        ↓ FC: (B, 2*C) → (B, C//r) → (B, C)
        ↓ Sigmoid → weights (B, C, 1, 1)
        Output: Input * weights
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # (B, C, H, W) -> (B, C, 1, 1)
        
        reduced_dim = max(channels // reduction, 8)  # Minimum 8 channels
        
        # Squeeze and Excitation network
        # (B, 2*C) -> (B, reduced_dim) -> (B, C)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, reduced_dim, bias=False),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(reduced_dim, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel-wise attention recalibration.
        
        Args:
            x: Input feature map (B, C, H, W)
        
        Returns:
            torch.Tensor: Recalibrated features (B, C, H, W)
        """
        b, c, _, _ = x.size()
        
        # Global pooling: (B, C, H, W) -> (B, C)
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        weights = self.fc(combined).view(b, c, 1, 1)  # (B, C, 1, 1)
        
        # Apply channel-wise recalibration
        return x * weights.expand_as(x)


class EnhancedCBAM(nn.Module):
    """
    Enhanced Convolutional Block Attention Module (CBAM).
    
    Sequentially applies channel and spatial attention to refine feature maps.
    Uses multiple statistical features (mean, max, std, min) for robust spatial attention.
    
    Args:
        channels (int): Number of input channels
        reduction (int): Reduction ratio for channel attention
        kernel_size (int): Kernel size for spatial attention convolution
    
    Input Shape:
        (B, C, H, W) - Feature map
    
    Output Shape:
        (B, C, H, W) - Attention-refined feature map
    
    Architecture:
        Input (B, C, H, W)
        ↓ Channel Attention (EnhancedSEBlock)
        ↓ Intermediate (B, C, H, W)
        ↓ Spatial Attention
        │  ├→ Mean across channels (B, 1, H, W)
        │  ├→ Max across channels (B, 1, H, W)
        │  ├→ Std across channels (B, 1, H, W)
        │  ├→ Min across channels (B, 1, H, W)
        │  ↓ Concat → (B, 4, H, W)
        │  ↓ Conv layers → (B, 1, H, W)
        │  ↓ Sigmoid → spatial weights
        Output: Intermediate * spatial_weights
    """
    
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        
        # Channel attention module
        self.channel_attention = EnhancedSEBlock(channels, reduction)
        
        # Spatial attention convolution
        # (B, 4, H, W) -> (B, 2, H, W) -> (B, 1, H, W)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(4, 2, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(2),
            nn.GELU(),
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention (channel then spatial).
        
        Args:
            x: Input feature map (B, C, H, W)
        
        Returns:
            torch.Tensor: Attention-refined features (B, C, H, W)
        """
        # Step 1: Channel attention
        x = self.channel_attention(x)  # (B, C, H, W)
        
        # Step 2: Spatial attention
        # Compute multiple statistical features across channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        std_out = torch.std(x, dim=1, keepdim=True)  # (B, 1, H, W)
        min_out, _ = torch.min(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate: (B, 4, H, W)
        spatial_input = torch.cat([avg_out, max_out, std_out, min_out], dim=1)
        
        # Generate spatial attention weights: (B, 4, H, W) -> (B, 1, H, W)
        spatial_weights = self.spatial_conv(spatial_input)
        
        # Apply spatial attention
        return x * spatial_weights

    def get_attention_maps(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input feature map (B, C, H, W)
        
        Returns:
            channel_att: Channel attention weights (C,)
            spatial_att: Spatial attention map (H, W)
        """
        b, c, h, w = x.size()
        
        # Channel attention weights
        avg_out = self.channel_attention.avg_pool(x).view(b, c)  # (B, C)
        max_out = self.channel_attention.max_pool(x).view(b, c)  # (B, C)
        
        # Concatenate and compute weights: (B, 2*C) -> (B, C)
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2*C)
        channel_att = self.channel_attention.fc(combined).view(b, c).mean(dim=0).cpu().detach().numpy()  # (C,)
        
        # Spatial attention map
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        std_out = torch.std(x, dim=1, keepdim=True)  # (B, 1, H, W)
        min_out, _ = torch.min(x, dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_input = torch.cat([avg_out, max_out, std_out, min_out], dim=1)  # (B, 4, H, W)
        spatial_att = self.spatial_conv(spatial_input).squeeze(1).mean(dim=0).cpu().detach().numpy()  # (H, W)
        
        return channel_att, spatial_att


# =============================================================================
# SECTION 5: CNN BUILDING BLOCKS
# =============================================================================

class ResidualBlock(nn.Module):
    """
    Enhanced residual block with dilated convolutions and SE attention.
    
    Uses GELU activation (smoother than ReLU) and includes squeeze-excitation
    for channel-wise feature recalibration.
    
    Args:
        channels (int): Number of channels (input = output)
        dilation (int): Dilation rate for second convolution
    
    Input Shape:
        (B, C, H, W)
    
    Output Shape:
        (B, C, H, W)
    
    Architecture:
        Input (B, C, H, W)
        ↓ Conv3x3 + BN + GELU
        ↓ Dropout
        ↓ Conv3x3 (dilated) + BN
        ↓ SE Block
        ↓ Add residual
        Output (B, C, H, W)
    """
    
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        # First convolution: standard 3x3
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # Second convolution: dilated for larger receptive field
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, 
                              dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
        # Channel attention
        self.se = EnhancedSEBlock(channels)
        
        self.dropout = nn.Dropout2d(0.1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        Args:
            x: Input (B, C, H, W)
        
        Returns:
            Output (B, C, H, W)
        """
        residual = x
        
        # First conv block
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        
        # Second conv block
        out = self.bn2(self.conv2(out))
        
        # SE attention
        out = self.se(out)
        
        # Residual connection
        return self.activation(residual + out)


class MultiScaleConvBlock(nn.Module):
    """
    Multi-scale convolution block inspired by Inception architecture.
    
    Processes input through parallel branches with different receptive fields
    (1x1, 3x3, 5x5, 7x7) and concatenates results.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (must be divisible by 4)
    
    Input Shape:
        (B, in_channels, H, W)
    
    Output Shape:
        (B, out_channels, H, W)
    
    Architecture:
        Input (B, in_channels, H, W)
        ├→ Conv1x1 → (B, out_channels//4, H, W)
        ├→ Conv3x3 → (B, out_channels//4, H, W)
        ├→ Conv5x5 → (B, out_channels//4, H, W)
        ├→ Conv7x7 → (B, out_channels//4, H, W)
        ↓ Concat → (B, out_channels, H, W)
        ↓ SE Attention
        Output (B, out_channels, H, W)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")
        
        branch_channels = out_channels // 4
        
        # Branch 1: 1x1 convolution (no spatial context)
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )
        
        # Branch 2: 3x3 convolution (small receptive field)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )
        
        # Branch 3: 5x5 convolution (medium receptive field)
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 5, padding=2, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )
        
        # Branch 4: 7x7 convolution (large receptive field)
        self.branch7 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 7, padding=3, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.GELU()
        )
        
        # Channel attention for fused features
        self.attention = EnhancedSEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process through parallel branches and fuse.
        
        Args:
            x: Input (B, in_channels, H, W)
        
        Returns:
            Output (B, out_channels, H, W)
        """
        # Concatenate all branches: 4 * (B, out_channels//4, H, W) -> (B, out_channels, H, W)
        out = torch.cat([
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch7(x)
        ], dim=1)
        
        # Apply channel attention
        return self.attention(out)


# =============================================================================
# SECTION 6: SPATIAL FEATURE EXTRACTOR
# =============================================================================

class SpatialFeatureExtractor(nn.Module):
    """
    CNN-based spatial feature extraction from hyperspectral image patches.
    
    Processes each spectral band independently through a shared CNN to extract
    spatial features, then projects to embedding space.
    
    Args:
        config (ModelConfig): Model configuration
    
    Input Shape:
        (B, T, H, W) where:
            B = batch size
            T = number of spectral bands
            H, W = spatial patch dimensions
    
    Output Shape:
        (B, T, emb_dim) - Sequence of spatial feature embeddings
    
    Architecture:
        Input (B, T, H, W) → Reshape → (B*T, 1, H, W)
        ↓ Stem (Multi-scale or standard conv)
        ↓ 4x Residual Blocks with increasing dilation
        ↓ CBAM Attention (optional)
        ↓ Global Average + Max Pooling → (B*T, base_c*2)
        ↓ Projection → (B*T, emb_dim)
        ↓ Reshape → (B, T, emb_dim)
        Output
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.use_checkpoint = config.use_checkpoint
        
        # Stem: Initial convolution layer(s)
        if config.use_multiscale:
            # Multi-scale inception-style stem
            # (B*T, 1, H, W) -> (B*T, base_c, H, W)
            self.stem = MultiScaleConvBlock(1, config.base_c)
        else:
            # Standard single-scale stem
            self.stem = nn.Sequential(
                nn.Conv2d(1, config.base_c, 3, padding=1, bias=False),
                nn.BatchNorm2d(config.base_c),
                nn.GELU()
            )
        
        # Stack of residual blocks with increasing dilation
        # Each maintains shape: (B*T, base_c, H, W)
        self.stages = nn.ModuleList([
            ResidualBlock(config.base_c, dilation=1 + i) for i in range(4)
        ])
        
        # Optional spatial-channel attention
        # (B*T, base_c, H, W) -> (B*T, base_c, H, W)
        self.attention = EnhancedCBAM(config.base_c) if config.use_cbam else nn.Identity()
        
        # Global pooling layers
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B*T, base_c, H, W) -> (B*T, base_c, 1, 1)
        self.gmp = nn.AdaptiveMaxPool2d(1)  # (B*T, base_c, H, W) -> (B*T, base_c, 1, 1)
        
        # Projection to embedding space
        # (B*T, base_c*2) -> (B*T, emb_dim)
        self.frame_proj = nn.Sequential(
            nn.Linear(config.base_c * 2, config.emb_dim),
            nn.LayerNorm(config.emb_dim)
        )

    def _forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Internal spatial processing function.
        
        Args:
            x: (B*T, 1, H, W)
        
        Returns:
            (B*T, emb_dim)
        """
        # Stem: (B*T, 1, H, W) -> (B*T, base_c, H, W)
        h = self.stem(x)
        
        # Residual stages: maintain (B*T, base_c, H, W)
        for stage in self.stages:
            h = stage(h)
        
        # Attention: (B*T, base_c, H, W) -> (B*T, base_c, H, W)
        h = self.attention(h)
        
        # Global pooling: (B*T, base_c, H, W) -> (B*T, base_c)
        gap = self.gap(h).flatten(1)  # (B*T, base_c)
        gmp = self.gmp(h).flatten(1)  # (B*T, base_c)
        h = torch.cat([gap, gmp], dim=1)  # (B*T, base_c*2)
        
        # Project: (B*T, base_c*2) -> (B*T, emb_dim)
        return self.frame_proj(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial features from hyperspectral patches.
        
        Args:
            x: Input tensor (B, T, H, W)
        
        Returns:
            Feature embeddings (B, T, emb_dim)
        """
        B, T, H, W = x.shape
        
        # Reshape: treat each band as separate image
        # (B, T, H, W) -> (B*T, 1, H, W)
        x = x.view(B * T, 1, H, W)
        
        # Process with optional gradient checkpointing
        if self.use_checkpoint and self.training:
            h = checkpoint(self._forward_spatial, x)
        else:
            h = self._forward_spatial(x)
        
        # Reshape back: (B*T, emb_dim) -> (B, T, emb_dim)
        return h.view(B, T, -1)


# =============================================================================
# SECTION 7: SPECTRAL ENCODER LAYER
# =============================================================================

class SpectralEncoderLayer(nn.Module):
    """
    Transformer-style encoder layer for spectral sequence processing.
    
    Combines RNN modules (LSTM + GRU) with multi-scale latent attention
    and feed-forward network in a residual manner.
    
    Args:
        config (ModelConfig): Model configuration
    
    Input Shape:
        (B, T, emb_dim) where T is the number of spectral bands
    
    Output Shape:
        (B, T, emb_dim)
    
    Architecture:
        Input (B, T, emb_dim)
        ↓
        ├→ Bi-LSTM (B, T, emb_dim) ─┐
        ├→ Bi-GRU (B, T, emb_dim) ──┤ Residual add
        ├→ Multi-Scale Latent Attention ┘
        ↓ LayerNorm
        ↓
        ├→ FFN: emb_dim → emb_dim*4 → emb_dim
        ↓ Residual add + LayerNorm
        Output (B, T, emb_dim)
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        emb_dim = config.emb_dim
        
        # Recurrent layers for temporal/spectral modeling
        # Both produce (B, T, emb_dim) output through bidirectional processing
        self.spectral_layers = nn.ModuleList([
            nn.LSTM(
                emb_dim, emb_dim // 2,  # Hidden size = emb_dim//2 per direction
                num_layers=1,
                batch_first=True,
                bidirectional=True,  # Total output: emb_dim
                dropout=0.1
            ),
            nn.GRU(
                emb_dim, emb_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.1
            )
        ])
        
        # Multi-scale spectral attention module
        # (B, T, emb_dim) -> (B, T, emb_dim)
        self.spectral_attention = MultiScaleSpectralLatentAttention(emb_dim)
        
        # Layer normalizations
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
        # Position-wise Feed-Forward Network
        # (B, T, emb_dim) -> (B, T, emb_dim*4) -> (B, T, emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim * 4, emb_dim)
        )

    def forward(self, x: torch.Tensor, return_att: bool = False):
        """
        Process spectral sequence through encoder layer.
        
        Args:
            x: Input (B, T, emb_dim)
            return_att: Whether to return attention weights
        
        Returns:
            output: Processed features (B, T, emb_dim)
            attns: Attention weights (if return_att=True)
        """
        # --- Sub-layer 1: RNN + Attention ---
        h = x
        
        # Apply RNN layers with residual connections
        for spectral_layer in self.spectral_layers:
            # Each RNN: (B, T, emb_dim) -> (B, T, emb_dim)
            h_spec, _ = spectral_layer(h)
            h = h + h_spec  # Residual connection
        
        # Apply multi-scale attention
        if return_att:
            h, attns = self.spectral_attention(h, return_att=True)
        else:
            h = self.spectral_attention(h)
        
        # Residual connection + normalization
        x = self.norm1(x + h)
        
        # --- Sub-layer 2: Feed-Forward Network ---
        h = self.ffn(x)  # (B, T, emb_dim) -> (B, T, emb_dim)
        x = self.norm2(x + h)  # Residual + Norm
        
        if return_att:
            return x, attns
        return x


# =============================================================================
# SECTION 8: COMPLETE CLARESNET MODEL
# =============================================================================

class CLAReSNet(nn.Module):
    """
    Convolutional Latent Attention Residual Spectral Network (CLAReSNet).
    
    A comprehensive deep learning architecture for hyperspectral image
    classification that combines:
    - CNN-based spatial feature extraction
    - Multi-scale spectral attention with latent compression
    - Transformer-style encoder layers with RNN modules
    - Cross-layer feature fusion
    
    Args:
        n_classes (int): Number of classification classes
        config (ModelConfig): Model architecture configuration
    
    Input Shape:
        (B, T, H, W) where:
            B = batch size
            T = number of spectral bands
            H, W = spatial patch dimensions (typically 11x11)
    
    Output Shape:
        (B, n_classes) - Class logits or probabilities
    
    Full Architecture Flow:
        Input (B, T, H, W)
        ↓
        ┌─ Spatial Feature Extractor (CNN) ─┐
        │  Processes each band independently  │
        └────────────────────────────────────┘
        ↓ (B, T, emb_dim)
        ↓
        ┌─ Stack of Spectral Encoder Layers ─┐
        │  Layer 1: RNN + Attention + FFN     │
        │  Layer 2: RNN + Attention + FFN     │
        │  ...                                 │
        │  Layer N: RNN + Attention + FFN     │
        │  (with residual connections)         │
        └────────────────────────────────────┘
        ↓ [(B, T, emb_dim)] * N layers
        ↓
        ┌─ Cross-Layer Attention Fusion ─────┐
        │  Combines features from all layers  │
        │  using multi-head attention         │
        └────────────────────────────────────┘
        ↓ (B, emb_dim)
        ↓
        ┌─ Classification Head ──────────────┐
        │  LayerNorm → Dropout → FC → GELU   │
        │  → Dropout → FC → Logits            │
        └────────────────────────────────────┘
        ↓
        Output (B, n_classes)
    """
    
    def __init__(self, n_classes: int, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. Spatial Feature Extractor (CNN Stem)
        # Converts spatial patches to feature embeddings
        # (B, T, H, W) -> (B, T, emb_dim)
        self.spatial_extractor = SpatialFeatureExtractor(config)
        
        # 2. Stack of Spectral Encoder Layers
        # Each layer: (B, T, emb_dim) -> (B, T, emb_dim)
        self.encoder_layers = nn.ModuleList([
            SpectralEncoderLayer(config) 
            for _ in range(config.n_encoder_layers)
        ])
        
        # 3. Cross-Attention for fusing features from all layers
        if config.n_encoder_layers > 1:
            # Query: last layer summary (B, 1, emb_dim)
            # Key/Value: all layer summaries (B, n_layers, emb_dim)
            # Output: (B, 1, emb_dim)
            self.cross_attn = nn.MultiheadAttention(
                config.emb_dim, num_heads=8, dropout=0.1, batch_first=True
            )
            self.cross_norm = nn.LayerNorm(config.emb_dim)
        
        # 4. Final Classification Head
        # (B, emb_dim) -> (B, n_classes)
        self.head = nn.Sequential(
            nn.LayerNorm(config.emb_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.emb_dim, config.emb_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout * 0.5),
            nn.Linear(config.emb_dim // 2, n_classes)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize model weights using best practices.
        - Linear layers: truncated normal initialization
        - LayerNorm: weight=1, bias=0
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, return_feat: bool = False, 
                return_att: bool = False, return_prob: bool = False):
        """
        Forward pass through CLAReSNet.
        
        Args:
            x: Input tensor (B, T, H, W)
            return_feat: If True, return features instead of logits
            return_att: If True, return attention weights
            return_prob: If True, return probabilities instead of logits
        
        Returns:
            If return_feat=True: (B, emb_dim) feature vectors
            If return_prob=True: (B, n_classes) class probabilities
            If return_att=True: (logits, attention_dicts)
            Otherwise: (B, n_classes) class logits
        
        Processing Steps:
            1. Extract spatial features from patches
            2. Process through encoder layers with residual connections
            3. Fuse multi-layer features via cross-attention
            4. Global pooling and classification
        """
        # Step 1: Spatial Feature Extraction
        # (B, T, H, W) -> (B, T, emb_dim)
        x = self.spatial_extractor(x)
        
        # Step 2: Process through Spectral Encoder Layers
        all_layer_outputs = []
        all_attns = [] if return_att else None
        residual = x
        
        for layer in self.encoder_layers:
            if return_att:
                # (B, T, emb_dim) -> (B, T, emb_dim), attention_dict
                layer_out, attn = layer(residual, return_att=True)
                all_attns.append(attn)
            else:
                # (B, T, emb_dim) -> (B, T, emb_dim)
                layer_out = layer(residual)
            
            # Main residual connection between layers
            layer_out = layer_out + residual
            all_layer_outputs.append(layer_out)
            residual = layer_out  # Update for next layer
        
        # Step 3: Cross-Layer Feature Fusion
        if self.config.n_encoder_layers > 1:
            # Compute mean representation for each layer (like a [CLS] token)
            # [(B, T, emb_dim)] * N -> [(B, emb_dim)] * N
            layer_summaries = [torch.mean(out, dim=1) for out in all_layer_outputs]
            
            # Stack: [(B, emb_dim)] * N -> (B, N, emb_dim)
            layer_stack = torch.stack(layer_summaries, dim=1)
            
            # Last layer summary as query: (B, emb_dim) -> (B, 1, emb_dim)
            query = layer_stack[:, -1, :].unsqueeze(1)
            
            # Cross-attention: fuse all layer information
            # Query: (B, 1, emb_dim), K/V: (B, N, emb_dim)
            # Output: (B, 1, emb_dim)
            attn_out, _ = self.cross_attn(
                query=query, key=layer_stack, value=layer_stack
            )
            
            # Final features with residual
            # (B, emb_dim)
            final_features = self.cross_norm(
                attn_out.squeeze(1) + layer_summaries[-1]
            )
        else:
            # If only one layer, just take its mean
            # (B, T, emb_dim) -> (B, emb_dim)
            final_features = torch.mean(all_layer_outputs[0], dim=1)
        
        # Step 4: Return features if requested
        if return_feat:
            return final_features
        
        # Step 5: Classification Head
        # (B, emb_dim) -> (B, n_classes)
        logits = self.head(final_features)
        
        # Return probabilities if requested
        if return_prob:
            probs = F.softmax(logits, dim=1)
            return probs
        
        # Return with attention weights if requested
        if return_att:
            return logits, all_attns
        
        return logits
