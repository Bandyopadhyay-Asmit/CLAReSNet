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
import torch.optim as optim
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score,
                             balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
                             precision_recall_fscore_support, roc_auc_score, precision_recall_curve,
                             top_k_accuracy_score, adjusted_rand_score, roc_curve, auc)
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm

# Set visualization style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

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


# =============================================================================
# SECTION 9: DATASET AND DATA LOADING
# =============================================================================

class HyperspectralDataset(Dataset):
    """
    PyTorch Dataset for hyperspectral image patch extraction.
    
    Extracts spatial patches centered at labeled pixel locations from
    the full hyperspectral image.
    
    Args:
        reduced_image: Hyperspectral image array (H, W, C)
        samples_df: DataFrame with columns ['row', 'col', 'label']
        patch_size: Spatial patch size (default: 11x11)
        augment: Whether to apply data augmentation
    
    Returns:
        patch: Tensor of shape (C, patch_size, patch_size)
        label: Integer label (0-indexed)
    """
    
    def __init__(self, reduced_image: np.ndarray, samples_df: pd.DataFrame, 
                 patch_size: int = 11, augment: bool = False):
        """
        Initialize dataset with padding for edge pixels.
        
        The image is padded so that patches can be extracted even for
        pixels at the edges.
        """
        self.pad = patch_size // 2
        self.augment = augment
        self.training = False
        
        # Pad image with reflection to handle boundary pixels
        # (H, W, C) -> (H+2*pad, W+2*pad, C)
        self.padded_image = np.pad(
            reduced_image,
            ((self.pad, self.pad), (self.pad, self.pad), (0, 0)),
            mode='reflect'
        )
        
        # Store sample locations and adjust for padding
        self.samples = samples_df.reset_index(drop=True)
        self.samples['pad_row'] = self.samples['row'] + self.pad
        self.samples['pad_col'] = self.samples['col'] + self.pad

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a patch and its label.
        
        Args:
            idx: Sample index
        
        Returns:
            patch: (C, patch_size, patch_size) tensor
            label: (1,) tensor with class index (0-indexed)
        """
        # Get padded coordinates
        pr = self.samples.loc[idx, 'pad_row']
        pc = self.samples.loc[idx, 'pad_col']
        
        # Extract patch centered at (pr, pc)
        # (patch_size, patch_size, C)
        patch = self.padded_image[
            pr - self.pad:pr + self.pad + 1,
            pc - self.pad:pc + self.pad + 1,
            :
        ]
        
        # Transpose: (H, W, C) -> (C, H, W)
        patch = np.transpose(patch, (2, 0, 1))
        patch = torch.from_numpy(patch).float()
        
        # Convert label to 0-indexed
        label = self.samples.loc[idx, 'label'] - 1

        # Data augmentation (only during training)
        if self.augment and self.training:
            # Gaussian noise
            if np.random.rand() < 0.5:
                patch = patch + torch.randn_like(patch) * 0.05
            
            # Random rotation (90, 180, or 270 degrees)
            if np.random.rand() < 0.5:
                k = np.random.randint(1, 4)
                patch = torch.rot90(patch, k=k, dims=(1, 2))
            
            # Random flip (horizontal or vertical)
            if np.random.rand() < 0.5:
                flip_dim = np.random.choice([1, 2])
                patch = torch.flip(patch, dims=[flip_dim])

        return patch, torch.tensor(label).long()

    def set_training(self, training: bool):
        """Set training mode (affects augmentation)."""
        self.training = training


def load_dataset_from_csv(csv_path: str, height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load hyperspectral dataset from CSV file.
    
    Args:
        csv_path: Path to CSV file
        height: Image height
        width: Image width
    
    Returns:
        spectral_image: (H, W, C) array of spectral data
        label_image: (H, W) array of class labels
    
    CSV Format:
        - Each row is a pixel
        - Columns are [band_1, band_2, ..., band_C, label]
        - Labels are 1-indexed (0 means unlabeled)
    """
    df = pd.read_csv(csv_path)
    
    # Separate spectral bands and labels
    bands = df.iloc[:, :-1].values.astype(np.float32)  # (H*W, C)
    labels = df.iloc[:, -1].values.astype(np.int32)    # (H*W,)
    
    # Reshape to spatial format
    spectral_image = bands.reshape(height, width, bands.shape[1])  # (H, W, C)
    label_image = labels.reshape(height, width)                     # (H, W)
    
    return spectral_image, label_image


def get_labeled_samples(label_image: np.ndarray) -> pd.DataFrame:
    """
    Extract labeled pixel locations from label image.
    
    Args:
        label_image: (H, W) array with labels (0 = unlabeled)
    
    Returns:
        DataFrame with columns ['row', 'col', 'label']
    """
    # Find all labeled pixels (label > 0)
    rows, cols = np.where(label_image > 0)
    labels = label_image[rows, cols]
    
    return pd.DataFrame({
        'row': rows,
        'col': cols,
        'label': labels
    })


def get_class_weights(samples_df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """
    Compute inverse frequency class weights for balanced training.
    
    Args:
        samples_df: DataFrame with 'label' column
        device: Device to place tensor on
    
    Returns:
        Tensor of shape (n_classes,) with normalized weights
    
    Formula:
        weight[c] = 1 / count[c]
        Then normalize so weights sum to n_classes
    """
    class_counts = samples_df['label'].value_counts().sort_index()
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(class_counts)
    return torch.tensor(weights.values).float().to(device)


# =============================================================================
# SECTION 10: TRAINING FUNCTIONS
# =============================================================================

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                   criterion: nn.Module, device: torch.device, desc: str = "Train") -> Tuple[float, float]:
    """
    Train model for one epoch.
    
    Args:
        model: CLAReSNet model
        loader: Training data loader
        optimizer: Optimizer (e.g., AdamW)
        criterion: Loss function (e.g., CrossEntropyLoss)
        device: Device to train on
        desc: Description for progress bar
    
    Returns:
        epoch_loss: Average loss for the epoch
        epoch_acc: Training accuracy (%)
    
    Training Loop:
        For each batch:
            1. Forward pass
            2. Compute loss
            3. Backward pass
            4. Update weights
            5. Track metrics
    """
    model.train()
    loader.dataset.set_training(True)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in tqdm(loader, desc=desc):
        # Move to device
        # inputs: (B, C, H, W), labels: (B,)
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass: (B, C, H, W) -> (B, n_classes)
        outputs = model(inputs)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        _, pred = outputs.max(1)  # Get predicted class
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = 100.0 * correct / total
    
    return epoch_loss, epoch_acc


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module,
            device: torch.device, desc: str = "Eval", 
            return_prob: bool = False) -> Tuple:
    """
    Evaluate model on validation/test set.
    
    Args:
        model: CLAReSNet model
        loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        desc: Description for progress bar
        return_prob: Whether to return class probabilities
    
    Returns:
        acc: Accuracy (%)
        all_labels: Ground truth labels (N,)
        all_preds: Predicted labels (N,)
        avg_loss: Average loss
        all_probs: Class probabilities (N, n_classes) if return_prob=True
    """
    model.eval()
    loader.dataset.set_training(False)
    
    correct = 0
    running_loss = 0.0
    total = 0
    all_labels = []
    all_preds = []
    all_probs = [] if return_prob else None
    
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=desc):
            # Move to device
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)  # (B, n_classes)
            
            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Get predictions
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
            
            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            
            if return_prob:
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                all_probs.extend(probs)
    
    acc = 100.0 * correct / total
    avg_loss = running_loss / len(loader)
    
    out = [acc, np.array(all_labels), np.array(all_preds), avg_loss]
    if return_prob:
        out.append(np.array(all_probs))
    
    return tuple(out)


# =============================================================================
# SECTION 11: METRICS AND VISUALIZATION
# =============================================================================

class MetricsTracker:
    """
    Track training metrics across epochs.
    
    Stores and visualizes:
    - Training loss and accuracy
    - Validation loss and accuracy
    
    Methods:
        update(): Add metrics for an epoch
        plot_training_curves(): Visualize training progress
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Initialize/reset all metric lists."""
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.epochs = []
    
    def update(self, epoch: int, train_loss: float, train_acc: float,
               val_loss: float, val_acc: float):
        """
        Add metrics for an epoch.
        
        Args:
            epoch: Epoch number
            train_loss: Training loss
            train_acc: Training accuracy (%)
            val_loss: Validation loss
            val_acc: Validation accuracy (%)
        """
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
    
    def plot_training_curves(self, title: str = "Training Curves", save_path: Optional[str] = None):
        """
        Plot training and validation curves.
        
        Args:
            title: Plot title
            save_path: Path to save figure (optional)
        
        Creates a 2-subplot figure:
            Left: Loss curves (train & val)
            Right: Accuracy curves (train & val)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{title} - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracy
        ax2.plot(self.epochs, self.train_accs, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(self.epochs, self.val_accs, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title(f'{title} - Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                            y_prob: Optional[np.ndarray] = None,
                            n_classes: Optional[int] = None) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth labels (N,)
        y_pred: Predicted labels (N,)
        y_prob: Class probabilities (N, n_classes) - optional
        n_classes: Number of classes
    
    Returns:
        Dictionary containing:
        - accuracy: Overall accuracy
        - balanced_accuracy: Balanced accuracy (accounts for class imbalance)
        - cohen_kappa: Cohen's kappa coefficient
        - matthews_corrcoef: Matthews correlation coefficient
        - ari: Adjusted Rand Index
        - per_class_precision/recall/f1/support
        - macro/micro/weighted precision/recall/f1
        - roc_auc_ovr/ovo: ROC AUC scores (if probabilities provided)
        - top3_acc, top5_acc: Top-k accuracy (if probabilities provided)
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['matthews_corrcoef'] = matthews_corrcoef(y_true, y_pred)
    metrics['ari'] = adjusted_rand_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    metrics['per_class_precision'] = precision
    metrics['per_class_recall'] = recall
    metrics['per_class_f1'] = f1
    metrics['per_class_support'] = support
    
    # Macro-averaged metrics (unweighted mean across classes)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    metrics['precision_macro'] = p_macro
    metrics['recall_macro'] = r_macro
    metrics['f1_macro'] = f1_macro
    
    # Micro-averaged metrics (global averaging)
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )
    metrics['precision_micro'] = p_micro
    metrics['recall_micro'] = r_micro
    metrics['f1_micro'] = f1_micro
    
    # Weighted-averaged metrics (weighted by support)
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['precision_weighted'] = p_weighted
    metrics['recall_weighted'] = r_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # Probability-based metrics
    if y_prob is not None and n_classes is not None:
        # ROC AUC: One-vs-Rest and One-vs-One
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
        metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
        
        # Top-k accuracy
        metrics['top3_acc'] = top_k_accuracy_score(y_true, y_prob, k=3)
        metrics['top5_acc'] = top_k_accuracy_score(y_true, y_prob, k=5)
    
    return metrics


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: Optional[List[str]] = None,
                         title: str = "Confusion Matrix",
                         save_path: Optional[str] = None):
    """
    Plot normalized confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names for axis labels
        title: Plot title
        save_path: Path to save figure
    
    Returns:
        Confusion matrix array (n_classes, n_classes)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    
    # Normalize by true class (row-wise)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
    
    # Plot heatmap
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names if class_names else range(len(cm)),
                yticklabels=class_names if class_names else range(len(cm)))
    
    plt.title(f'{title} (Normalized)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    
    return cm


def plot_per_class_metrics(metrics: Dict, class_names: Optional[List[str]] = None,
                           title: str = "Per-Class Metrics",
                           save_path: Optional[str] = None):
    """
    Plot per-class precision, recall, and F1-score as bar charts.
    
    Args:
        metrics: Dictionary from compute_detailed_metrics()
        class_names: List of class names
        title: Plot title
        save_path: Path to save figure
    """
    n_classes = len(metrics['per_class_precision'])
    x_labels = class_names if class_names else [f'Class {i+1}' for i in range(n_classes)]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Precision
    bars1 = ax1.bar(range(n_classes), metrics['per_class_precision'])
    ax1.set_title('Per-Class Precision')
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Precision')
    ax1.set_xticks(range(n_classes))
    ax1.set_xticklabels(x_labels, rotation=45, ha="right")
    ax1.grid(True, alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # Recall
    bars2 = ax2.bar(range(n_classes), metrics['per_class_recall'])
    ax2.set_title('Per-Class Recall')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Recall')
    ax2.set_xticks(range(n_classes))
    ax2.set_xticklabels(x_labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    # F1-Score
    bars3 = ax3.bar(range(n_classes), metrics['per_class_f1'])
    ax3.set_title('Per-Class F1-Score')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('F1-Score')
    ax3.set_xticks(range(n_classes))
    ax3.set_xticklabels(x_labels, rotation=45, ha="right")
    ax3.grid(True, alpha=0.3)
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


def plot_tsne_features(model: nn.Module, loader: DataLoader, device: torch.device,
                      n_classes: int, title: str = "TSNE Features",
                      save_path: Optional[str] = None):
    """
    Visualize learned features using t-SNE dimensionality reduction.
    
    Args:
        model: Trained CLAReSNet model
        loader: Data loader
        device: Device
        n_classes: Number of classes
        title: Plot title
        save_path: Path to save figure
    
    Process:
        1. Extract features from model (before classification head)
        2. Apply t-SNE to reduce to 2D
        3. Plot colored by class
    """
    model.eval()
    features = []
    labels = []
    
    # Extract features
    with torch.no_grad():
        for inputs, y in loader:
            inputs = inputs.to(device)
            # Get features before classification: (B, emb_dim)
            feats = model(inputs, return_feat=True)
            features.append(feats.cpu().numpy())
            labels.extend(y.cpu().numpy())
    
    features = np.vstack(features)  # (N, emb_dim)
    
    # Apply t-SNE: (N, emb_dim) -> (N, 2)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(features)
    
    # Plot
    plt.figure(figsize=(8, 6))
    cmap = plt.colormaps.get_cmap("tab20").resampled(n_classes)
    scatter = plt.scatter(tsne[:, 0], tsne[:, 1], c=labels, cmap=cmap, s=10)
    
    # Manual legend
    handles = [mpatches.Patch(color=cmap(i), label=str(i)) for i in range(n_classes)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", title="Classes")
    
    plt.xlabel('First t-SNE Component')
    plt.ylabel('Second t-SNE Component')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


def plot_pca_visualization(model: nn.Module, loader: DataLoader, device: torch.device,
                          n_classes: int, title: str = "PCA Features",
                          save_path: Optional[str] = None):
    """
    Visualize learned features using PCA.
    
    Similar to t-SNE but uses linear PCA instead.
    """
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        for inputs, y in loader:
            inputs = inputs.to(device)
            feats = model(inputs, return_feat=True)
            features.append(feats.cpu().numpy())
            labels.extend(y.cpu().numpy())
    
    features = np.vstack(features)
    
    # Apply PCA
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)
    
    plt.figure(figsize=(8, 6))
    cmap = plt.colormaps.get_cmap("tab20").resampled(n_classes)
    scatter = plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap=cmap, s=10)
    
    handles = [mpatches.Patch(color=cmap(i), label=str(i)) for i in range(n_classes)]
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left", title="Classes")
    
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


def plot_classification_map(model: nn.Module, spectral_image: np.ndarray,
                           device: torch.device, patch_size: int, n_components: int,
                           dataset_name: str = "Dataset",
                           save_path: Optional[str] = None):
    """
    Generate and visualize full classification map.
    
    Args:
        model: Trained model
        spectral_image: Full hyperspectral image (H, W, C)
        device: Device
        patch_size: Patch size used during training
        n_components: Number of PCA components
        dataset_name: Dataset name for title
        save_path: Path to save figure
    
    Process:
        1. Create dataset with all pixels
        2. Predict class for each pixel
        3. Reshape to spatial format
        4. Visualize as colored map
    """
    height, width, _ = spectral_image.shape
    
    # Create dataset with all pixels
    all_pixels_df = pd.DataFrame([
        (r, c, 0) for r in range(height) for c in range(width)
    ], columns=['row', 'col', 'label'])
    
    dataset = HyperspectralDataset(spectral_image, all_pixels_df, patch_size)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # Predict all pixels
    all_preds = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc=f"Generating map for {dataset_name}"):
            inputs = inputs.to(device)
            preds = model(inputs).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
    
    # Reshape to spatial format and convert to 1-indexed
    pred_map = np.array(all_preds).reshape(height, width) + 1
    
    # Visualize
    plt.figure(figsize=(10, 10))
    plt.imshow(pred_map, cmap='tab20', vmin=0, vmax=16)
    plt.colorbar(ticks=range(17), label='Class')
    plt.title(f'{dataset_name} Classification Map')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()


# =============================================================================
# SECTION 12: MAIN EXECUTION PIPELINE
# =============================================================================

def comprehensive_evaluation(model: nn.Module, loader: DataLoader, device: torch.device,
                            dataset_name: str = "Dataset",
                            class_names: Optional[List[str]] = None,
                            plot_cm: bool = True, plot_metrics: bool = True,
                            plot_features: bool = True,
                            spectral_image: Optional[np.ndarray] = None,
                            label_image: Optional[np.ndarray] = None,
                            samples_df: Optional[pd.DataFrame] = None,
                            patch_size: int = 11, n_components: int = 30) -> Dict:
    """
    Perform comprehensive model evaluation with visualizations.
    
    Args:
        model: Trained model
        loader: Test data loader
        device: Device
        dataset_name: Name for titles
        class_names: List of class names
        plot_cm: Whether to plot confusion matrix
        plot_metrics: Whether to plot per-class metrics
        plot_features: Whether to plot feature visualizations
        spectral_image: Full image for classification map
        label_image: Ground truth labels
        samples_df: Test samples dataframe
        patch_size: Patch size
        n_components: Number of PCA components
    
    Returns:
        Dictionary of detailed metrics
    
    Generates:
        - Confusion matrix
        - Per-class metrics plots
        - t-SNE and PCA visualizations
        - Classification map
        - ROC and PR curves
    """
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE EVALUATION: {dataset_name}")
    print(f"{'='*50}")
    
    # Evaluate and get predictions
    criterion = nn.CrossEntropyLoss()
    acc, y_true, y_pred, _, y_prob = evaluate(
        model, loader, criterion, device, desc="Eval", return_prob=True
    )
    
    # Compute detailed metrics
    detailed_metrics = compute_detailed_metrics(y_true, y_pred, y_prob, len(class_names))
    
    # Print metrics
    print(f"\nDetailed Metrics for {dataset_name}:")
    print(f"  Accuracy: {detailed_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {detailed_metrics['balanced_accuracy']:.4f}")
    print(f"  Cohen's Kappa: {detailed_metrics['cohen_kappa']:.4f}")
    print(f"  Matthews Corr Coef: {detailed_metrics['matthews_corrcoef']:.4f}")
    print(f"  Macro F1-Score: {detailed_metrics['f1_macro']:.4f}")
    print(f"  Weighted F1-Score: {detailed_metrics['f1_weighted']:.4f}")
    print(f"  ROC AUC (OVR): {detailed_metrics['roc_auc_ovr']:.4f}")
    print(f"  Top-3 Accuracy: {detailed_metrics['top3_acc']:.4f}")
    
    # Plot visualizations
    if plot_cm:
        plot_confusion_matrix(y_true, y_pred, class_names,
                            title=f"{dataset_name} - Confusion Matrix",
                            save_path=f"{dataset_name}_confusion_matrix.png")
    
    if plot_metrics:
        plot_per_class_metrics(detailed_metrics, class_names,
                              title=f"{dataset_name} - Per-Class Metrics",
                              save_path=f"{dataset_name}_per_class_metrics.png")
    
    if plot_features:
        plot_tsne_features(model, loader, device, len(class_names),
                          title=f"{dataset_name} - TSNE Features",
                          save_path=f"{dataset_name}_tsne_features.png")
        plot_pca_visualization(model, loader, device, len(class_names),
                              title=f"{dataset_name} - PCA Features",
                              save_path=f"{dataset_name}_pca_features.png")
    
    if spectral_image is not None:
        plot_classification_map(model, spectral_image, device, patch_size,
                               n_components, dataset_name,
                               save_path=f"{dataset_name}_classification_map.png")
    
    return detailed_metrics

"""
CLAReSNet Training Pipeline for Salinas Hyperspectral Dataset
==============================================================

This script provides a complete training pipeline for the Salinas dataset.

Dataset Dimensions:
- Spatial: 512 × 217 pixels (111,104 total pixels)
- Spectral bands: 204 (after water absorption removal)
- Classes: 16 land cover types
- Labeled samples: ~54,129 pixels

Input/Output Dimensions Throughout Pipeline:
--------------------------------------------
1. Raw CSV data: (111104, 205) - [204 bands + 1 label column]
2. Reshaped spectral: (512, 217, 204)
3. After PCA: (512, 217, 30)
4. Training batch: (16, 30, 11, 11) - [batch_size, bands, height, width]
5. Model output: (16, 16) - [batch_size, n_classes]
"""

# =============================================================================
# MAIN PIPELINE FOR SALINAS DATASET
# =============================================================================

if __name__ == "__main__":
    """
    Main training pipeline for Salinas hyperspectral dataset.
    
    Pipeline Steps:
    ---------------
    1. Setup device and configuration
    2. Load raw dataset from CSV
    3. Apply PCA dimensionality reduction (204 → 30 bands)
    4. Extract labeled samples and create train/val/test splits
    5. Create data loaders with augmentation
    6. Initialize CLAReSNet model
    7. Train for specified epochs with validation
    8. Comprehensive evaluation on test set
    
    Data Flow:
    ----------
    CSV (111104, 205)
    → Reshape → (512, 217, 204)
    → PCA → (512, 217, 30)
    → Extract patches → (B, 30, 11, 11)
    → Model → (B, 16)
    → Loss & Backprop
    """
    
    # =========================================================================
    # STEP 1: DEVICE AND CONFIGURATION SETUP
    # =========================================================================
    
    # Determine device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Model configuration
    # Creates a ModelConfig dataclass with architecture hyperparameters
    config = ModelConfig(
        emb_dim=256,         # Embedding dimension for all features
        base_c=64,           # Base number of channels in CNN layers
        n_encoder_layers=3,  # Number of spectral encoder layers to stack
        dropout=0.5          # Dropout probability for regularization
    )
    
    # Training hyperparameters
    patch_size = 11      # Extract 11×11 spatial patches around each pixel
    n_components = 30    # Reduce 204 spectral bands to 30 PCA components
    PRETRAINED_PATH = "claret_salinas_pretrained.pth"  # Model save path
    
    # Class names for Salinas dataset (16 land cover types)
    # Used for visualization and result reporting
    salinas_class_names = [
        'Brocoli_green_weeds_1',      # Class 1
        'Brocoli_green_weeds_2',      # Class 2
        'Fallow',                      # Class 3
        'Fallow_rough_plow',          # Class 4
        'Fallow_smooth',              # Class 5
        'Stubble',                    # Class 6
        'Celery',                     # Class 7
        'Grapes_untrained',           # Class 8
        'Soil_vinyard_develop',       # Class 9
        'Corn_senesced_green_weeds',  # Class 10
        'Lettuce_romaine_4wk',        # Class 11
        'Lettuce_romaine_5wk',        # Class 12
        'Lettuce_romaine_6wk',        # Class 13
        'Lettuce_romaine_7wk',        # Class 14
        'Vinyard_untrained',          # Class 15
        'Vinyard_vertical_trellis'    # Class 16
    ]
    
    # =========================================================================
    # STEP 2: LOAD SALINAS DATASET FROM CSV
    # =========================================================================
    
    print("\n" + "="*70)
    print("LOADING SALINAS DATASET")
    print("="*70)
    
    # Load dataset from CSV file
    # CSV format: Each row is a pixel with [band_1, band_2, ..., band_204, label]
    # Output dimensions:
    #   sal_spectral: (512, 217, 204) - spectral image
    #   sal_label_img: (512, 217) - ground truth labels (0 = unlabeled, 1-16 = classes)
    sal_spectral, sal_label_img = load_dataset_from_csv(
        "/kaggle/input/hsi-net/Salinas_Dataset.csv",
        height=512,  # Number of rows in the image
        width=217    # Number of columns in the image
    )
    
    print(f"Dataset loaded:")
    print(f"  Spectral image shape: {sal_spectral.shape}")  # (512, 217, 204)
    print(f"  Label image shape: {sal_label_img.shape}")    # (512, 217)
    print(f"  Total pixels: {sal_spectral.shape[0] * sal_spectral.shape[1]:,}")  # 111,104
    print(f"  Spectral bands: {sal_spectral.shape[-1]}")  # 204
    
    # =========================================================================
    # STEP 3: APPLY PCA DIMENSIONALITY REDUCTION
    # =========================================================================
    
    print("\n" + "="*70)
    print("APPLYING PCA DIMENSIONALITY REDUCTION")
    print("="*70)
    
    # Initialize PCA with desired number of components
    # PCA reduces computational cost while retaining most spectral information
    pca_sal = PCA(n_components=n_components)
    
    # Reshape for PCA: (512, 217, 204) → (111104, 204)
    # PCA operates on 2D arrays where rows are samples
    sal_reshaped = sal_spectral.reshape(-1, sal_spectral.shape[-1])
    print(f"Reshaped for PCA: {sal_reshaped.shape}")  # (111104, 204)
    
    # Fit PCA and transform data: (111104, 204) → (111104, 30)
    # This learns the principal components and projects the data
    sal_transformed = pca_sal.fit_transform(sal_reshaped)
    print(f"After PCA transformation: {sal_transformed.shape}")  # (111104, 30)
    
    # Reshape back to image format: (111104, 30) → (512, 217, 30)
    sal_reduced = sal_transformed.reshape(
        sal_spectral.shape[0],   # height: 512
        sal_spectral.shape[1],   # width: 217
        n_components             # bands: 30
    )
    print(f"Final reduced image shape: {sal_reduced.shape}")  # (512, 217, 30)
    
    # Print variance explained by PCA
    total_variance = pca_sal.explained_variance_ratio_.sum()
    print(f"Total variance explained by {n_components} components: {total_variance:.2%}")
    
    # =========================================================================
    # STEP 4: EXTRACT LABELED SAMPLES
    # =========================================================================
    
    print("\n" + "="*70)
    print("EXTRACTING LABELED SAMPLES")
    print("="*70)
    
    # Extract only labeled pixels (where label > 0)
    # Returns DataFrame with columns: ['row', 'col', 'label']
    # Dimensions: (N_labeled_samples, 3) where N ≈ 54,129 for Salinas
    sal_samples = get_labeled_samples(sal_label_img)
    
    print(f"Total labeled samples: {len(sal_samples):,}")  # ~54,129
    
    # Show class distribution
    class_distribution = sal_samples['label'].value_counts().sort_index()
    print(f"\nClass distribution:")
    for idx, count in enumerate(class_distribution):
        print(f"  Class {idx+1:2d} ({salinas_class_names[idx]:30s}): {count:5d} samples")
    
    # =========================================================================
    # STEP 5: CREATE TRAIN/VALIDATION/TEST SPLITS
    # =========================================================================
    
    print("\n" + "="*70)
    print("CREATING DATA SPLITS")
    print("="*70)
    
    # First split: 80% train+val, 20% test
    # Stratified to maintain class proportions in each split
    # Input: sal_samples (54129, 3)
    # Output: train_val (~43303, 3), test_sal (~10826, 3)
    train_val, test_sal = train_test_split(
        sal_samples,
        test_size=0.2,                    # 20% for testing
        stratify=sal_samples['label'],    # Maintain class distribution
        random_state=42                   # For reproducibility
    )
    
    # Second split: 90% train, 10% val (from train+val)
    # This results in: 72% train, 8% val, 20% test of total data
    # Input: train_val (~43303, 3)
    # Output: train_sal (~38973, 3), val_sal (~4330, 3)
    train_sal, val_sal = train_test_split(
        train_val,
        test_size=0.1/0.8,                # 10% of train+val = 8% of total
        stratify=train_val['label'],      # Maintain class distribution
        random_state=42
    )
    
    # Print split statistics
    print(f"Data split summary:")
    print(f"  Training:   {len(train_sal):6d} samples ({len(train_sal)/len(sal_samples)*100:.1f}%)")
    print(f"  Validation: {len(val_sal):6d} samples ({len(val_sal)/len(sal_samples)*100:.1f}%)")
    print(f"  Test:       {len(test_sal):6d} samples ({len(test_sal)/len(sal_samples)*100:.1f}%)")
    print(f"  Total:      {len(sal_samples):6d} samples (100.0%)")
    
    # =========================================================================
    # STEP 6: CREATE DATA LOADERS
    # =========================================================================
    
    print("\n" + "="*70)
    print("CREATING DATA LOADERS")
    print("="*70)
    
    # Training data loader with augmentation
    # HyperspectralDataset extracts 11×11 patches around each labeled pixel
    # Augmentation includes: Gaussian noise, rotation, flipping
    # Input per sample: sal_reduced (512, 217, 30) + coordinates
    # Output per sample: patch (30, 11, 11), label (scalar)
    # Batch output: patches (16, 30, 11, 11), labels (16,)
    train_loader_sal = DataLoader(
        HyperspectralDataset(sal_reduced, train_sal, patch_size, augment=True),
        batch_size=16,    # Process 16 samples per batch
        shuffle=True      # Shuffle for better training
    )
    
    # Validation data loader (no augmentation)
    # Batch output: patches (32, 30, 11, 11), labels (32,)
    val_loader_sal = DataLoader(
        HyperspectralDataset(sal_reduced, val_sal, patch_size),
        batch_size=32,    # Larger batch size for evaluation (no gradients)
        shuffle=False     # No need to shuffle for validation
    )
    
    # Test data loader (no augmentation)
    # Batch output: patches (32, 30, 11, 11), labels (32,)
    test_loader_sal = DataLoader(
        HyperspectralDataset(sal_reduced, test_sal, patch_size),
        batch_size=32,
        shuffle=False
    )
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader_sal)} (batch_size=16)")
    print(f"  Val batches:   {len(val_loader_sal)} (batch_size=32)")
    print(f"  Test batches:  {len(test_loader_sal)} (batch_size=32)")
    
    # =========================================================================
    # STEP 7: INITIALIZE MODEL
    # =========================================================================
    
    print("\n" + "="*70)
    print("INITIALIZING CLARESNET MODEL")
    print("="*70)
    
    # Create CLAReSNet model for 16-class classification
    # Input: (B, 30, 11, 11) - batch of spectral patches
    # Output: (B, 16) - class logits
    model = CLAReSNet(n_classes=16, config=config).to(device)
    
    # Setup optimizer
    # AdamW: Adam with decoupled weight decay regularization
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,           # Learning rate
        weight_decay=1e-2  # L2 regularization strength
    )
    
    # Setup loss function
    # CrossEntropyLoss: Combines log_softmax and NLLLoss
    # Input: (B, 16) logits, (B,) labels
    # Output: scalar loss value
    criterion = nn.CrossEntropyLoss()
    
    # Calculate total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Print model architecture summary
    print(f"\nModel architecture:")
    print(f"  Spatial extractor: CNN with {config.base_c} base channels")
    print(f"  Spectral encoders: {config.n_encoder_layers} layers")
    print(f"  Embedding dim: {config.emb_dim}")
    print(f"  Dropout: {config.dropout}")
    
    # =========================================================================
    # STEP 8: TRAINING LOOP
    # =========================================================================
    
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)
    print(f"Training for 40 epochs...\n")
    
    # Initialize metrics tracker to store loss and accuracy per epoch
    sal_tracker = MetricsTracker()
    
    # Track best validation accuracy for model checkpointing
    best_acc = 0
    
    # Training loop
    for epoch in range(40):
        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        # Forward pass through all training batches
        # For each batch:
        #   Input: (16, 30, 11, 11)
        #   Model output: (16, 16)
        #   Loss: scalar
        #   Backprop and weight update
        # Returns: average loss and accuracy for the epoch
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader_sal,
            optimizer,
            criterion,
            device,
            desc=f"Pretrain Epoch {epoch+1}"
        )
        
        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        # Evaluate on validation set without gradient computation
        # For each batch:
        #   Input: (32, 30, 11, 11)
        #   Model output: (32, 16)
        #   Predictions: argmax over classes
        # Returns: accuracy, labels, predictions, loss
        val_acc, _, _, val_loss = evaluate(
            model,
            val_loader_sal,
            criterion,
            device,
            desc="Val"
        )
        
        # =====================================================================
        # METRICS TRACKING
        # =====================================================================
        # Store metrics for this epoch
        sal_tracker.update(epoch+1, train_loss, train_acc, val_loss, val_acc)
        
        # Print epoch summary
        print(
            f"Epoch {epoch + 1:02d}: "
            f"Loss = {train_loss:.4f}, Train Acc = {train_acc:6.2f}%, "
            f"Val Loss = {val_loss:.4f}, Val Acc = {val_acc:6.2f}%"
        )
        
        # =====================================================================
        # MODEL CHECKPOINTING
        # =====================================================================
        # Save model if validation accuracy improved
        if val_acc > best_acc:
            # Save model state dictionary (weights and biases)
            torch.save(model.state_dict(), PRETRAINED_PATH)
            best_acc = val_acc
            print(f"  → Best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # =========================================================================
    # STEP 9: PLOT TRAINING CURVES
    # =========================================================================
    
    print("\n" + "="*70)
    print("VISUALIZING TRAINING PROGRESS")
    print("="*70)
    
    # Plot training and validation curves
    # Creates 2 subplots: Loss vs Epoch, Accuracy vs Epoch
    # Each showing both training and validation curves
    sal_tracker.plot_training_curves(title="Salinas Pretraining")
    
    # =========================================================================
    # STEP 10: COMPREHENSIVE EVALUATION ON TEST SET
    # =========================================================================
    
    print("\n" + "="*70)
    print("COMPREHENSIVE EVALUATION")
    print("="*70)
    
    # Perform detailed evaluation on test set
    # Includes:
    #   - Confusion matrix: (16, 16) normalized by row
    #   - Per-class metrics: precision, recall, F1 for each class
    #   - t-SNE visualization: (N_test, 256) features → (N_test, 2)
    #   - PCA visualization: (N_test, 256) features → (N_test, 2)
    #   - Classification map: (512, 217) predicted labels
    #   - Uncertainty map: (512, 217) entropy values
    #   - ROC curves: One curve per class
    #   - Precision-Recall curves: One curve per class
    #
    # For each test batch:
    #   Input: (32, 30, 11, 11)
    #   Features: (32, 256) - from return_feat=True
    #   Logits: (32, 16)
    #   Probabilities: (32, 16) - after softmax
    #   Predictions: (32,) - argmax of logits
    sal_metrics = comprehensive_evaluation(
        model,
        test_loader_sal,
        device,
        "Salinas",
        salinas_class_names,
        spectral_image=sal_reduced,   # (512, 217, 30)
        label_image=sal_label_img,    # (512, 217)
        samples_df=test_sal,          # (~10826, 3)
        patch_size=patch_size,        # 11
        n_components=n_components     # 30
    )
    
    # =========================================================================
    # STEP 11: FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    
    print(f"\nFinal Results:")
    print(f"  Best Validation Accuracy: {best_acc:.2f}%")
    print(f"  Test Accuracy: {sal_metrics['accuracy']*100:.2f}%")
    print(f"  Test Balanced Accuracy: {sal_metrics['balanced_accuracy']*100:.2f}%")
    print(f"  Cohen's Kappa: {sal_metrics['cohen_kappa']:.4f}")
    print(f"  Matthews Correlation: {sal_metrics['matthews_corrcoef']:.4f}")
    print(f"  Macro F1-Score: {sal_metrics['f1_macro']:.4f}")
    print(f"  Weighted F1-Score: {sal_metrics['f1_weighted']:.4f}")
    
    print(f"\nModel saved to: {PRETRAINED_PATH}")
    print(f"Visualizations saved to current directory")
    
    print("\n" + "="*70)
    print("ALL OPERATIONS COMPLETED SUCCESSFULLY")
    print("="*70)


"""
=============================================================================
DIMENSION TRACKING SUMMARY FOR SALINAS DATASET
=============================================================================

Data Loading and Preprocessing:
--------------------------------
1. CSV file: (111104, 205)
   - 111104 pixels (512 × 217)
   - 204 spectral bands + 1 label column

2. Reshape to image: (512, 217, 204)
   - Height: 512 pixels
   - Width: 217 pixels
   - Spectral bands: 204

3. PCA transformation:
   Input: (111104, 204) - flattened image
   Output: (111104, 30) - reduced to 30 components
   Reshape: (512, 217, 30)

4. Labeled samples: (~54129, 3)
   - Columns: [row, col, label]
   - Only pixels with label > 0

5. Data splits:
   - Train: (~38973, 3) - 72% of total
   - Val: (~4330, 3) - 8% of total
   - Test: (~10826, 3) - 20% of total

Data Loading (Batches):
-----------------------
Training batch:
   - Input: (16, 30, 11, 11)
     * 16 samples per batch
     * 30 spectral bands (PCA components)
     * 11×11 spatial patch
   - Labels: (16,) - class indices (0-15)

Validation/Test batch:
   - Input: (32, 30, 11, 11)
   - Labels: (32,)

Model Forward Pass (see main documentation for detailed breakdown):
-------------------------------------------------------------------
Input: (B, 30, 11, 11)
↓ Spatial Feature Extractor
↓ (B, 30, 256)
↓ 3× Spectral Encoder Layers
↓ (B, 30, 256)
↓ Cross-Layer Fusion
↓ (B, 256)
↓ Classification Head
Output: (B, 16) - logits for 16 classes

Model Forward Pass Dimensions:
-------------------------------

Input: x = (B, T, H, W) where T=C_reduced=30, H=W=11
Example: (16, 30, 11, 11)

1. Spatial Feature Extractor:
   Input: (B, T, H, W) = (16, 30, 11, 11)
   → Reshape: (B*T, 1, H, W) = (480, 1, 11, 11)
   → CNN stem: (480, 1, 11, 11) → (480, 64, 11, 11)
   → Residual blocks: (480, 64, 11, 11) → (480, 64, 11, 11)
   → CBAM attention: (480, 64, 11, 11) → (480, 64, 11, 11)
   → Global pooling: (480, 64, 11, 11) → (480, 64) + (480, 64)
   → Concatenate: (480, 128)
   → Projection: (480, 128) → (480, 256)
   → Reshape: (B, T, emb_dim) = (16, 30, 256)
   Output: (16, 30, 256)

2. Spectral Encoder Layer (×3 layers):
   Input: (B, T, emb_dim) = (16, 30, 256)
   
   a) Bi-LSTM:
      Input: (16, 30, 256)
      Hidden per direction: 128
      Output: (16, 30, 256)
   
   b) Bi-GRU:
      Input: (16, 30, 256)
      Output: (16, 30, 256)
   
   c) Multi-Scale Latent Attention:
      Input: (16, 30, 256)
      
      - Positional encoding: (16, 30, 256) → (16, 30, 256)
      
      - Number of latent tokens: ~16-24 (adaptive based on T=30)
        Latent tokens: (16, num_latents, 256)
      
      For each scale [1, 2, 4]:
        Scale 1 (full resolution):
          - Input-to-latent: Q=(16,20,256), K/V=(16,30,256) → (16,20,256)
          - Latent self-attn: (16,20,256) → (16,20,256)
          - Latent FFN: (16,20,256) → (16,20,512) → (16,20,256)
          - Latent-to-output: Q=(16,30,256), K/V=(16,20,256) → (16,30,256)
        
        Scale 2 (half resolution):
          - Downsample input: (16,30,256) → (16,15,256)
          - Process similarly
          - Upsample: (16,15,256) → (16,30,256)
        
        Scale 4 (quarter resolution):
          - Downsample: (16,30,256) → (16,8,256)
          - Process similarly
          - Upsample: (16,8,256) → (16,30,256)
      
      - Concatenate scales: 3×(16,30,256) → (16,30,768)
      - Fusion network: (16,30,768) → (16,30,512) → (16,30,256)
      - Residual + Norm: (16,30,256)
      
      Output: (16, 30, 256)
   
   d) Feed-Forward Network:
      Input: (16, 30, 256)
      → (16, 30, 1024)
      → (16, 30, 256)
      Output: (16, 30, 256)
   
   Final layer output: (16, 30, 256)

3. Cross-Layer Attention Fusion:
   All layer outputs: [(16,30,256)] × 3 layers
   
   - Mean pooling each: [(16,256)] × 3
   - Stack: (16, 3, 256)
   - Query (last layer): (16, 1, 256)
   - Cross-attention: Q=(16,1,256), K/V=(16,3,256) → (16,1,256)
   - Squeeze + residual: (16, 256)
   
   Output: (16, 256)

4. Classification Head:
   Input: (16, 256)
   → LayerNorm: (16, 256)
   → Dropout: (16, 256)
   → Linear: (16, 256) → (16, 128)
   → GELU: (16, 128)
   → Dropout: (16, 128)
   → Linear: (16, 128) → (16, 16)
   
   Output: (16, 16) - logits for 16 classes

Output Formats:
---------------
1. Training/Inference (default):
   Output: (B, n_classes) - class logits
   Example: (16, 16)

2. Feature extraction (return_feat=True):
   Output: (B, emb_dim) - learned features
   Example: (16, 256)

3. Probability prediction (return_prob=True):
   Output: (B, n_classes) - class probabilities (after softmax)
   Example: (16, 16) with values summing to 1

4. With attention weights (return_att=True):
   Output: (logits, attention_dicts)
   - logits: (B, n_classes)
   - attention_dicts: List of dicts containing attention maps

Evaluation Outputs:
-------------------
- Confusion Matrix: (16, 16)
- Per-class metrics: (16,) for each metric
- t-SNE features: (N_test, 2) where N_test ≈ 10826
- PCA features: (N_test, 2)
- Classification map: (512, 217)
- Uncertainty map: (512, 217)
- ROC curves: 16 curves (one per class)
- Precision-Recall curves: 16 curves

Computational Complexity:
--------------------------
- Spatial CNN: O(B × T × H² × W² × C²)
- Multi-head attention: O(B × T² × emb_dim)
- Latent attention: O(B × T × num_latents × emb_dim) - much more efficient!
- Overall: ~O(B × T × emb_dim²) per encoder layer

The latent attention mechanism reduces complexity from O(T²) to O(T × num_latents),
where num_latents << T, making it highly efficient for long sequences.

Memory Requirements (approximate):
----------------------------------
- Training batch (fp32): ~1.5 MB
- Model parameters: ~8-10 MB
- Optimizer state: ~16-20 MB (AdamW with momentum)
- Peak GPU memory: ~500 MB - 1 GB (depends on batch size)

Memory Considerations:
-----------------------
With batch_size=16, emb_dim=256, T=30:
- Spatial features: 16 × 30 × 256 = 122,880 values
- Per encoder layer: ~4× due to intermediate computations
- Multi-scale attention: Additional 2-3× due to multiple scales
- Peak memory: ~50-100MB per batch (fp32) for activations

Total model parameters: ~2-5 million (depends on base_c and n_encoder_layers)
=============================================================================
"""
