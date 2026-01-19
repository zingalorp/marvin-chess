"""
Chessformer - Human-like chess transformer model.

Architecture features:
- Per-token input encoding (board history + metadata per square)
- Absolute position embeddings (per-square learned embeddings)
- Policy head with 1/sqrt(d_model) scaling
- Legal move masking
- Resign/flag as policy outputs (indices 4096, 4097)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

# Special policy indices
RESIGN_MOVE_INDEX = 4096
FLAG_MOVE_INDEX = 4097
NUM_POLICY_OUTPUTS = 4098

# Classification head dimensions
NUM_TIME_BINS = 256  # Time prediction as 256-class classification
NUM_VALUE_CLASSES = 3  # WDL: Win/Draw/Loss
NUM_SQUARES = 64


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x)). Better gradient flow than ReLU/SiLU."""
    def forward(self, x):
        return F.mish(x)


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================
# Both configs use token-based conditioning (6 tokens prepended to 64 square tokens)
# for ELO, time control, urgency, increment, and move timing.

# ~100M trainable parameters
CONFIG_LARGE = {
    "d_model": 704,           # Width increased (fits 22 heads)
    "n_layers": 20,           # Keep Depth 20 for calculation
    "n_heads": 22,            # 704 / 32 = 22 Heads
    "d_head": 32,             # Revert to 32-dim
    "d_ff": 1408,             # 2.0x expansion. (Compromise between 1.0x Small and 2.7x Std)
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,
    "num_piece_types": 13,
    "num_tc_cats": 3,
    "embedding_ffn": True,
    "smolgen": True,
    "smolgen_hidden": 256,    
    "smolgen_per_head": 128,  # Reduced slightly per head since we have 22 heads now
}

# Small config (~23M params)
# Uses 6 conditioning tokens prepended to 64 square tokens = 70 total tokens
# Trained weights: inference/marvin_token_bf16.pt, checkpoints/chessformer_token-new_best.pt
CONFIG_SMALL = {
    "d_model": 448,           # Base width
    "n_layers": 12,           # 12 transformer layers
    "n_heads": 14,            # 448 / 32 = 14
    "d_head": 32,
    "d_ff": 448,              # 1.0x d_model
    "dropout": 0.1,
    "max_rel_dist": 7,
    "history_len": 8,         # Used for board history
    "num_piece_types": 13,
    "num_tc_cats": 3,         # Blitz, Rapid, Classical
    "embedding_ffn": True,
    "smolgen": True,
    "smolgen_hidden": 256,
    "smolgen_per_head": 256,
}

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.scale


# ============================================================================
# Token-Based Conditioning
# ============================================================================

# Bins for log-scaled time: 16 bins covering 0-1800s (30 min)
# Boundaries at approximately: 0, 1, 2, 4, 7, 12, 20, 35, 60, 100, 180, 300, 500, 900, 1800
NUM_TIME_LOG_BINS = 16
NUM_ELO_ANCHORS = 14  # 1200-2500 in 100 ELO increments
NUM_TC_CATEGORIES = 3  # Blitz, Rapid, Classical
NUM_INC_CATEGORIES = 5  # 0, 1, 2, 5, 10+
NUM_CONDITIONING_TOKENS = 6  # ELO, TC, URGENCY, INC, MY_TIME, OPP_TIME


def log_bin_time(seconds: torch.Tensor, num_bins: int = NUM_TIME_LOG_BINS) -> torch.Tensor:
    """
    Bin time values logarithmically. Returns bin indices in [0, num_bins-1].
    
    Covers 0 to ~1800 seconds (30 min) with denser bins at lower times.
    Uses log1p for smooth handling of zero.
    
    ONNX-compatible: uses torch.clamp with min/max tensors to avoid type mismatch.
    """
    # log1p(1800) ≈ 7.5, so divide by 7.5 and scale to bins
    log_time = torch.log1p(torch.clamp(seconds, min=0.0))
    # Normalize to [0, 1] range, then to bin indices
    normalized = log_time / 7.5  # Max expected log1p value
    # Use explicit min/max for ONNX compatibility
    bins = (torch.clamp(normalized, min=0.0, max=0.9999) * num_bins).long()
    return bins


def bin_increment(inc: torch.Tensor) -> torch.Tensor:
    """
    Bin increment values into 5 categories: 0, 1, 2, 5, 10+.
    Returns indices 0-4.
    
    ONNX-compatible: uses arithmetic instead of torch.tensor() in traced context.
    """
    # Categories: 0->0, 1->1, 2->2, 3-9->3, 10+->4
    # Use arithmetic masking for ONNX compatibility
    bins = torch.zeros_like(inc)
    bins = bins + (inc == 1).to(inc.dtype)  # 1 -> 1
    bins = bins + 2 * (inc == 2).to(inc.dtype)  # 2 -> 2
    bins = bins + 3 * ((inc >= 3) & (inc < 10)).to(inc.dtype)  # 3-9 -> 3
    bins = bins + 4 * (inc >= 10).to(inc.dtype)  # 10+ -> 4
    return bins.long()


class TokenConditioningEncoder(nn.Module):
    """
    Encode global context as explicit tokens prepended to the square sequence.
    
    Tokens:
    1. [ELO_TOKEN] - Interpolated embedding for player skill level
    2. [TC_TOKEN] - Time control category (Blitz, Rapid, Classical)
    3. [URGENCY_TOKEN] - Log-binned remaining time
    4. [INC_TOKEN] - Increment category (0, 1, 2, 5, 10+)
    5. [MY_LAST_TIME] - Log-binned time spent on my last move
    6. [OPP_LAST_TIME] - Log-binned time spent on opponent's last move
    
    This approach lets the model attend to conditioning information directly 
    through self-attention.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # ELO: Interpolated embedding with anchor points
        # Anchors at 1200-2500 in 100 ELO increments (matches dataset range)
        self.elo_anchors = nn.Parameter(
            torch.tensor([1200.0, 1300.0, 1400.0, 1500.0, 1600.0, 1700.0, 1800.0,
                          1900.0, 2000.0, 2100.0, 2200.0, 2300.0, 2400.0, 2500.0]),
            requires_grad=False
        )
        self.elo_embeddings = nn.Embedding(NUM_ELO_ANCHORS, d_model)
        
        # TC: Categorical embedding (Blitz=0, Rapid=1, Classical=2)
        self.tc_embedding = nn.Embedding(NUM_TC_CATEGORIES, d_model)
        
        # Urgency: Log-binned remaining time
        self.urgency_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
        
        # Increment: Categorical (0, 1, 2, 5, 10+)
        self.inc_embedding = nn.Embedding(NUM_INC_CATEGORIES, d_model)
        
        # My last move time: Log-binned
        self.my_time_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
        
        # Opponent's last move time: Log-binned
        self.opp_time_embedding = nn.Embedding(NUM_TIME_LOG_BINS, d_model)
        
        # Position embeddings for the 6 conditioning tokens
        self.token_pos_embedding = nn.Embedding(NUM_CONDITIONING_TOKENS, d_model)
        
        # Pre-computed token position indices
        self.register_buffer(
            "token_pos_ids",
            torch.arange(NUM_CONDITIONING_TOKENS, dtype=torch.long),
            persistent=False
        )
    
    def _interpolate_elo(self, elo: torch.Tensor) -> torch.Tensor:
        """
        Interpolate ELO embedding from anchor points.
        
        Args:
            elo: (B,) raw ELO values
        Returns:
            (B, d_model) interpolated embeddings
        """
        B = elo.shape[0]
        device = elo.device
        
        # Clamp ELO to anchor range
        elo_clamped = elo.clamp(self.elo_anchors[0], self.elo_anchors[-1])
        
        # Find which segment each ELO falls into
        # For each ELO, find the largest anchor <= elo
        anchors = self.elo_anchors  # (NUM_ELO_ANCHORS,)
        
        # Broadcast: (B, 1) vs (NUM_ELO_ANCHORS,) -> (B, NUM_ELO_ANCHORS)
        below_mask = elo_clamped.unsqueeze(1) >= anchors.unsqueeze(0)
        
        # Index of lower anchor (last True in each row)
        lower_idx = below_mask.sum(dim=1) - 1  # (B,)
        lower_idx = lower_idx.clamp(0, NUM_ELO_ANCHORS - 2)  # Ensure valid upper idx
        upper_idx = lower_idx + 1
        
        # Get anchor values
        lower_anchor = anchors[lower_idx]  # (B,)
        upper_anchor = anchors[upper_idx]  # (B,)
        
        # Interpolation weight
        t = (elo_clamped - lower_anchor) / (upper_anchor - lower_anchor + 1e-6)  # (B,)
        t = t.clamp(0, 1).unsqueeze(1)  # (B, 1)
        
        # Get embeddings and interpolate
        lower_emb = self.elo_embeddings(lower_idx)  # (B, d_model)
        upper_emb = self.elo_embeddings(upper_idx)  # (B, d_model)
        
        return (1 - t) * lower_emb + t * upper_emb  # (B, d_model)
    
    def forward(
        self,
        player_elo: torch.Tensor,      # (B,) raw ELO
        tc_cat: torch.Tensor,           # (B,) 0=Blitz, 1=Rapid, 2=Classical
        remaining_time: torch.Tensor,   # (B,) seconds remaining
        increment: torch.Tensor,        # (B,) increment in seconds
        my_last_time: torch.Tensor,     # (B,) seconds spent on my last move
        opp_last_time: torch.Tensor,    # (B,) seconds spent on opponent's last move
    ) -> torch.Tensor:
        """
        Generate conditioning token embeddings.
        
        Returns:
            (B, 6, d_model) - 6 conditioning tokens
        """
        B = player_elo.shape[0]
        
        # 1. ELO token (interpolated)
        elo_token = self._interpolate_elo(player_elo)  # (B, d_model)
        
        # 2. TC token (categorical)
        tc_token = self.tc_embedding(tc_cat)  # (B, d_model)
        
        # 3. Urgency token (log-binned remaining time)
        urgency_bins = log_bin_time(remaining_time)
        urgency_token = self.urgency_embedding(urgency_bins)  # (B, d_model)
        
        # 4. Increment token (categorical)
        inc_bins = bin_increment(increment)
        inc_token = self.inc_embedding(inc_bins)  # (B, d_model)
        
        # 5. My last move time token (log-binned)
        my_time_bins = log_bin_time(my_last_time)
        my_time_token = self.my_time_embedding(my_time_bins)  # (B, d_model)
        
        # 6. Opponent's last move time token (log-binned)
        opp_time_bins = log_bin_time(opp_last_time)
        opp_time_token = self.opp_time_embedding(opp_time_bins)  # (B, d_model)
        
        # Stack tokens: (B, 6, d_model)
        tokens = torch.stack([
            elo_token, tc_token, urgency_token,
            inc_token, my_time_token, opp_time_token
        ], dim=1)
        
        # Add position embeddings
        tokens = tokens + self.token_pos_embedding(self.token_pos_ids)  # broadcast over B
        
        return tokens


class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns which squares matter for value/time prediction.
    
    Unlike mean pooling which weights all 64 squares equally, this learns to
    focus on tactically relevant squares (e.g., the hanging queen, not empty squares).
    
    Optionally conditioned on global context (ELO might focus on different features).
    """
    def __init__(self, d_model, context_dim=None):
        super().__init__()
        self.d_model = d_model
        self.use_context = context_dim is not None
        
        if self.use_context:
            # Query is generated from context
            self.query_proj = nn.Linear(context_dim, d_model)
        else:
            # Learned query vector
            self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.scale = 1.0 / math.sqrt(d_model)
    
    def forward(self, x, context=None):
        """
        Args:
            x: (B, 64, d_model) - token representations
            context: (B, context_dim) - optional global context
        Returns:
            pooled: (B, d_model) - weighted combination of tokens
        """
        B = x.shape[0]
        keys = self.key_proj(x)  # (B, 64, d_model)
        
        if self.use_context and context is not None:
            query = self.query_proj(context)  # (B, d_model)
        else:
            query = self.query.unsqueeze(0).expand(B, -1)  # (B, d_model)
        
        # Compute attention scores
        attn_scores = (keys * query.unsqueeze(1)).sum(dim=-1) * self.scale  # (B, 64)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 64)
        
        # Weighted sum
        pooled = (x * attn_weights.unsqueeze(-1)).sum(dim=1)  # (B, d_model)
        return pooled


class PaperValueHead(nn.Module):
    """
    Paper implementation:
    1. Project tokens to d_val=32 (bottleneck)
    2. Flatten (preserve spatial structure)
    3. Project to value embedding
    """
    def __init__(self, d_model, d_val=32):
        super().__init__()
        self.bottleneck = nn.Linear(d_model, d_val, bias=False)
        self.flatten_dim = NUM_SQUARES * d_val
        
        # Value prediction (scalar)
        self.final_proj = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            Mish(),
            nn.Linear(128, 1)
        )
        
        # WDL Classification
        self.wdl_proj = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            Mish(),
            nn.Linear(128, NUM_VALUE_CLASSES)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 64, d_model)
        Returns:
            val: (B, 1) scalar value
            wdl: (B, 3) WDL logits
        """
        b = self.bottleneck(x)      # (B, 64, 32)
        flat = b.flatten(1)         # (B, 2048)
        val = self.final_proj(flat)
        wdl = self.wdl_proj(flat)
        return val, wdl


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit FFN."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class RelativePositionGenerator(nn.Module):
    """Generate relative position indices for chess board."""
    def __init__(self, max_dist=7):
        super().__init__()
        self.max_dist = max_dist
        self.num_buckets = (2 * max_dist + 1) ** 2
        
        # Precompute relative indices
        indices = torch.zeros(64, 64, dtype=torch.long)
        for i in range(64):
            for j in range(64):
                ri, ci = i // 8, i % 8
                rj, cj = j // 8, j % 8
                dr = max(-max_dist, min(max_dist, rj - ri))
                dc = max(-max_dist, min(max_dist, cj - ci))
                bucket = (dr + max_dist) * (2 * max_dist + 1) + (dc + max_dist)
                indices[i, j] = bucket
        self.register_buffer("indices", indices)

    def forward(self):
        return self.indices


class Smolgen(nn.Module):
    """
    Smolgen: Dynamic attention bias generation based on position content.
    
    From Leela Chess Zero:
    1. Compress the current representation into a small vector (e.g., 256d)
    2. For each head, generate supplemental attention logits (64x64)
    
    This allows the model to dynamically adjust which squares attend to which,
    based on the current position (open vs closed, piece placement, etc.)
    """
    def __init__(self, d_model, n_heads, hidden_size=256, per_head_size=256):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.per_head_size = per_head_size
        
        # Step 1: Compress position into a global vector
        # Project each token to a smaller dimension, then flatten and compress
        self.token_compress = nn.Linear(d_model, 32, bias=False)
        self.global_compress = nn.Sequential(
            nn.Linear(64 * 32, hidden_size),
            RMSNorm(hidden_size),
        )
        
        # Step 2: Generate per-head attention biases using a single batched projection
        # Project from hidden_size to (n_heads * per_head_size) in one operation
        # This replaces the old ModuleList approach for better torch.compile compatibility
        self.head_proj = nn.Linear(hidden_size, n_heads * per_head_size, bias=False)
        
        # Shared projection from per_head_size to 64x64 attention logits
        # This is shared across all heads (as per Leela's description)
        self.attn_logit_proj = nn.Linear(per_head_size, 64 * 64, bias=False)
        
        # Flag to track if we've converted from legacy format
        self._converted_from_legacy = False
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """Handle loading from old checkpoint format with head_projections ModuleList."""
        # Check if this is an old-format checkpoint (has head_projections.0.weight but no head_proj.weight)
        legacy_key_0 = f"{prefix}head_projections.0.weight"
        new_key = f"{prefix}head_proj.weight"
        
        if legacy_key_0 in state_dict and new_key not in state_dict:
            # Convert from legacy ModuleList format to new batched format
            # Old format: head_projections.{i}.weight with shape (per_head_size, hidden_size)
            # New format: head_proj.weight with shape (n_heads * per_head_size, hidden_size)
            legacy_weights = []
            for i in range(self.n_heads):
                key = f"{prefix}head_projections.{i}.weight"
                if key in state_dict:
                    legacy_weights.append(state_dict.pop(key))
                    # Add to unexpected_keys removal
                else:
                    # Missing head projection - this shouldn't happen in valid checkpoints
                    error_msgs.append(f"Missing legacy weight: {key}")
                    break
            
            if len(legacy_weights) == self.n_heads:
                # Stack all head weights: (n_heads, per_head_size, hidden_size)
                # Then reshape to (n_heads * per_head_size, hidden_size)
                stacked = torch.cat(legacy_weights, dim=0)  # (n_heads * per_head_size, hidden_size)
                state_dict[new_key] = stacked
                self._converted_from_legacy = True
        
        # Call parent implementation
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        
    def forward(self, x):
        """
        Args:
            x: (B, 64, d_model) - token representations
        Returns:
            attn_bias: (B, n_heads, 64, 64) - dynamic attention biases
        """
        B = x.shape[0]
        
        # Compress tokens: (B, 64, d_model) -> (B, 64, 32)
        compressed = self.token_compress(x)
        
        # Flatten and compress to global: (B, 64*32) -> (B, hidden_size)
        global_repr = self.global_compress(compressed.flatten(1))
        
        # Generate all head representations in one go: (B, hidden_size) -> (B, n_heads * per_head_size)
        all_head_repr = self.head_proj(global_repr)
        
        # Reshape to (B, n_heads, per_head_size)
        all_head_repr = all_head_repr.view(B, self.n_heads, self.per_head_size)
        
        # Apply shared projection to all heads: (B, n_heads, per_head_size) -> (B, n_heads, 64*64)
        # Using einsum for batched linear: weight is (per_head_size, 64*64)
        head_logits = torch.matmul(all_head_repr, self.attn_logit_proj.weight.T)
        
        # Reshape to (B, n_heads, 64, 64)
        return head_logits.view(B, self.n_heads, 64, 64)


class ShawAttention(nn.Module):
    """Self-attention with Shaw relative position encoding and optional smolgen."""
    def __init__(self, d_model, n_heads, d_head, num_rel_pos, dropout=0.1, bias=False):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_head
        self.scale = 1.0 / math.sqrt(d_head)
        
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.o_proj = nn.Linear(d_model, d_model, bias=bias)

        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)
        
        self.rel_k_emb = nn.Embedding(num_rel_pos, d_head)
        self.rel_v_emb = nn.Embedding(num_rel_pos, d_head)
        
        self.dropout = nn.Dropout(dropout)

        # Debug/inference: populated with attention probabilities from the most recent forward.
        # Shape: (B, n_heads, 64, 64)
        self.last_attn_probs = None

        # Debug/inference: populated with the smolgen dynamic bias logits from the most recent forward.
        # Shape: (B, n_heads, 64, 64)
        self.last_smolgen_bias = None

    def forward(self, x, rel_indices, smolgen_bias=None):
        B, L, D = x.shape
        H, d_k = self.n_heads, self.d_head
        
        q = self.q_proj(x).view(B, L, H, d_k)
        k = self.k_proj(x).view(B, L, H, d_k)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, d_k).transpose(1, 2)
        
        r_k = self.rel_k_emb(rel_indices)
        r_v = self.rel_v_emb(rel_indices)
        
        content_score = torch.matmul(q, k.transpose(-2, -1))
        rel_score = torch.einsum('bhid,ijd->bhij', q, r_k)
        
        attn_scores = (content_score + rel_score) * self.scale
        
        # Add smolgen dynamic attention bias if provided
        if smolgen_bias is not None:
            attn_scores = attn_scores + smolgen_bias
        
        attn_probs = self.dropout(F.softmax(attn_scores, dim=-1))

        # Stash for inference-time visualization only (not during training to avoid graph breaks)
        if not self.training:
            self.last_smolgen_bias = smolgen_bias.detach() if smolgen_bias is not None else None
            self.last_attn_probs = attn_probs.detach()
        
        content_out = torch.matmul(attn_probs, v)
        rel_out = torch.einsum('bhij,ijd->bhid', attn_probs, r_v)
        
        output = (content_out + rel_out).transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    """Transformer block with Mish activation and bias=False for attention."""
    def __init__(self, config, num_rel_pos):
        super().__init__()
        d_model = config['d_model']
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Bias=False for QKV (Paper)
        self.attn = ShawAttention(
            d_model, config['n_heads'], config['d_head'],
            num_rel_pos, config['dropout'], bias=False
        )
        
        # SwiGLU FFN for improved convergence
        self.ff = SwiGLU(d_model, config['d_ff'], config['dropout'])

    def forward(self, x, rel_indices, smolgen_bias=None):
        # Pre-LN structure
        x = x + self.attn(self.norm1(x), rel_indices=rel_indices, smolgen_bias=smolgen_bias)
        x = x + self.ff(self.norm2(x))
        return x


class PerSquareInputEncoder(nn.Module):
    """
    Encode per-square input features following the paper's approach.
    
    Per-token input = [
        8 × one-hot(piece_code, 13)  # 8 board states × 13 types = 104 dims
        + 1 × en_passant_flag        # 1 if this is the EP target square
        + 8 × time_history           # 8 floats for move times
        + 8 × repetition_flags       # 8 binary flags
        + 4 × castling_rights        # 4 binary flags
        + 8 × scalars                # normalized scalar features
    ] = 104 + 1 + 8 + 8 + 4 + 8 = 133 dims per token
    """
    def __init__(self, config):
        super().__init__()
        self.history_len = config['history_len']
        self.num_piece_types = config['num_piece_types']
        
        # Input dimension per token
        # 8 * 13 (piece one-hots) + 1 (ep) + 8 (time) + 8 (rep) + 4 (castle) + 8 (scalars)
        self.input_dim = (self.history_len * self.num_piece_types + 1 + 
                 self.history_len + self.history_len + 4 + 8)

        # Separate board features from metadata so metadata starts near-silent
        self.board_dim = self.history_len * self.num_piece_types + 1
        self.meta_dim = self.input_dim - self.board_dim

        self.board_proj = nn.Linear(self.board_dim, config['d_model'], bias=False)
        self.meta_proj = nn.Linear(self.meta_dim, config['d_model'], bias=False)
        
        # Per-square absolute position embedding
        self.square_emb = nn.Embedding(64, config['d_model'])

        # Avoid per-forward torch.arange allocations (exclude from state_dict)
        self.register_buffer("square_ids", torch.arange(64, dtype=torch.long), persistent=False)
        
        # Per-square scale and bias (paper's "multiply by learned offset")
        self.square_scale = nn.Parameter(torch.ones(64, config['d_model']))
        self.square_bias = nn.Parameter(torch.zeros(64, config['d_model']))
        
        # TC category embedding (added to all squares)
        self.tc_emb = nn.Embedding(config['num_tc_cats'], config['d_model'])
        
        # Leela-style: optional FFN after embedding
        # "We also add an FFN layer after the embedding so the model can make use of this information"
        self.use_embedding_ffn = config.get('embedding_ffn', False)
        if self.use_embedding_ffn:
            d_model = config['d_model']
            # Use 4x expansion for the embedding FFN (this is where capacity goes)
            self.emb_ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model),
            )
            self.emb_norm = RMSNorm(d_model)

        # Initialize projections: full-strength board, near-silent metadata
        nn.init.normal_(self.board_proj.weight, std=0.02)
        nn.init.normal_(self.meta_proj.weight, std=0.001)
        # Mark to skip global re-init
        self.board_proj._no_reinit = True
        self.meta_proj._no_reinit = True

    def forward(self, board_history, time_history, rep_flags, castling, 
                ep_mask, scalars, tc_cat):
        """
        Args:
            board_history: (B, 8, 64) int64 - piece codes per square per history step
            time_history: (B, 8) float - normalized time per historical move
            rep_flags: (B, 8) float - repetition flags per history step
            castling: (B, 4) float - castling rights
            ep_mask: (B, 64) float - 1.0 at EP square, else 0
            scalars: (B, 8) float - normalized scalar features
            tc_cat: (B,) long - time control category
        
        Returns:
            (B, 64, d_model) token embeddings
        """
        B = board_history.shape[0]
        device = board_history.device
        
        # 1. One-hot encode board history: (B, 8, 64, 13)
        board_onehot = F.one_hot(board_history, num_classes=self.num_piece_types).float()
        # Reshape to (B, 64, 8*13) - flatten history dimension per square
        board_onehot = board_onehot.permute(0, 2, 1, 3).reshape(B, 64, -1)
        
        # 2. Broadcast time_history to all squares: (B, 64, 8)
        time_hist_broadcast = time_history.unsqueeze(1).expand(-1, 64, -1)
        
        # 3. Broadcast rep_flags to all squares: (B, 64, 8)
        rep_broadcast = rep_flags.unsqueeze(1).expand(-1, 64, -1)
        
        # 4. Broadcast castling to all squares: (B, 64, 4)
        castle_broadcast = castling.unsqueeze(1).expand(-1, 64, -1)
        
        # 5. EP mask is already per-square: (B, 64, 1)
        ep_per_square = ep_mask.unsqueeze(-1)
        
        # 6. Broadcast scalars to all squares: (B, 64, 8)
        scalars_broadcast = scalars.unsqueeze(1).expand(-1, 64, -1)
        
        # Split board vs metadata pathways
        board_input = torch.cat([
            board_onehot,   # (B, 64, 104)
            ep_per_square,  # (B, 64, 1)
        ], dim=-1)

        meta_input = torch.cat([
            time_hist_broadcast, # (B, 64, 8)
            rep_broadcast,       # (B, 64, 8)
            castle_broadcast,    # (B, 64, 4)
            scalars_broadcast,   # (B, 64, 8)
        ], dim=-1)

        # Project to d_model with gated metadata
        x = self.board_proj(board_input) + self.meta_proj(meta_input)
        
        # Add absolute position embedding
        x = x + self.square_emb(self.square_ids)
        
        # Apply per-square scale and bias
        x = x * self.square_scale + self.square_bias
        
        # Add TC category embedding (broadcast to all squares)
        tc_emb = self.tc_emb(tc_cat)  # (B, d_model)
        x = x + tc_emb.unsqueeze(1)
        
        # Leela-style: FFN after embedding to let model make use of full board info
        if self.use_embedding_ffn:
            x = x + self.emb_ffn(self.emb_norm(x))
        
        return x


class Chessformer(nn.Module):
    """
    Chessformer model with token-based conditioning.
    
    Uses 6 conditioning tokens prepended to 64 square tokens for:
    - ELO (interpolated embedding)
    - Time control category (Blitz/Rapid/Classical)
    - Urgency (remaining time)
    - Increment category
    - My last move time
    - Opponent's last move time
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        
        # Token-based conditioning (6 tokens prepended to 64 square tokens = 70 total)
        self.token_conditioning = TokenConditioningEncoder(d_model)
        
        # Gradient checkpointing for memory-efficient training
        self.gradient_checkpointing = config.get('gradient_checkpointing', False)
        
        # Input encoder (per-square features only, globals handled by token conditioning)
        self.input_encoder = PerSquareInputEncoder(config)
        
        # Transformer body - using TransformerBlock with Mish activation
        self.rel_pos_gen = RelativePositionGenerator(config['max_rel_dist'])
        
        # Extended relative position for 70 tokens (6 conditioning + 64 squares)
        # Conditioning tokens use the max distance bucket for all positions
        ext_indices = torch.zeros(70, 70, dtype=torch.long)
        # Fill in square-to-square relative positions (tokens 6-69)
        ext_indices[NUM_CONDITIONING_TOKENS:, NUM_CONDITIONING_TOKENS:] = self.rel_pos_gen.indices
        # All other positions (involving conditioning tokens) use max bucket
        max_bucket = self.rel_pos_gen.num_buckets - 1
        ext_indices[:NUM_CONDITIONING_TOKENS, :] = max_bucket
        ext_indices[:, :NUM_CONDITIONING_TOKENS] = max_bucket
        self.register_buffer("extended_rel_indices", ext_indices)
        
        self.layers = nn.ModuleList([
            TransformerBlock(config, self.rel_pos_gen.num_buckets)
            for _ in range(config['n_layers'])
        ])
        self.norm_f = RMSNorm(d_model)
        
        # Smolgen: dynamic attention biases (shared across all layers)
        self.use_smolgen = config.get('smolgen', False)
        if self.use_smolgen:
            self.smolgen = Smolgen(
                d_model=d_model,
                n_heads=config['n_heads'],
                hidden_size=config.get('smolgen_hidden', 256),
                per_head_size=config.get('smolgen_per_head', 256),
            )
        
        # --- Output Heads ---
        
        # Policy head: produces (B, 4098) logits
        # 4096 for regular moves + resign + flag
        self.policy_query = nn.Linear(d_model, d_model, bias=False)
        self.policy_key = nn.Linear(d_model, d_model, bias=False)
        self.policy_scale = 1.0 / math.sqrt(d_model)  # Scaling per paper
        
        # Special action tokens (resign, flag) - learned embeddings
        self.resign_key = nn.Parameter(torch.randn(d_model) * 0.02)
        self.flag_key = nn.Parameter(torch.randn(d_model) * 0.02)
        
        # Promotion head: per destination square
        # Paper's approach: additive bias from key vectors at promotion rank
        self.promo_bias_proj = nn.Linear(d_model, 4)  # Q, R, B, N promotions
        
        # Aux: Start Square Head (Human Mimicry Booster)
        # Predicts which of the 64 squares the piece moves FROM
        self.start_square_head = nn.Linear(d_model, 1)
        
        # Value head (Paper Style - flattened spatial + MLP)
        self.value_head = PaperValueHead(d_model)

        # Value error head to model volatility
        self.value_error_pool = AttentionPooling(d_model, context_dim=None)
        self.value_error_head = nn.Sequential(
            nn.Linear(d_model, 256),
            Mish(),
            nn.Linear(256, 1),
        )
        
        # Time head: 256-bin classification with attention pooling
        # Time distribution is multimodal (instant moves, short thinks, long thinks)
        self.time_pooling = AttentionPooling(d_model, context_dim=None)
        
        self.time_head = nn.Sequential(
            nn.Linear(d_model, 256, bias=False),
            Mish(),
            nn.Linear(256, NUM_TIME_BINS),  # logits for 256 time bins
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if getattr(module, "_no_reinit", False):
            return
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, batch, return_promo=False):
        """
        Args:
            batch: dict with keys from dataset
            return_promo: if True, also return promotion logits
        
        Returns:
            move_logits: (B, 4098) policy logits (masked if legal_mask provided)
            value_out: (B, 1) win probability
            value_error_out: (B, 1) predicted squared error / volatility proxy
            time_out: (B, 1) predicted time in normalized units
            promo_logits: (B, 8, 4) promotion logits for dest squares 56-63 (optional)
        """
        # Extract inputs from batch
        board_history = batch['board_history']
        time_history = batch['time_history']
        rep_flags = batch['rep_flags']
        castling = batch['castling']
        ep_mask = batch['ep_mask']
        scalars = batch['scalars']
        tc_cat = batch['tc_cat']
        legal_mask = batch.get('legal_mask', None)
        
        B = board_history.shape[0]
        device = board_history.device
        
        # Encode square inputs
        x = self.input_encoder(
            board_history, time_history, rep_flags, castling,
            ep_mask, scalars, tc_cat
        )  # (B, 64, d_model)
        
        # Token conditioning: prepend conditioning tokens to the sequence
        # Extract raw values for token conditioning
        # ELO: denormalize from scalars (was (elo - 1900) / 700)
        player_elo_raw = scalars[:, 0] * 700 + 1900  # (B,)
        
        # TC category is already in the batch
        # tc_cat: (B,) with values 0=Blitz, 1=Rapid, 2=Classical
        
        # Remaining time: denormalize from scalars (was log1p(seconds) / 10)
        # Use exp(x) - 1 instead of expm1 for ONNX compatibility
        remaining_time_raw = torch.exp(scalars[:, 3] * 10) - 1  # (B,)
        
        # Increment: denormalize from scalars (was inc / 30)
        increment_raw = (scalars[:, 5] * 30).long()  # (B,)
        
        # My last move time: time_history[:, 1] (index 1 is my last move, index 0 is opponent's last)
        # Denormalize from /60 and handle edge cases
        my_last_time_raw = time_history[:, 1] * 60  # (B,) in seconds
        
        # Opponent's last move time: time_history[:, 0]
        opp_last_time_raw = time_history[:, 0] * 60  # (B,) in seconds
        
        # Generate conditioning token embeddings
        cond_tokens = self.token_conditioning(
            player_elo=player_elo_raw,
            tc_cat=tc_cat,
            remaining_time=remaining_time_raw,
            increment=increment_raw,
            my_last_time=my_last_time_raw,
            opp_last_time=opp_last_time_raw,
        )  # (B, 6, d_model)
        
        # Prepend conditioning tokens to square tokens
        x = torch.cat([cond_tokens, x], dim=1)  # (B, 70, d_model)
        
        # Compute smolgen bias once (shared across layers)
        # Smolgen is designed for 64x64 square attention, apply only to square-to-square
        smolgen_bias = None
        if self.use_smolgen:
            # Compute smolgen on square tokens only
            square_tokens = x[:, NUM_CONDITIONING_TOKENS:, :]  # (B, 64, d_model)
            smolgen_bias_64 = self.smolgen(square_tokens)  # (B, n_heads, 64, 64)
            
            # Extend to 70x70 with zeros for conditioning token interactions
            n_heads = smolgen_bias_64.shape[1]
            smolgen_bias = torch.zeros(
                B, n_heads, 70, 70, 
                device=device, dtype=smolgen_bias_64.dtype
            )
            smolgen_bias[:, :, NUM_CONDITIONING_TOKENS:, NUM_CONDITIONING_TOKENS:] = smolgen_bias_64
        
        # Transformer layers
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # use_reentrant=False is required for torch.compile compatibility
                x = gradient_checkpoint(
                    layer, x, self.extended_rel_indices, smolgen_bias,
                    use_reentrant=False
                )
            else:
                x = layer(x, rel_indices=self.extended_rel_indices, smolgen_bias=smolgen_bias)
        x = self.norm_f(x)  # (B, 70, d_model)
        
        # Extract square tokens for output heads
        x_squares = x[:, NUM_CONDITIONING_TOKENS:, :]  # (B, 64, d_model)
        
        # --- Policy Head ---
        pol_q = self.policy_query(x_squares)  # (B, 64, d_model)
        pol_k = self.policy_key(x_squares)    # (B, 64, d_model)
        
        # Move logits: (B, 64, 64) -> (B, 4096)
        move_logits = torch.matmul(pol_q, pol_k.transpose(1, 2)) * self.policy_scale
        move_logits = move_logits.view(B, -1)
        
        # Add resign and flag logits
        # Score = mean query dot product with special keys
        mean_q = pol_q.mean(dim=1)  # (B, d_model)
        resign_logit = (mean_q * self.resign_key).sum(dim=-1, keepdim=True) * self.policy_scale
        flag_logit = (mean_q * self.flag_key).sum(dim=-1, keepdim=True) * self.policy_scale
        
        # Concatenate: (B, 4098)
        move_logits = torch.cat([move_logits, resign_logit, flag_logit], dim=-1)
        
        # Apply legal move mask if provided (with fix for 4096 vs 4098 mask size)
        if legal_mask is not None:
            mask_dim = legal_mask.shape[1]
            if mask_dim == 4096:
                # Append True for Resign/Flag (always legal to resign/flag in training)
                extras = torch.ones(B, 2, dtype=torch.bool, device=legal_mask.device)
                full_mask = torch.cat([legal_mask, extras], dim=1)
            else:
                full_mask = legal_mask
            move_logits = move_logits.masked_fill(~full_mask, float('-inf'))
        
        # --- Promotion Head ---
        # Get key vectors for promotion rank (squares 56-63)
        promo_keys = pol_k[:, 56:64, :]  # (B, 8, d_model)
        promo_logits = self.promo_bias_proj(promo_keys)  # (B, 8, 4)
        
        # --- Aux: Start Square Head ---
        # Predicts which of the 64 squares the piece moves FROM
        start_square_logits = self.start_square_head(x_squares).squeeze(-1)  # (B, 64)
        
        # --- Value and Time Heads ---
        # Value uses PaperValueHead (flattened spatial structure)
        value_out, value_cls_out = self.value_head(x_squares)  # (B, 1), (B, 3)

        # Value error models volatility
        value_error_feat = self.value_error_pool(x_squares)
        value_error_out = self.value_error_head(value_error_feat)
        
        # Time uses attention pooling
        time_pooled = self.time_pooling(x_squares)    # (B, d_model)
        time_cls_out = self.time_head(time_pooled)     # (B, 256) time bin logits
        
        if return_promo:
            return move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits, promo_logits
        return move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("--- Testing Small Config (~23M params) ---")
    model = Chessformer(CONFIG_SMALL)
    print(f"Small Config Trainable Parameters: {count_parameters(model):,}")
    
    # Create dummy batch
    B = 2
    dummy_batch = {
        'board_history': torch.randint(0, 13, (B, 8, 64)),
        'time_history': torch.randn(B, 8),
        'rep_flags': torch.randint(0, 2, (B, 8)).float(),
        'castling': torch.randint(0, 2, (B, 4)).float(),
        'ep_mask': torch.zeros(B, 64),
        'scalars': torch.randn(B, 8),
        'tc_cat': torch.randint(0, 3, (B,)),
        'legal_mask': torch.ones(B, NUM_POLICY_OUTPUTS, dtype=torch.bool),
    }
    
    move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits = model(dummy_batch)
    print(f"Move logits: {move_logits.shape}")           # (2, 4098)
    print(f"Value (reg): {value_out.shape}")             # (2, 1)
    print(f"Value (cls): {value_cls_out.shape}")         # (2, 3) WDL
    print(f"Value error: {value_error_out.shape}")       # (2, 1)
    print(f"Time (cls): {time_cls_out.shape}")           # (2, 256) bins
    print(f"Start square logits: {start_square_logits.shape}")  # (2, 64)
    
    # Test large config
    print("\n--- Testing Large Config (~100M params) ---")
    model_large = Chessformer(CONFIG_LARGE)
    print(f"Large Config Trainable Parameters: {count_parameters(model_large):,}")
    
    # Run forward pass with large config
    move_logits, value_out, value_cls_out, value_error_out, time_cls_out, start_square_logits = model_large(dummy_batch)
    print(f"Move logits: {move_logits.shape}")           # (2, 4098)
    print(f"Value (reg): {value_out.shape}")             # (2, 1)
    print(f"Value (cls): {value_cls_out.shape}")         # (2, 3) WDL
    print(f"Value error: {value_error_out.shape}")       # (2, 1)
    print(f"Time (cls): {time_cls_out.shape}")           # (2, 256) bins
    print(f"Start square logits: {start_square_logits.shape}")  # (2, 64)