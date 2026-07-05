"""
Griffin with the Complex-Gated Linear Recurrent Unit (CG-LRU)
=============================================================

A self-contained PyTorch reference implementation of the architecture from:

    De, Smith, Fernando, Botev et al.,
    "Griffin: Mixing Gated Linear Recurrences with Local Attention
     for Efficient Language Models" (2024), arXiv:2402.19427

This implements the *complex-gated* variant of the recurrent layer described
in Appendix B (CG-LRU), plugged into the full Griffin stack:

    - Gated MLP block                (Section 2.2)
    - Recurrent block: Linear -> causal depthwise Conv1D -> CG-LRU,
      gated by a parallel GeLU branch (Section 2.3)
    - Local sliding-window MQA with RoPE   (Section 2.3)
    - Alternating residual backbone: [recurrent, recurrent, local-attn] x k
      (Section 3, "Griffin")

The recurrence is evaluated with a simple sequential (linear) scan over time,
which is the clear/correct baseline. The paper's speedups come from a custom
Pallas linear-scan kernel; for a PyTorch port you'd swap this loop for an
associative scan or a fused kernel, but the math below is what matters.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class GriffinConfig:
    vocab_size: int = 32000
    d_model: int = 2048        # D            (model width)
    d_rnn: int = 2560          # D_RNN        (recurrent width; MUST be even for CG-LRU)
    depth: int = 24            # N            (number of residual blocks)
    mlp_expansion: int = 3     # M            (gated MLP expansion factor)
    head_dim: int = 128        # d_head       (fixed at 128 in the paper)
    window_size: int = 1024    # local attention window
    conv_kernel: int = 4       # temporal Conv1D filter width
    rg_lru_c: float = 8.0      # scalar constant c in a = sigmoid(Lambda), a_t = a^(c r_t)
    rope_base: float = 10000.0
    attn_period: int = 3       # every 3rd block is local attention (2 recurrent : 1 attn)
    parallel_scan: bool = True # use log-depth scan (fast) vs sequential loop (reference)
    dropout: float = 0.0       # residual/attention/embedding dropout (0 = off)

    @property
    def num_heads(self) -> int:
        assert self.d_model % self.head_dim == 0, "d_model must be a multiple of head_dim"
        return self.d_model // self.head_dim


# --------------------------------------------------------------------------- #
# Norm + MLP
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.weight


class GatedMLP(nn.Module):
    """Gated MLP (GeGeLU): two D->MD branches, GeLU-gate one, multiply, MD->D."""

    def __init__(self, cfg: GriffinConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.mlp_expansion
        self.gate = nn.Linear(cfg.d_model, hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x)) * self.up(x))


# --------------------------------------------------------------------------- #
# Complex-Gated Linear Recurrent Unit (CG-LRU)  -- Appendix B, eqs (8)-(14)
# --------------------------------------------------------------------------- #
class CGLRU(nn.Module):
    r"""
    Complex-gated linear recurrence.

    Input x_t is real of size d_rnn. Its channels are split in half and read as
    the real / imaginary parts of a complex vector x~_t of size d_rnn/2:

        x~_t = x_t[:half] + i * x_t[half:]

    Per-channel complex recurrence parameter:
        a~   = sigmoid(Lambda) * exp(i * theta)          (|a~| = sigmoid(Lambda))

    Gates (real, half-dim), computed from the *full* real input:
        r_t  = sigmoid(W_a x_t + b_a)                    recurrence gate
        i_t  = sigmoid(W_x x_t + b_x)                    input gate

    Recurrence (all element-wise, complex):
        a~_t = a~ ** (c * r_t)
        h~_t = a~_t * h~_{t-1} + sqrt(1 - |a~_t|^2) * (i_t * x~_t)

    Output stacks real and imaginary parts back to d_rnn:
        y_t  = [ Re(h~_t) ; Im(h~_t) ]

    Numerics: |a~_t| is kept in log-space via log|a~| = log-sigmoid(Lambda),
    so a~_t = exp( c * r_t * (log-sigmoid(Lambda) + i*theta) ). This avoids the
    a**power instability the paper flags in Appendix A.
    """

    def __init__(self, d_rnn: int, c: float = 8.0, parallel_scan: bool = True):
        super().__init__()
        assert d_rnn % 2 == 0, "d_rnn must be even for the complex (CG-LRU) variant"
        self.d_rnn = d_rnn
        self.half = d_rnn // 2
        self.c = c
        self.parallel_scan = parallel_scan

        # Gates map the full real input (d_rnn) -> half-dim gate values.
        self.recurrence_gate = nn.Linear(d_rnn, self.half)
        self.input_gate = nn.Linear(d_rnn, self.half)

        # Complex recurrence params, one per (half) channel.
        self.lamb = nn.Parameter(torch.empty(self.half))   # magnitude logit
        self.theta = nn.Parameter(torch.empty(self.half))  # phase
        self.reset_parameters()

    def reset_parameters(self):
        # LeCun init for the gate projections (std = 1/sqrt(fan_in)).
        for lin in (self.recurrence_gate, self.input_gate):
            fan_in = lin.weight.shape[1]
            nn.init.normal_(lin.weight, std=fan_in ** -0.5)
            nn.init.zeros_(lin.bias)

        # Init so |a~|^c ~ U(0.9, 0.999), with |a~| = sigmoid(Lambda)  (paper Sec 2.4).
        a_c = torch.empty(self.half).uniform_(0.9, 0.999)     # this is |a~|^c
        mag = a_c ** (1.0 / self.c)                           # = sigmoid(Lambda)
        self.lamb.data = torch.log(mag) - torch.log1p(-mag)   # logit(mag)
        # Small initial phases (as in the LRU line of work).
        self.theta.data.uniform_(0.0, math.pi / 10.0)

    def forward(self, x):                       # x: [B, T, d_rnn] (real)
        B, T, _ = x.shape
        r = torch.sigmoid(self.recurrence_gate(x))     # [B,T,half]
        ig = torch.sigmoid(self.input_gate(x))         # [B,T,half]

        # Real input -> complex state input.
        x_c = torch.complex(x[..., :self.half], x[..., self.half:])   # [B,T,half]

        # log a~_t = c * r_t * (log|a~| + i*theta),  log|a~| = log-sigmoid(Lambda)
        log_mag = F.logsigmoid(self.lamb)              # [half]  (<= 0)
        log_a_re = self.c * r * log_mag                # [B,T,half]
        log_a_im = self.c * r * self.theta             # [B,T,half]
        a_t = torch.exp(torch.complex(log_a_re, log_a_im))           # [B,T,half] complex

        # sqrt(1 - |a~_t|^2), with |a~_t|^2 = exp(2 * log_a_re)
        mag_sq = torch.exp(2.0 * log_a_re)
        scale = torch.sqrt(torch.clamp(1.0 - mag_sq, min=1e-6))       # [B,T,half] real

        # Driving term: sqrt(1-|a|^2) * (i_t * x~_t)   -> complex
        drive = x_c * (scale * ig).to(x_c.dtype)                      # [B,T,half] complex

        # Solve the complex linear recurrence  h_t = a_t * h_{t-1} + drive_t.
        scan = self._parallel_scan if self.parallel_scan else self._sequential_scan
        h_seq = scan(a_t, drive)                       # [B,T,half] complex

        return torch.cat([h_seq.real, h_seq.imag], dim=-1)           # [B,T,d_rnn]

    # --- scan implementations ------------------------------------------------ #
    @staticmethod
    def _sequential_scan(a, b):
        """Reference O(T) loop. Clear and exact; slow due to per-step kernel launches."""
        B, T, H = a.shape
        h = torch.zeros(B, H, dtype=a.dtype, device=a.device)
        outs = []
        for t in range(T):
            h = a[:, t] * h + b[:, t]
            outs.append(h)
        return torch.stack(outs, dim=1)

    @staticmethod
    def _parallel_scan(a, b):
        """
        Hillis-Steele inclusive scan of the affine recurrence
            h_t = a_t * h_{t-1} + b_t          (h_{-1} = 0).

        The recurrence is a composition of affine maps h -> a*h + b, which is
        associative with combine( (a_e,b_e), (a_l,b_l) ) = (a_l a_e, a_l b_e + b_l).
        This runs in ceil(log2 T) fully-vectorized steps instead of T Python
        iterations -- the win that matters on old GPUs, where launch overhead
        dominates the tiny per-step elementwise work.
        """
        B, T, H = a.shape
        d = 1
        while d < T:
            ones = torch.ones(B, d, H, dtype=a.dtype, device=a.device)
            zeros = torch.zeros(B, d, H, dtype=b.dtype, device=b.device)
            a_prev = torch.cat([ones, a[:, :-d]], dim=1)     # identity (a=1) for t<d
            b_prev = torch.cat([zeros, b[:, :-d]], dim=1)    # identity (b=0) for t<d
            b = a * b_prev + b                               # update b using old a
            a = a * a_prev                                   # then update a
            d *= 2
        return b


# --------------------------------------------------------------------------- #
# Recurrent block  (Section 2.3, Figure 2c)
# --------------------------------------------------------------------------- #
class RecurrentBlock(nn.Module):
    def __init__(self, cfg: GriffinConfig):
        super().__init__()
        self.in_rnn = nn.Linear(cfg.d_model, cfg.d_rnn, bias=False)   # branch 1
        self.in_gate = nn.Linear(cfg.d_model, cfg.d_rnn, bias=False)  # branch 2
        # Separable (depthwise) causal Conv1D, temporal filter width 4  (4 * d_rnn params).
        self.conv = nn.Conv1d(
            cfg.d_rnn, cfg.d_rnn,
            kernel_size=cfg.conv_kernel,
            groups=cfg.d_rnn,
            padding=cfg.conv_kernel - 1,
        )
        self.cglru = CGLRU(cfg.d_rnn, cfg.rg_lru_c, cfg.parallel_scan)
        self.out = nn.Linear(cfg.d_rnn, cfg.d_model, bias=False)

    def forward(self, x):
        T = x.shape[1]
        # Branch 1: linear -> causal depthwise conv -> CG-LRU
        b1 = self.in_rnn(x).transpose(1, 2)            # [B, d_rnn, T]
        b1 = self.conv(b1)[..., :T].transpose(1, 2)    # causal truncate -> [B,T,d_rnn]
        b1 = self.cglru(b1)
        # Branch 2: linear -> GeLU
        b2 = F.gelu(self.in_gate(x))
        return self.out(b1 * b2)


# --------------------------------------------------------------------------- #
# Local sliding-window Multi-Query Attention + RoPE  (Section 2.3)
# --------------------------------------------------------------------------- #
class LocalMQA(nn.Module):
    """
    Sliding-window causal MQA, evaluated blockwise so cost is O(T*W) instead
    of O(T^2): chop the sequence into non-overlapping blocks of size W (the
    configured window), and let each query block attend only to itself plus
    the immediately preceding block. A query at local position i in block n
    can see at most W-1 positions back, and the preceding block holds exactly
    W positions immediately before the current block's start -- so
    "previous ++ current" (2W keys) is always a sufficient key context;
    nothing outside it would have survived the causal+window mask anyway.
    Mathematically identical to computing the full T x T scores and masking
    down to the local window (verified in this file's __main__ self-test),
    just without ever materializing the O(T^2) matrix.
    """

    def __init__(self, cfg: GriffinConfig):
        super().__init__()
        self.h = cfg.num_heads
        self.dh = cfg.head_dim
        self.window = cfg.window_size
        self.base = cfg.rope_base
        self.q = nn.Linear(cfg.d_model, self.h * self.dh, bias=False)
        self.k = nn.Linear(cfg.d_model, self.dh, bias=False)   # single KV head (MQA)
        self.v = nn.Linear(cfg.d_model, self.dh, bias=False)
        self.o = nn.Linear(self.h * self.dh, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)

        # Causal+window mask over a (prev-block ++ current-block) key context,
        # shared by every block except the first (no real previous block --
        # zeroed per-call in forward). Depends only on `window`, not on batch
        # or sequence length, so it's computed once here.
        W = self.window
        i = torch.arange(W).unsqueeze(1)             # [W,1]  query, local pos in current block
        c = torch.arange(2 * W).unsqueeze(0)         # [1,2W] key, position in prev++curr context
        prev_valid = (c < W) & (i < c)               # earlier block: only its last W-1 positions
        curr_valid = (c >= W) & ((c - W) <= i)        # current block: ordinary causal
        self.register_buffer("_mask", prev_valid | curr_valid, persistent=False)   # [W,2W] bool

    def _rope(self, x, T, device):                 # x: [..., T, dh]
        half = self.dh // 2
        freqs = self.base ** (-torch.arange(half, device=device).float() / half)
        ang = torch.outer(torch.arange(T, device=device).float(), freqs)   # [T, half]
        cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
        sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin

    def forward(self, x):
        B, T, _ = x.shape
        W = self.window
        device = x.device

        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)   # [B,H,T,dh]
        k = self.k(x).unsqueeze(1)                                   # [B,1,T,dh]
        v = self.v(x).unsqueeze(1)                                   # [B,1,T,dh]
        q = self._rope(q, T, device)                                 # rope uses TRUE positions,
        k = self._rope(k, T, device)                                 # applied before any padding

        pad = (-T) % W
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
        Tp = T + pad
        nb = Tp // W

        qb = q.view(B, self.h, nb, W, self.dh)
        kb = k.view(B, 1, nb, W, self.dh)
        vb = v.view(B, 1, nb, W, self.dh)

        k_prev = torch.cat([torch.zeros_like(kb[:, :, :1]), kb[:, :, :-1]], dim=2)  # [B,1,nb,W,dh]
        v_prev = torch.cat([torch.zeros_like(vb[:, :, :1]), vb[:, :, :-1]], dim=2)
        k_ctx = torch.cat([k_prev, kb], dim=3)                        # [B,1,nb,2W,dh]
        v_ctx = torch.cat([v_prev, vb], dim=3)

        scores = torch.matmul(qb, k_ctx.transpose(-1, -2)) / math.sqrt(self.dh)  # [B,H,nb,W,2W]

        mask = self._mask.unsqueeze(0).expand(nb, W, 2 * W).clone()
        mask[0, :, :W] = False                      # block 0 has no real previous block
        scores = scores.masked_fill(~mask.view(1, 1, nb, W, 2 * W), float("-inf"))

        attn = scores.softmax(-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v_ctx)                               # [B,H,nb,W,dh]
        out = out.reshape(B, self.h, Tp, self.dh)[:, :, :T]           # drop padding -> [B,H,T,dh]
        out = out.transpose(1, 2).reshape(B, T, self.h * self.dh)
        return self.o(out)


# --------------------------------------------------------------------------- #
# Residual block + full model  (Sections 2.1, 3)
# --------------------------------------------------------------------------- #
class ResidualBlock(nn.Module):
    def __init__(self, cfg: GriffinConfig, temporal: nn.Module):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.temporal = temporal
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = GatedMLP(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.temporal(self.norm1(x)))
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x


class Griffin(nn.Module):
    def __init__(self, cfg: GriffinConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        blocks = []
        for i in range(cfg.depth):
            # Pattern: two recurrent blocks then one local-attention block.
            if (i + 1) % cfg.attn_period == 0:
                temporal = LocalMQA(cfg)
            else:
                temporal = RecurrentBlock(cfg)
            blocks.append(ResidualBlock(cfg, temporal))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(cfg.d_model)
        self.embed_drop = nn.Dropout(cfg.dropout)
        # Output projection weights are tied to the input embedding.

        # Weight init. Small std keeps initial logits ~uniform (loss ~ ln(vocab));
        # the default nn.Embedding std=1 otherwise explodes the tied output logits.
        self.apply(self._init_weights)
        # Restore the CG-LRU's specialized init (Lambda/theta + LeCun gates),
        # which the generic pass above would have overwritten.
        for m in self.modules():
            if isinstance(m, CGLRU):
                m.reset_parameters()

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens):                     # tokens: [B, T] long
        x = self.embed_drop(self.embed(tokens))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_f(x)
        return x @ self.embed.weight.t()           # [B, T, vocab]  (tied)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


def _brute_force_local_mqa(mqa, x):
    """Reference O(T^2) computation (the old LocalMQA implementation) --
    full scores, then masked down to the causal+window band. Test-only: the
    real forward() never materializes this."""
    B, T, _ = x.shape
    q = mqa.q(x).view(B, T, mqa.h, mqa.dh).transpose(1, 2)
    k = mqa.k(x).unsqueeze(1)
    v = mqa.v(x).unsqueeze(1)
    q = mqa._rope(q, T, x.device)
    k = mqa._rope(k, T, x.device)
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(mqa.dh)
    idx = torch.arange(T, device=x.device)
    causal = idx[None, :] <= idx[:, None]
    in_window = (idx[:, None] - idx[None, :]) < mqa.window
    mask = causal & in_window
    scores = scores.masked_fill(~mask, float("-inf"))
    attn = scores.softmax(-1)
    out = (attn @ v).transpose(1, 2).reshape(B, T, mqa.h * mqa.dh)
    return mqa.o(out)


# --------------------------------------------------------------------------- #
# Sanity check
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)

    # LocalMQA's O(T*W) blockwise attention must be numerically identical to
    # the O(T^2) full-scores-then-mask reference it replaced -- same causal +
    # window pattern, just never materializing the T x T matrix. Check across
    # T shorter than / equal to / not a multiple of / much larger than window.
    for T, W in [(15, 32), (32, 32), (40, 32), (100, 32), (65, 16)]:
        mqa = LocalMQA(GriffinConfig(vocab_size=256, d_model=128, d_rnn=192, depth=1,
                                     head_dim=32, window_size=W, dropout=0.0)).eval()
        xin = torch.randn(2, T, 128)
        with torch.no_grad():
            out_block = mqa(xin)
            out_brute = _brute_force_local_mqa(mqa, xin)
        max_diff = (out_block - out_brute).abs().max().item()
        assert torch.allclose(out_block, out_brute, atol=1e-5), \
            f"LocalMQA mismatch at T={T}, W={W}: max diff {max_diff}"
        print(f"LocalMQA blockwise == brute-force at T={T:3d} W={W:3d}: OK (max diff {max_diff:.2e})")

    # Small config so it runs on CPU in a second.
    cfg = GriffinConfig(
        vocab_size=256, d_model=256, d_rnn=384, depth=6,
        head_dim=64, window_size=32,
    )
    model = Griffin(cfg)
    print(f"num_heads={cfg.num_heads}  params={model.num_params()/1e6:.2f}M")

    x = torch.randint(0, cfg.vocab_size, (2, 40))
    logits = model(x)
    print("logits:", tuple(logits.shape))          # -> (2, 40, 256)

    # Backward works end-to-end through the complex recurrence.
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, cfg.vocab_size),
                           x[:, 1:].reshape(-1))
    loss.backward()
    print("loss:", loss.item(), "| grad OK")
