"""Dual-branch spatio-temporal fusion model for solar storm forecasting.

Branch 1 — Spatial: EfficientNet-B3 (5-channel input) applied to 4 time-steps.
Branch 2 — Temporal: Bidirectional LSTM over OMNI2 time series.
Fusion   — Cross-attention  + MLP decoder with two heads (point & uncertainty).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


# ---------------------------------------------------------------------------
# Branch 1 — Image encoder (EfficientNet-B3, shared across time-steps)
# ---------------------------------------------------------------------------


class ImageEncoder(nn.Module):
    """EfficientNet-B3 adapted to 5-channel solar images.

    Pretrained ImageNet weights are loaded and the first conv layer is
    adapted by averaging the 3-channel weights across the 5 new input
    channels.  The last 2 blocks are unfrozen for fine-tuning.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            cfg.efficientnet_variant,
            pretrained=True,
            in_chans=cfg.image_channels,
            num_classes=0,          # strip classifier → global pool output
            global_pool="avg",
        )
        backbone_out_dim = self.backbone.num_features  # 1536 for effnet-b3
        self.proj = nn.Linear(backbone_out_dim, cfg.image_feature_dim)

        self._freeze_early_blocks()

    # ------------------------------------------------------------------
    def _freeze_early_blocks(self) -> None:
        """Freeze everything except the last 2 EfficientNet blocks."""
        # timm EfficientNet exposes `.blocks` — a nn.Sequential of stages
        blocks = list(self.backbone.blocks.children())
        n_blocks = len(blocks)
        unfreeze_from = max(n_blocks - 2, 0)
        for i, block in enumerate(blocks):
            if i < unfreeze_from:
                for p in block.parameters():
                    p.requires_grad = False

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(batch, T, C, H, W)`` — batch of T time-steps, C channels.

        Returns
        -------
        Tensor
            Shape ``(batch, T, image_feature_dim)``
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feat = self.backbone(x)         # (B*T, backbone_out_dim)
        feat = self.proj(feat)          # (B*T, image_feature_dim)
        return feat.view(B, T, -1)     # (B, T, F)

    @property
    def last_conv(self) -> nn.Module:
        """Return the last conv layer — useful for Grad-CAM hooks."""
        return self.backbone.blocks[-1]


# ---------------------------------------------------------------------------
# Branch 2 — Temporal OMNI2 encoder (BiLSTM)
# ---------------------------------------------------------------------------


class TemporalEncoder(nn.Module):
    """2-layer bidirectional LSTM over OMNI2 features.

    Takes ``(batch, look_back_x, 8)`` and outputs a 256-d summary vector
    (final hidden state concatenated from both directions).
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.n_features = len(cfg.omni_col_names)
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
        )
        self.out_dim = cfg.lstm_hidden * 2  # bidirectional

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Shape ``(batch, look_back_x, n_features)``

        Returns
        -------
        seq_out : Tensor
            Full sequence output ``(batch, look_back_x, 256)``.
        summary : Tensor
            Last-step summary ``(batch, 256)``.
        """
        seq_out, (h_n, _c_n) = self.lstm(x)
        # h_n: (num_layers*2, batch, hidden) — take last layer both dirs
        fwd = h_n[-2]   # (batch, hidden)
        bwd = h_n[-1]   # (batch, hidden)
        summary = torch.cat([fwd, bwd], dim=-1)  # (batch, 256)
        return seq_out, summary


# ---------------------------------------------------------------------------
# Fusion — cross-attention
# ---------------------------------------------------------------------------


class FusionLayer(nn.Module):
    """Cross-attention: LSTM summary queries image time-step features.

    Then concatenate attended image features with LSTM summary and project.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        lstm_dim = cfg.lstm_hidden * 2          # 256
        img_dim = cfg.image_feature_dim         # 512

        # Project LSTM summary to query dim matching image features
        self.query_proj = nn.Linear(lstm_dim, img_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=img_dim,
            num_heads=cfg.fusion_heads,
            kdim=img_dim,
            vdim=img_dim,
            batch_first=True,
            dropout=cfg.dropout,
        )
        # Output: concat [attended (512) + lstm_summary(256)] → 768 → 256
        self.layer_norm = nn.LayerNorm(img_dim + lstm_dim)
        self.fc = nn.Linear(img_dim + lstm_dim, cfg.decoder_hidden)
        self.act = nn.GELU()

    def forward(
        self,
        img_features: torch.Tensor,
        lstm_summary: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        img_features : Tensor
            ``(batch, T, 512)``
        lstm_summary : Tensor
            ``(batch, 256)``

        Returns
        -------
        Tensor
            ``(batch, 256)``
        """
        query = self.query_proj(lstm_summary).unsqueeze(1)  # (B, 1, 512)
        attended, _ = self.cross_attn(query, img_features, img_features)
        attended = attended.squeeze(1)                       # (B, 512)
        fused = torch.cat([attended, lstm_summary], dim=-1)  # (B, 768)
        fused = self.layer_norm(fused)
        return self.act(self.fc(fused))                      # (B, 256)


# ---------------------------------------------------------------------------
# Decoder + output heads
# ---------------------------------------------------------------------------


class Decoder(nn.Module):
    """MLP decoder with two heads: point prediction and uncertainty."""

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.log_std_min = -6.0
        self.log_std_max = 2.0
        self.shared = nn.Sequential(
            nn.Linear(cfg.decoder_hidden, cfg.decoder_out),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        # Head 1 — point prediction: forecast_y values
        self.head_point = nn.Linear(cfg.decoder_out, cfg.forecast_y)
        # Head 2 — log-std for reparameterised uncertainty
        self.head_log_std = nn.Linear(cfg.decoder_out, cfg.forecast_y)

        # Single flux prediction (log10 peak flux)
        self.head_flux = nn.Linear(cfg.decoder_out, 1)
        self.head_flux_log_std = nn.Linear(cfg.decoder_out, 1)

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            ``(batch, 256)`` — fused representation.

        Returns
        -------
        dict with keys:
            ``dst_pred``: ``(batch, forecast_y)``
            ``dst_log_std``: ``(batch, forecast_y)``
            ``flux_pred``: ``(batch, 1)``
            ``flux_log_std``: ``(batch, 1)``
        """
        h = self.shared(x)
        return {
            "dst_pred": self.head_point(h),
            "dst_log_std": self.head_log_std(h).clamp(
                min=self.log_std_min, max=self.log_std_max
            ),
            "flux_pred": self.head_flux(h).squeeze(-1),
            "flux_log_std": self.head_flux_log_std(h).squeeze(-1).clamp(
                min=self.log_std_min, max=self.log_std_max
            ),
        }


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class SolarStormModel(nn.Module):
    """Dual-branch spatio-temporal fusion model for solar storm forecasting.

    Combines:
    - ``ImageEncoder``: EfficientNet-B3 across 4 SDO time-steps
    - ``TemporalEncoder``: BiLSTM over OMNI2 time-series
    - ``FusionLayer``: Cross-attention merging both branches
    - ``Decoder``: Two-head MLP (point + uncertainty)
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.image_encoder = ImageEncoder(cfg)
        self.temporal_encoder = TemporalEncoder(cfg)
        self.fusion = FusionLayer(cfg)
        self.decoder = Decoder(cfg)

    def forward(
        self,
        images: torch.Tensor,
        omni: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        images : Tensor
            ``(B, 4, 5, 256, 256)``
        omni : Tensor
            ``(B, look_back_x, 8)``
        image_mask : Tensor, optional
            ``(B, 4, 5)`` — ``True`` where a channel image exists.

        Returns
        -------
        dict
            ``dst_pred``, ``dst_log_std``, ``flux_pred``, ``flux_log_std``
        """
        # Zero out missing channels explicitly (safety)
        if image_mask is not None:
            # Expand mask to spatial dims: (B, 4, 5) → (B, 4, 5, 1, 1)
            m = image_mask.unsqueeze(-1).unsqueeze(-1).float()
            images = images * m

        img_feat = self.image_encoder(images)              # (B, 4, 512)
        _, lstm_summary = self.temporal_encoder(omni)      # (B, 256)
        fused = self.fusion(img_feat, lstm_summary)        # (B, 256)
        return self.decoder(fused)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def asymmetric_log_mae(
    pred_log: torch.Tensor,
    target_log: torch.Tensor,
    alpha: float = 2.0,
) -> torch.Tensor:
    """Log-space asymmetric MAE: extra penalty for under-predicting flare intensity.

    Parameters
    ----------
    pred_log : Tensor
        Predicted log10 values.
    target_log : Tensor
        Ground-truth log10 values.
    alpha : float
        Multiplicative penalty when the model under-predicts (``target > pred``).
    """
    residual = target_log - pred_log  # positive ⇒ under-prediction
    weight = torch.where(
        residual > 0,
        alpha * torch.ones_like(residual),
        torch.ones_like(residual),
    )
    return (weight * torch.abs(residual)).mean()


def gaussian_nll(
    pred: torch.Tensor,
    log_std: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Negative log-likelihood assuming diagonal Gaussian.

    Used as an auxiliary loss to train the uncertainty head.
    """
    std = (torch.exp(log_std) + 1e-6).clamp(min=1e-6)
    return (0.5 * ((target - pred) / std) ** 2 + log_std).mean()


def combined_loss(
    outputs: Dict[str, torch.Tensor],
    target_log_flux: torch.Tensor,
    target_log_dst: torch.Tensor,
    cfg: Optional[Config] = None,
    alpha: Optional[float] = None,
    dst_weight: float = 0.5,
    nll_weight: float = 0.1,
) -> torch.Tensor:
    """Weighted combination of asymmetric MAE + Gaussian NLL for both heads.

    Parameters
    ----------
    outputs : dict
        Model output dict.
    target_log_flux : Tensor
        ``(B,)``
    target_log_dst : Tensor
        ``(B, forecast_y)``
    cfg : Config, optional
        Supplies ``asymmetric_alpha`` when ``alpha`` is not passed explicitly.
    alpha : float, optional
        Explicit asymmetric penalty. Falls back to ``cfg.asymmetric_alpha`` and
        finally ``2.0`` if neither override is provided.
    """
    if alpha is None:
        alpha = cfg.asymmetric_alpha if cfg is not None else 2.0

    # Flux loss
    flux_mae = asymmetric_log_mae(outputs["flux_pred"], target_log_flux, alpha)
    flux_nll = gaussian_nll(
        outputs["flux_pred"], outputs["flux_log_std"], target_log_flux
    )

    # Dst loss
    dst_mae = asymmetric_log_mae(outputs["dst_pred"], target_log_dst, alpha)
    dst_nll = gaussian_nll(
        outputs["dst_pred"], outputs["dst_log_std"], target_log_dst
    )

    total = flux_mae + nll_weight * flux_nll + dst_weight * (dst_mae + nll_weight * dst_nll)
    return total
