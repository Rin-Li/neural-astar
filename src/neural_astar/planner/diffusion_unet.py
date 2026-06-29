"""Conditional U-Net for DDPM trajectory-heatmap generation.

Ported from GDPP (gdpp/src/gdpp/network/unet.py). Depends only on torch.
The conditioning is (obstacle_map, start_heatmap, goal_heatmap) = 3 channels,
concatenated with the noisy target on the channel dimension. Fully convolutional,
so it works on 32x32 as well as 64x64 inputs.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=device, dtype=torch.float32) * -scale)
        args = timesteps.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimeMLP(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.net(timesteps)


def _groups(channels: int) -> int:
    return min(8, channels)


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels * 2)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int, downsample: bool) -> None:
        super().__init__()
        self.res1 = ResBlock(in_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim)
        self.down = nn.Conv2d(out_channels, out_channels, 4, stride=2, padding=1) if downsample else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return self.down(x), x


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, time_emb_dim: int, upsample: bool) -> None:
        super().__init__()
        self.res1 = ResBlock(in_channels + skip_channels, out_channels, time_emb_dim)
        self.res2 = ResBlock(out_channels, out_channels, time_emb_dim)
        self.up = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1) if upsample else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return self.up(x)


class ConvStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool) -> None:
        super().__init__()
        stride = 2 if downsample else 1
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        stages = []
        current = in_channels
        for idx, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            stages.append(ConvStage(current, out_channels, downsample=idx > 0))
            current = out_channels
        self.stages = nn.ModuleList(stages)

    def forward(self, cond: torch.Tensor) -> list[torch.Tensor]:
        features = []
        x = cond
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features


class CrossAttention2D(nn.Module):
    def __init__(self, x_channels: int, cond_channels: int, num_heads: int = 8) -> None:
        super().__init__()
        if x_channels % num_heads != 0:
            raise ValueError(f"x_channels={x_channels} must be divisible by num_heads={num_heads}")
        self.norm_x = nn.GroupNorm(_groups(x_channels), x_channels)
        self.norm_c = nn.GroupNorm(_groups(cond_channels), cond_channels)
        self.q_proj = nn.Linear(x_channels, x_channels)
        self.k_proj = nn.Linear(cond_channels, x_channels)
        self.v_proj = nn.Linear(cond_channels, x_channels)
        self.attn = nn.MultiheadAttention(x_channels, num_heads=num_heads, batch_first=True)
        self.out_proj = nn.Linear(x_channels, x_channels)

    def forward(self, x: torch.Tensor, cond_feat: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if cond_feat.shape[-2:] != (height, width):
            cond_feat = F.interpolate(cond_feat, size=(height, width), mode="nearest")

        x_tokens = self.norm_x(x).flatten(2).transpose(1, 2)
        cond_tokens = self.norm_c(cond_feat).flatten(2).transpose(1, 2)
        query = self.q_proj(x_tokens)
        key = self.k_proj(cond_tokens)
        value = self.v_proj(cond_tokens)
        out, _ = self.attn(query, key, value, need_weights=False)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(batch, channels, height, width)
        return x + out


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
    ) -> None:
        super().__init__()
        self.time_mlp = TimeMLP(time_emb_dim)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * mult for mult in channel_mults]
        self.downs = nn.ModuleList()
        current = base_channels
        for idx, channels_out in enumerate(channels):
            self.downs.append(
                DownBlock(current, channels_out, time_emb_dim, downsample=idx < len(channels) - 1)
            )
            current = channels_out

        self.mid1 = ResBlock(current, current, time_emb_dim)
        self.mid2 = ResBlock(current, current, time_emb_dim)

        self.ups = nn.ModuleList()
        for idx, skip_channels in enumerate(reversed(channels)):
            is_last = idx == len(channels) - 1
            out_ch = skip_channels
            self.ups.append(
                UpBlock(current, skip_channels, out_ch, time_emb_dim, upsample=not is_last)
            )
            current = out_ch

        self.final = nn.Sequential(
            nn.GroupNorm(_groups(current), current),
            nn.SiLU(),
            nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        obstacle_map: torch.Tensor,
        start_heatmap: torch.Tensor,
        goal_heatmap: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        model_input = torch.cat([x_t, obstacle_map, start_heatmap, goal_heatmap], dim=1)
        t_emb = self.time_mlp(timesteps)

        x = self.init_conv(model_input)
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.mid1(x, t_emb)
        x = self.mid2(x, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)

        return self.final(x)


class ConditionalAttentionUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 64,
        time_emb_dim: int = 256,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        cond_channels: int = 3,
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        x_channels = in_channels - cond_channels
        if x_channels <= 0:
            raise ValueError(f"in_channels must include target + {cond_channels} condition channels")

        self.time_mlp = TimeMLP(time_emb_dim)
        self.cond_encoder = ConditionEncoder(cond_channels, base_channels, channel_mults)
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        channels = [base_channels * mult for mult in channel_mults]
        self.downs = nn.ModuleList()
        current = base_channels
        for idx, channels_out in enumerate(channels):
            self.downs.append(
                DownBlock(current, channels_out, time_emb_dim, downsample=idx < len(channels) - 1)
            )
            current = channels_out

        self.mid1 = ResBlock(current, current, time_emb_dim)
        self.cross_attn = CrossAttention2D(
            x_channels=current,
            cond_channels=channels[-1],
            num_heads=attention_heads,
        )
        self.mid2 = ResBlock(current, current, time_emb_dim)

        self.ups = nn.ModuleList()
        for idx, skip_channels in enumerate(reversed(channels)):
            is_last = idx == len(channels) - 1
            out_ch = skip_channels
            self.ups.append(
                UpBlock(current, skip_channels, out_ch, time_emb_dim, upsample=not is_last)
            )
            current = out_ch

        self.final = nn.Sequential(
            nn.GroupNorm(_groups(current), current),
            nn.SiLU(),
            nn.Conv2d(current, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        obstacle_map: torch.Tensor,
        start_heatmap: torch.Tensor,
        goal_heatmap: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        cond = torch.cat([obstacle_map, start_heatmap, goal_heatmap], dim=1)
        cond_features = self.cond_encoder(cond)
        model_input = torch.cat([x_t, cond], dim=1)
        t_emb = self.time_mlp(timesteps)

        x = self.init_conv(model_input)
        skips = []
        for down in self.downs:
            x, skip = down(x, t_emb)
            skips.append(skip)

        x = self.mid1(x, t_emb)
        x = self.cross_attn(x, cond_features[-1])
        x = self.mid2(x, t_emb)

        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, t_emb)

        return self.final(x)


def build_unet_from_config(config: dict) -> nn.Module:
    target_channels = int(config.get("target_channels", config.get("out_channels", 1)))
    model_type = config.get("model_type", "baseline")
    kwargs = {
        "in_channels": config.get("in_channels", target_channels + 3),
        "out_channels": config.get("out_channels", target_channels),
        "base_channels": config.get("base_channels", 64),
        "time_emb_dim": config.get("time_emb_dim", 256),
        "channel_mults": tuple(config.get("channel_mults", (1, 2, 4, 8))),
    }
    if model_type in {"baseline", "conditional_unet"}:
        return ConditionalUNet(**kwargs)
    if model_type in {"attention", "conditional_attention_unet"}:
        return ConditionalAttentionUNet(
            **kwargs,
            cond_channels=config.get("cond_channels", 3),
            attention_heads=config.get("attention_heads", 8),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")
