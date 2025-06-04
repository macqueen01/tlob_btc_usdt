import torch
from torch import Tensor
import torch.nn as nn
from models.bin import BiN
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from typing import Optional
from einops import rearrange, reduce

# This enables type checking on runtime
patch_typeguard()


class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.query = nn.Linear(hidden_dim, output_dim)
        self.key = nn.Linear(hidden_dim, output_dim)
        self.value = nn.Linear(hidden_dim, output_dim)

    @typechecked
    def forward(
        self,
        input: TensorType["batch", "hidden_dim", "seq_len"],  # noqa: F821
    ) -> tuple[
        TensorType["batch", "output_dim", "seq_len"],  # noqa: F821
        TensorType["batch", "output_dim", "seq_len"],  # noqa: F821
        TensorType["batch", "output_dim", "seq_len"],  # noqa: F821
    ]:
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)
        return query, key, value


class TransformLayer(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        self.qkv = ComputeQKV(hidden_dim, hidden_dim * n_heads)
        self.attn = nn.MultiheadAttention(
            hidden_dim * n_heads, n_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(hidden_dim)  # Pre-norm for attention
        self.ln2 = nn.LayerNorm(hidden_dim)  # Pre-norm for MLP
        self.down = nn.Linear(hidden_dim * n_heads, hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * 4, hidden_dim)  # Standard 4x expansion

    @typechecked
    def forward(
        self,
        input: TensorType["batch", "hidden_dim", "seq_len"],  # noqa: F821
    ) -> TensorType["batch", "hidden_dim", "seq_len"]:  # noqa: F821
        # Attention block with pre-norm
        residual = input
        x = self.ln1(input)
        query, key, value = self.qkv(x)
        attn_output, _ = self.attn(query, key, value)
        attn_output = self.down(attn_output)
        x = residual + attn_output

        # MLP block with pre-norm
        residual = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(
        self,
        input: TensorType["batch", "input_dim", "seq_len"],  # noqa: F821
    ) -> TensorType["batch", "output_dim", "seq_len"]:  # noqa: F821
        x = self.hidden_layer(input)
        x = self.activation(x)
        x = self.output_layer(x)
        return x


class TLOB(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_features: int,
        seq_length: int,
        num_residual_layers: int,
        num_heads: int,
        num_classes: int = 3,
        is_sin_emd: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.seq_length = seq_length
        self.num_residual_layers = num_residual_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.is_sin_emd = is_sin_emd

        self.norm_layer = BiN(self.num_features, self.seq_length)
        self.embedding_layer = nn.Linear(num_features, hidden_dim)

        # Alternating feature and temporal transformers
        self.feature_transformers = nn.ModuleList(
            [TransformLayer(hidden_dim, num_heads) for _ in range(num_residual_layers)]
        )
        self.temporal_transformers = nn.ModuleList(
            [TransformLayer(seq_length, num_heads) for _ in range(num_residual_layers)]
        )

        self.reduction_layers = self._build_reduction_layers()

        final_dim = self._get_final_dim()
        self.decision_head = nn.Linear(final_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def _build_reduction_layers(self) -> nn.Sequential:
        """Build progressive dimensionality reduction layers"""
        layers = []
        current_dim = self.hidden_dim * self.seq_length

        while current_dim > 128:
            next_dim = current_dim // 4
            layers.extend(
                [nn.Linear(current_dim, next_dim), nn.GELU(), nn.LayerNorm(next_dim)]
            )
            current_dim = next_dim

        return nn.Sequential(*layers)

    def _get_final_dim(self) -> int:
        """Calculate the final dimension after reduction layers"""
        current_dim = self.hidden_dim * self.seq_length
        while current_dim > 128:
            current_dim //= 4
        return current_dim

    def forward(
        self,
        input: TensorType["batch", "num_features", "seq_len"],  # noqa: F821
        target: Optional[TensorType["batch", "seq_len"]] = None,  # noqa: F821
    ) -> tuple[Tensor, Optional[Tensor]]:
        x = self.norm_layer(input)  # [b, f, t]
        x = self.embedding_layer(x)  # [b, h, t]

        for feat_layer, temp_layer in zip(
            self.feature_transformers, self.temporal_transformers
        ):
            x = feat_layer(x)  # [b, h, t]

            # Temporal transformer expects [b, t, h]
            x = rearrange(x, "b h t -> b t h")
            x = temp_layer(x)  # [b, t, h]
            x = rearrange(x, "b t h -> b h t")

        x = rearrange(x, "b h t -> b (h t)")
        x = self.reduction_layers(x)
        logits = self.decision_head(x)

        if target is None:
            return logits, None

        loss = self.loss_fn(logits, target)
        return logits, loss
