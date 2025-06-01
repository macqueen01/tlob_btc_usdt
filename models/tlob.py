from cycler import V
import torch
from torch import Tensor
import torch.nn as nn
from models.bin import BiN
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

# This enables type checking on runtime
patch_typeguard()


class TLOB(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_features: int,
        seq_length: int,
        num_layers: int,
        num_heads: int,
        is_sin_emd: bool = False,
    ) -> None:
        super().__init__()

        self.num_features = num_features
        self.seq_length = seq_length
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.is_sin_emd = is_sin_emd
        self.norm_layer = BiN(self.num_features, self.seq_length)


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
        self.ln = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim * n_heads, hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim, hidden_dim)

    @typechecked
    def forward(
        self,
        input: TensorType["batch", "hidden_dim", "seq_len"],  # noqa: F821
    ) -> TensorType["batch", "hidden_dim", "seq_len"]:  # noqa: F821
        res = input  # [batch, hidden_dim, seq_len]
        query, key, value = self.qkv(
            input
        )  # tuple of 3 x [batch, hidden_dim * n_heads, seq_len]
        attn_output, _ = self.attn(
            query, key, value
        )  # attn_output: [batch, hidden_dim * n_heads, seq_len]
        attn_output = self.down(attn_output)  # [batch, hidden_dim, seq_len]
        res = res + attn_output  # [batch, hidden_dim, seq_len]
        res = self.ln(res)  # [batch, hidden_dim, seq_len]
        res = self.mlp(res)  # [batch, hidden_dim, seq_len]
        res = res + input  # [batch, hidden_dim, seq_len]
        return res


class MLP(nn.Module):
    def __init__(self, start_dim: int, hidden_dim: int, final_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(start_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, final_dim)
        self.activation = nn.GELU()
        self.ln = nn.LayerNorm(final_dim)

    def forward(
        self,
        input: TensorType["batch", "start_dim", "seq_len"],  # noqa: F821
    ) -> TensorType["batch", "final_dim", "seq_len"]:  # noqa: F821
        res = input
        input = self.hidden_layer(input)
        input = self.activation(input)
        input = self.output_layer(input)
        res = res + input
        res = self.ln(res)
        res = self.activation(res)
        return res
