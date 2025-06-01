import torch
from torch import nn
import constants as cst
import einops as einops

class BiN(nn.Module):
    """Bidirectional Normalization Layer
    
    Performs normalization in two directions:
    1. Temporal: normalize each feature across time steps
    2. Feature: normalize each time step across features
    """
    
    def __init__(self, num_features, seq_length):
        super().__init__()
        self.seq_length = seq_length
        self.num_features = num_features

        # Feature normalization parameters (applied across features at each timestep)
        feature_bias = torch.Tensor(seq_length, 1)
        self.feature_bias = nn.Parameter(feature_bias)
        nn.init.constant_(self.feature_bias, 0)

        feature_scale = torch.Tensor(seq_length, 1)
        self.feature_scale = nn.Parameter(feature_scale)
        nn.init.xavier_normal_(self.feature_scale)

        # Temporal normalization parameters (applied across time for each feature)
        temporal_bias = torch.Tensor(num_features, 1)
        self.temporal_bias = nn.Parameter(temporal_bias)
        nn.init.constant_(self.temporal_bias, 0)

        temporal_scale = torch.Tensor(num_features, 1)
        self.temporal_scale = nn.Parameter(temporal_scale)
        nn.init.xavier_normal_(self.temporal_scale)

        # Mixing weights for combining both normalization types
        feature_weight = torch.Tensor(1, )
        self.feature_weight = nn.Parameter(feature_weight)
        nn.init.constant_(self.feature_weight, 0.5)

        temporal_weight = torch.Tensor(1, )
        self.temporal_weight = nn.Parameter(temporal_weight)
        nn.init.constant_(self.temporal_weight, 0.5)

    def forward(self, x):
        # Ensure mixing weights stay positive
        if (self.feature_weight[0] < 0):
            feature_weight_new = torch.empty(1, device=self.feature_weight.device, dtype=self.feature_weight.dtype)
            self.feature_weight = nn.Parameter(feature_weight_new)
            nn.init.constant_(self.feature_weight, 0.01)

        if (self.temporal_weight[0] < 0):
            temporal_weight_new = torch.empty(1, device=self.temporal_weight.device, dtype=self.temporal_weight.dtype)
            self.temporal_weight = nn.Parameter(temporal_weight_new)
            nn.init.constant_(self.temporal_weight, 0.01)
            
        B, F, T = x.shape

        temporal_output = self._temporal_norm(x, B, F, T)
        feature_output = self._feature_norm(x, B, F, T)

        output = self.feature_weight * feature_output + self.temporal_weight * temporal_output

        return output

    def _temporal_norm(self, x, B, F, T):
        # Normalize each feature across time dimension (dim=2)
        temporal_mean = torch.mean(x, dim=2, keepdim=True) # (B, F, T) -> (B, F, 1)
        temporal_std = torch.std(x, dim=2, keepdim=True) # (B, F, T) -> (B, F, 1)
        temporal_std = temporal_std.clone()
        # Handle edge cases: if T=1, std will be NaN, so set to 1
        temporal_std[torch.isnan(temporal_std)] = 1
        temporal_std[temporal_std < 1e-4] = 1  # Avoid division by zero

        # Normalize: (x - mean) / std
        temporal_mean_broadcast = einops.repeat(temporal_mean, 'b f 1 -> b f t', t=T)
        temporal_std_broadcast = einops.repeat(temporal_std, 'b f 1 -> b f t', t=T)
        temporal_normalized = (x - temporal_mean_broadcast) / temporal_std_broadcast

        temporal_output = self._apply_temporal_scale_bias(temporal_normalized, B, F, T)
        return temporal_output

    def _feature_norm(self, x, B, F, T):
        # Normalize each timestep across feature dimension (dim=1)
        feature_mean = torch.mean(x, dim=1, keepdim=True) # (B, F, T) -> (B, 1, T)
        feature_std = torch.std(x, dim=1, keepdim=True) # (B, F, T) -> (B, 1, T)
        feature_std = feature_std.clone()
        # Handle edge cases: if F=1, std will be NaN, so set to 1
        feature_std[torch.isnan(feature_std)] = 1
        feature_std[feature_std < 1e-4] = 1  # Avoid division by zero

        feature_mean_broadcast = einops.repeat(feature_mean, 'b 1 t -> b f t', f=F)
        feature_std_broadcast = einops.repeat(feature_std, 'b 1 t -> b f t', f=F)
        feature_normalized = (x - feature_mean_broadcast) / feature_std_broadcast

        feature_output = self._apply_feature_scale_bias(feature_normalized, B, F, T)
        return feature_output

    def _apply_temporal_scale_bias(self, normalized, B, F, T):
        temporal_scale_broadcast = einops.repeat(self.temporal_scale, 'f 1 -> b f t', b=B, t=T)
        temporal_bias_broadcast = einops.repeat(self.temporal_bias, 'f 1 -> b f t', b=B, t=T)
        return temporal_scale_broadcast * normalized + temporal_bias_broadcast

    def _apply_feature_scale_bias(self, normalized, B, F, T):
        feature_scale_broadcast = einops.repeat(self.feature_scale, 't 1 -> b f t', b=B, f=F)
        feature_bias_broadcast = einops.repeat(self.feature_bias, 't 1 -> b f t', b=B, f=F)
        return feature_scale_broadcast * normalized + feature_bias_broadcast