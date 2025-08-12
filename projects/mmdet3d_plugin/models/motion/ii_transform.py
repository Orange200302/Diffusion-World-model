import torch
import torch.nn as nn
from mmcv.utils import registry
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)

@PLUGIN_LAYERS.register_module()
class FutureTransformerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(128, 256, kernel_size=4, stride=4)  # -> [256, 12, 12]
        self.pos_embed = nn.Parameter(torch.randn(1, 12*12, 256))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=2
        )
        self.fc = nn.Linear(256, 256)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.contiguous().view(B*T, C, H, W)
        x = self.patch_embed(x)  # [B*T, 256, 12, 12]
        x = x.flatten(2).transpose(1, 2).contiguous()  # [B*T, 144, 256]
        x = x + self.pos_embed  # 添加位置编码
        x = self.transformer(x)  # [B*T, 144, 256]
        x = x.mean(dim=1)  # 池化所有 patch
        x = self.fc(x)  # [B*T, 256]
        x = x.view(B, T, 256)
        return x
