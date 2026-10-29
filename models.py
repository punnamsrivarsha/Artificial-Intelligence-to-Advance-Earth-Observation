import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseSegmentationModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def get_num_parameters(self) -> int:
        return sum(p.num_elts() for p in self.parameters())

    def get_flops(self) -> int:
        # Placeholder for FLOPs calculation logic
        return 0

class UNetModel(BaseSegmentationModel):
    def __init__(self, encoder_depth=5, base_channels=64):
        super().__init__()
        self.encoder_depth = encoder_depth
        self.base_channels = base_channels
        # In a real scenario, initialize SMP UNet here
        self.layers = nn.Identity() 

    def forward(self, x):
        return self.layers(x)

class SwinTransformerModel(BaseSegmentationModel):
    def __init__(self, variant='tiny', img_size=224):
        super().__init__()
        self.variant = variant
        self.img_size = img_size

    def forward(self, x):
        return x

class DINOv2Model(BaseSegmentationModel):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Identity() # Frozen DINOv2
        self.seg_head = nn.Sequential(nn.Conv2d(3, 3, 1))

    def forward(self, x):
        return self.seg_head(self.backbone(x))