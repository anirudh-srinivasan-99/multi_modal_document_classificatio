import timm
import torch
import torch.nn as nn

from src.model_builder.projection_head import ProjectHead


class VisionFeatureExtractor(nn.Module):
    def __init__(
        self, 
        backbone_model_name: str,
        projection_dimension: int,
        backbone_trainable: bool
    ) -> None:
        super().__init__()
        self.backbone: nn.Module = timm.create_model(
            model_name=backbone_model_name,
            pretrained=True,
            num_classes=0
        )
        self.input_size: tuple[int, int] = self.backbone.default_cfg['input_size'][1::]
        backbone_output_dimensions: int = self.__get_backbone_dimension()

        self.backbone_trainable: bool = backbone_trainable

        for param in self.backbone.parameters():
            param.requires_grad = self.backbone_trainable

        self.projection_head: ProjectHead = ProjectHead(
            input_dimensions=backbone_output_dimensions,
            output_dimensions=projection_dimension
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.projection_head(out)
    
    def train(self, mode: bool = True) -> None:
        super().train(mode)

        if not self.backbone_trainable:
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
    
    def __get_backbone_dimension(self) -> int:
        h, w = self.input_size
        self.backbone.eval()
        with torch.no_grad():
            dummy_zeros = torch.zeros(1, 3, h, w)
            dummy_output = self.backbone(dummy_zeros)
        dim = dummy_output.shape[1]

        return dim
