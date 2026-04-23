import timm
import torch
import torch.nn as nn

from src.model_builder.projection_head import ProjectHead


class VisionBackbone(nn.Module):
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

        for param in self.backbone.parameters():
            param.requires_grad = backbone_trainable

        self.projection_head: ProjectHead = ProjectHead(
            input_dimensions=backbone_output_dimensions,
            output_dimensions=projection_dimension
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        return self.projection_head(out)
    
    def __get_backbone_dimension(self) -> int:
        h, w = self.input_size
        self.backbone.eval()
        with torch.no_grad():
            dummy_zeros = torch.zeros(1, 3, h, w)
            dummy_output = self.backbone(dummy_zeros)
        dim = dummy_output.shape[1]
        self.backbone.train()
        return dim
