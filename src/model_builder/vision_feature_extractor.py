import timm
import torch
import torch.nn as nn

from src.model_builder.projection_head import ProjectionHead


class VisionFeatureExtractor(nn.Module):
    def __init__(
        self, 
        backbone_model_name: str,
        projection_dimension: int,
        backbone_trainable: bool
    ) -> None:
        """
        Initializes the Vision Feature Extractor with different backbones.

        :param backbone_model_name: Model to be used for Image Feature Extractor.
            Refer to src.config.constants.ModelName to get the list of models being
            used.
        :type backbone_model_name: str
        :param projection_dimension: The output dimension of the final projection head.
        :type projection_dimension: int
        :param backbone_trainable: Whether the backbone parameters should be updated during training.
        :type backbone_trainable: bool

        :return: None
        :rtype: None
        """
        super().__init__()
        self.backbone: nn.Module = timm.create_model(
            model_name=backbone_model_name,
            pretrained=True,
            num_classes=0
        )
        self.input_size: tuple[int, int] = self.backbone.default_cfg['input_size'][1::]

        # The output dimensions are not preserved consistently and thus the best way to
        #   guage would be to do a forward pass.
        backbone_output_dimensions: int = self.__get_backbone_dimension()

        self.backbone_trainable: bool = backbone_trainable

        for param in self.backbone.parameters():
            param.requires_grad = self.backbone_trainable

        self.projection_head: ProjectionHead = ProjectionHead(
            input_dimensions=backbone_output_dimensions,
            output_dimensions=projection_dimension
        )
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the backbone and the projection head.

        :param image: The input image tensor.
        :type image: torch.Tensor

        :return: The projected feature vector.
        :rtype: torch.Tensor
        """
        vision_features = self.backbone(image)
        return self.projection_head(vision_features)
    
    def train(self, mode: bool = True) -> None:
        """
        Sets the module in training mode, with special handling for frozen backbones.

        :param mode: Whether to set the module to training mode (True) or evaluation mode (False).
        :type mode: bool
        :return: None
        :rtype: None
        """
        super().train(mode)
        # Even if we set gradients to be false, BN layers have
        #   non-gradient distribution parameters that would change.
        #   To completely freeze the model, we need to set the
        #   BatchNorm layers to eval() mode.
        #   Thus we are overriding the train method.

        if not self.backbone_trainable:
            for m in self.backbone.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
    
    def __get_backbone_dimension(self) -> int:
        """
        Runs a dummy forward pass to determine the output dimension of the backbone.

        :return: The size of the feature dimension.
        :rtype: int
        """
        h, w = self.input_size
        self.backbone.eval()
        with torch.no_grad():
            dummy_zeros = torch.zeros(1, 3, h, w)
            dummy_output = self.backbone(dummy_zeros)
        dim = dummy_output.shape[1]

        return dim
