import torch.nn as nn
import torch


class ProjectionHead(nn.Module):
    def __init__(
        self, input_dimensions: int,
        output_dimensions: int
    ) -> None:
        """
        Initializes the ProjectionHead with specified input and output dimensions.

        :param input_dimensions: The number of features in the input tensor (from the backbone).
        :type input_dimensions: int
        :param output_dimensions: The desired number of features in the output latent space.
        :type output_dimensions: int

        :return: None
        :rtype: None
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(
                in_features=input_dimensions,
                out_features=output_dimensions,
                bias=False
            ),
            nn.BatchNorm1d(output_dimensions),
            nn.ReLU(),
            nn.Dropout(0.25),
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the projection layers.

        :param features: Features extracted by the respective Extractors.
        :type features: torch.Tensor

        :return: The projected latent representation.
        :rtype: torch.Tensor
        """
        return self.projection(features)
