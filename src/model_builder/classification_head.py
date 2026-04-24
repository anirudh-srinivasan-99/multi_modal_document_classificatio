import torch.nn as nn
import torch


class ClassificationHead(nn.Module):
    def __init__(
        self, input_dimensions: int,
        num_classes: int
    ) -> None:
        """
        Initializes the ClassificationHead with specified input and output dimensions.

        :param input_dimensions: The number of features in the input tensor (from the backbone).
        :type input_dimensions: int
        :param num_classes: Number of classes to be classified.
        :type num_classes: int

        :return: None
        :rtype: None
        """
        super().__init__()
        self.classification = nn.Sequential(
            nn.Linear(
                in_features=input_dimensions,
                out_features=1024,
                bias=False
            ),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(
                in_features=1024,
                out_features=256,
                bias=False
            ),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(
                in_features=256,
                out_features=num_classes,
                bias=False
            )
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the classification layers.

        :param features: Features extracted by the respective Extractors.
        :type features: torch.Tensor

        :return: The Logits.
        :rtype: torch.Tensor
        """
        return self.classification(features)
