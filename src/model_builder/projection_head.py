import torch.nn as nn
import torch


class ProjectHead(nn.Module):
    def __init__(
        self, input_dimensions: int,
        output_dimensions: int
    ) -> None:
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)
