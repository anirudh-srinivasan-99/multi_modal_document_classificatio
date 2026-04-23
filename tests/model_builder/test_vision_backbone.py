import pytest
import torch

from src.config.constants import ModelName
from src.model_builder.vision_backbone import VisionBackbone


@pytest.fixture
def model_name() -> str:
    return ModelName.MOBILE_NET_V2


@pytest.fixture
def projection_dimension() -> int:
    return 512


@pytest.fixture
def is_trainable() -> bool:
    return False

@pytest.fixture
def vision_backbone(
    model_name: str,
    projection_dimension: int,
    is_trainable: bool
) -> VisionBackbone:
    return VisionBackbone(
        backbone_model_name=model_name,
        projection_dimension=projection_dimension,
        backbone_trainable=is_trainable
    )

@pytest.fixture
def batch_size() -> int:
    return 8

@pytest.fixture
def input_tensor(
    batch_size: int,
    vision_backbone: VisionBackbone
) -> torch.Tensor:
    h, w = vision_backbone.input_size
    return torch.randn(batch_size, 3, h, w)


def test_vision_backbone_freezing(
    is_trainable: int,
    vision_backbone: VisionBackbone,
):
    """
    Verifies that the backbone freezing logic correctly toggles parameter updates.

    Check: Whether the Projection Head remains trainable regardless of backbone state.
    Check: Whether backbone parameters respect the ``is_trainable`` flag.
    Why: In transfer learning, we often treat the backbone as a fixed feature extractor. 
        If the freezing logic fails, we risk 'catastrophic forgetting' where the model 
        loses its general visual knowledge while trying to adapt to a small dataset.

    :param is_trainable: The expected gradient status for the backbone.
    :type is_trainable: bool
    :param vision_backbone: The initialized model wrapper under test.
    :type vision_backbone: VisionBackbone
    """
    
    # 1. Verify Backbone is frozen
    for name, param in vision_backbone.backbone.named_parameters():
        assert param.requires_grad is is_trainable, f"Backbone parameter {name} should be frozen!"

    # 2. Verify Projector is still trainable (active)
    for name, param in vision_backbone.projection_head.named_parameters():
        assert param.requires_grad is True, f"Projection head parameter {name} must stay trainable!"


def test_vision_backbone_forward_pass(
    batch_size: int,
    projection_dimension: int,
    input_tensor: torch.Tensor,
    vision_backbone: VisionBackbone # Assuming our wrapped class
):
    """
    Verifies that the full VisionBackbone can process a batch of images.

    Check: Whether a dummy image tensor can pass through the backbone and projector.
    Why: It ensures that the output dimension of the timm 
        backbone exactly matches what the ProjectHead expects.

    Check: Whether the output tensor is differentiable (connected to gradients).
    Why: If `requires_grad` is False on the output, backpropagation 
        cannot occur, and the model will never learn.

    :param batch_size: Number of images in the test batch.
    :type batch_size: int
    :param input_tensor: Dummy Input Tensor (e.g., 512).
    :type input_tensor: int
    :param projection_dimension: Projection Dimension
    :type projection_dimension: int
    :param vision_backbone: The initialized VisionBackbone.
    :type vision_backbone: VisionBackbone
    """
    projection = vision_backbone(input_tensor)
    
    assert projection.shape == (batch_size, projection_dimension), f'Output Shape: {projection.shape} | Expected Shape: {(batch_size, projection_dimension)}'
    assert projection.requires_grad is True
