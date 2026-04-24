import pytest
import torch

from src.config.constants import ModelName
from src.model_builder.vision_feature_extractor import VisionFeatureExtractor


@pytest.fixture
def model_name() -> str:
    """
    Provides the identifier for the timm vision backbone.

    :return: The model name (e.g., 'mobilenetv2_100').
    :rtype: str
    """
    return ModelName.MOBILE_NET_V2


@pytest.fixture
def projection_dimension() -> int:
    """
    Provides the target output dimension for the vision projection head.

    :return: The size of the projected latent space.
    :rtype: int
    """
    return 512


@pytest.fixture
def is_trainable() -> bool:
    """
    Determines if the vision backbone parameters should be frozen.

    :return: True if the backbone should be trainable, False otherwise.
    :rtype: bool
    """
    return False


@pytest.fixture
def vision_fe(
    model_name: str,
    projection_dimension: int,
    is_trainable: bool
) -> VisionFeatureExtractor:
    """
    Initializes a VisionFeatureExtractor instance for testing.

    :param model_name: The backbone identifier (from fixture).
    :type model_name: str
    :param projection_dimension: The output latent dimension (from fixture).
    :type projection_dimension: int
    :param is_trainable: Freeze/unfreeze toggle (from fixture).
    :type is_trainable: bool
    :return: An instance of the VisionFeatureExtractor.
    :rtype: VisionFeatureExtractor
    """
    return VisionFeatureExtractor(
        backbone_model_name=model_name,
        projection_dimension=projection_dimension,
        backbone_trainable=is_trainable
    )


@pytest.fixture
def batch_size() -> int:
    """
    Provides a standard batch size for vision testing.

    :return: The number of images in a batch.
    :rtype: int
    """
    return 8


@pytest.fixture
def input_tensor(
    batch_size: int,
    vision_fe: VisionFeatureExtractor
) -> torch.Tensor:
    """
    Generates a random image tensor scaled to the backbone's expected resolution.

    :param batch_size: The number of images to generate (from fixture).
    :type batch_size: int
    :param vision_fe: The extractor instance used to determine input size (from fixture).
    :type vision_fe: VisionFeatureExtractor
    :return: A tensor of shape (batch_size, 3, height, width).
    :rtype: torch.Tensor
    """
    h, w = vision_fe.input_size
    return torch.randn(batch_size, 3, h, w)


def test_vision_fe_freezing(
    is_trainable: int,
    vision_fe: VisionFeatureExtractor,
):
    """
    Verifies that the fe freezing logic correctly toggles parameter updates.

    Check: Whether the Projection Head remains trainable regardless of backbone state.
    Check: Whether backbone parameters respect the ``is_trainable`` flag.
    Why: In transfer learning, we often treat the backbone as a fixed feature extractor. 
        If the freezing logic fails, we risk 'catastrophic forgetting' where the model 
        loses its general visual knowledge while trying to adapt to a small dataset.

    :param is_trainable: The expected gradient status for the backbone.
    :type is_trainable: bool
    :param vision_fe: The initialized model wrapper under test.
    :type vision_fe: VisionFeatureExtractor
    """
    
    # 1. Verify Backbone is frozen
    for name, param in vision_fe.backbone.named_parameters():
        assert param.requires_grad is is_trainable, f"Backbone parameter {name} should be frozen!"

    # 2. Verify Projector is still trainable (active)
    for name, param in vision_fe.projection_head.named_parameters():
        assert param.requires_grad is True, f"Projection head parameter {name} must stay trainable!"


def test_vision_fe_forward_pass(
    batch_size: int,
    projection_dimension: int,
    input_tensor: torch.Tensor,
    vision_fe: VisionFeatureExtractor
) -> None:
    """
    Verifies that the full VisionFeatureExtractor can process a batch of images.

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
    :param vision_fe: The initialized VisionFeatureExtractor.
    :type vision_fe: VisionFeatureExtractor
    """
    vision_features = vision_fe(input_tensor)
    
    assert vision_features.shape == (batch_size, projection_dimension), f'Output Shape: {vision_features.shape} | Expected Shape: {(batch_size, projection_dimension)}'
    assert vision_features.requires_grad is True
