import pytest
import torch

from src.model_builder.projection_head import ProjectionHead


@pytest.fixture
def batch_size() -> int:
    """
    Provides a standard batch size for testing.

    :return: The number of samples in a batch.
    :rtype: int
    """
    return 16


@pytest.fixture
def feature_dimension() -> int:
    """
    Provides the mock output dimension of a vision backbone.

    :return: The number of input features.
    :rtype: int
    """
    return 2186


@pytest.fixture
def input_tensor(
    batch_size: int,
    feature_dimension: int
) -> torch.Tensor:
    """
    Generates a random input tensor simulating backbone features.

    :param batch_size: The number of samples (from fixture).
    :type batch_size: int
    :param feature_dimension: The feature size (from fixture).
    :type feature_dimension: int

    :return: A tensor of shape (batch_size, feature_dimension).
    :rtype: torch.Tensor
    """
    return torch.randn(batch_size, feature_dimension)


@pytest.fixture
def projection_dimension() -> int:
    """
    Provides the target output dimension for the projection head.

    :return: The size of the projected latent space.
    :rtype: int
    """
    return 512


@pytest.fixture
def projection_head(
    feature_dimension: int,
    projection_dimension: int
) -> ProjectionHead:
    """
    Initializes a ProjectionHead instance for testing.

    :param feature_dimension: The input feature size (from fixture).
    :type feature_dimension: int
    :param projection_dimension: The output feature size (from fixture).
    :type projection_dimension: int

    :return: An instance of the ProjectionHead module.
    :rtype: ProjectionHead
    """
    return ProjectionHead(
        input_dimensions=feature_dimension,
        output_dimensions=projection_dimension
    )


def test_projection_head(
    batch_size: int,
    input_tensor: int,
    projection_dimension: int,
    projection_head: ProjectionHead
):
    """
    Verifies the functional integrity of the ProjectHead adapter.

    Check: Output tensor dimensionality after transformation.
    Why: The ProjectHead is our 'plug-and-play' bridge. If it 
        doesn't output the exact `projection_dimension`, the subsequent fusion 
        layers will crash due to shape mismatches.
    Check: Presence of non-negative values in the output (ReLU verification).
    Why: We need to ensure the activation layer is active. 
        If negative values pass through, it suggests the ReLU layer is missing 
        or bypassed, which would change the non-linear learning characteristics 
        of our multimodal system.

    :param batch_size: Number of samples in the synthetic input batch.
    :type batch_size: int
    :param feature_dimension: The raw output size from the vision backbone (e.g., 2152).
    :type feature_dimension: int
    :param input_tensor: Synthetic feature vector representing backbone output.
    :type input_tensor: torch.Tensor
    :param projection_dimension: The target size for our 'fusion space' (e.g., 512).
    :type projection_dimension: int
    :param projection_head: An instance of the ProjectHead being tested.
    :type projection_head: ProjectionHead
    """
    # Forward pass
    projection = projection_head(input_tensor)
    
    # --- Assertions ---
    # 1. Check Output Shape
    assert projection.shape == (batch_size, projection_dimension)
    
    # 2. Check ReLU Activation (No value should be less than 0)
    assert torch.all(projection >= 0), "ReLU failed: Found negative values in output"
