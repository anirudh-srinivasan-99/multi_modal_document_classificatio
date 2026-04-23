import pytest
import torch

from src.model_builder.projection_head import ProjectHead


@pytest.fixture
def batch_size() -> int:
    return 16


@pytest.fixture
def feature_dimension() -> int:
    return 2186


@pytest.fixture
def input_tensor(
    batch_size: int,
    feature_dimension: int
) -> torch.Tensor:
    return torch.randn(batch_size, feature_dimension)


@pytest.fixture
def projection_dimension() -> int:
    return 512


@pytest.fixture
def projection_head(
    feature_dimension: int,
    projection_dimension: int
) -> ProjectHead:
    return ProjectHead(
        input_dimensions=feature_dimension,
        output_dimensions=projection_dimension
    )


def test_projection_head(
    batch_size: int,
    input_tensor: int,
    projection_dimension: int,
    projection_head: ProjectHead
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
