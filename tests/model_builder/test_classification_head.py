import pytest
import torch

from src.model_builder.classification_head import ClassificationHead


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
    Provides the mock output dimension from feature extractors.

    :return: The number of input features.
    :rtype: int
    """
    return 1024


@pytest.fixture
def input_tensor(
    batch_size: int,
    feature_dimension: int
) -> torch.Tensor:
    """
    Generates a random input tensor simulating features extracted.

    :param batch_size: The number of samples (from fixture).
    :type batch_size: int
    :param feature_dimension: The feature size (from fixture).
    :type feature_dimension: int

    :return: A tensor of shape (batch_size, feature_dimension).
    :rtype: torch.Tensor
    """
    return torch.randn(batch_size, feature_dimension)


@pytest.fixture
def num_classes() -> int:
    """
    Provides the number of classes to classify.

    :return: Number of classes.
    :rtype: int
    """
    return 10


@pytest.fixture
def classification_head(
    feature_dimension: int,
    num_classes: int
) -> ClassificationHead:
    """
    Initializes a ClassificationHead instance for testing.

    :param feature_dimension: The input feature size (from fixture).
    :type feature_dimension: int
    :param num_classes: Number of classes (from fixture).
    :type num_classes: int

    :return: An instance of the ClassificationHead module.
    :rtype: ClassificationHead
    """
    return ClassificationHead(
        input_dimensions=feature_dimension,
        num_classes=num_classes
    )


def test_classification_head(
    batch_size: int,
    input_tensor: int,
    num_classes: int,
    classification_head: ClassificationHead
):
    """
    Verifies the functional integrity of the ProjectHead adapter.

    Check: Output tensor dimensionality after transformation.
    Why: The ProjectHead is our 'plug-and-play' bridge. If it 
        doesn't output the exact `projection_dimension`, the subsequent fusion 
        layers will crash due to shape mismatches.

    :param batch_size: Number of samples in the synthetic input batch.
    :type batch_size: int
    :param feature_dimension: The raw output size from the vision backbone (e.g., 2152).
    :type feature_dimension: int
    :param input_tensor: Synthetic feature vector representing backbone output.
    :type input_tensor: torch.Tensor
    :param num_classes: Number of Classes.
    :type num_classes: int
    :param classification_head: An instance of the ProjectHead being tested.
    :type classification_head: ClassificationHead
    """
    # Forward pass
    projection = classification_head(input_tensor)
    
    # --- Assertions ---
    # 1. Check Output Shape
    assert projection.shape == (batch_size, num_classes)
