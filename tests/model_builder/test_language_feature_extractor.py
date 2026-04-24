import pytest
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizer

from src.config.constants import ModelName
from src.model_builder.language_feature_extractor import LanguageFeatureExtractor


@pytest.fixture
def model_name() -> str:
    """
    Provides the identifier for the pretrained language model backbone.

    :return: The model name or path (e.g., 'distilbert-base-uncased').
    :rtype: str
    """
    return ModelName.DISTILBERT_BASE


@pytest.fixture
def projection_dimension() -> int:
    """
    Provides the target output dimension for the language projection head.

    :return: The size of the projected latent space.
    :rtype: int
    """
    return 512


@pytest.fixture
def is_trainable() -> bool:
    """
    Determines if the language backbone should be frozen or trainable.

    :return: True if the backbone parameters should update, False otherwise.
    :rtype: bool
    """
    return False


@pytest.fixture
def language_fe(
    model_name: str,
    projection_dimension: int,
    is_trainable: bool
) -> LanguageFeatureExtractor:
    """
    Initializes a LanguageFeatureExtractor instance for testing.

    :param model_name: The backbone identifier (from fixture).
    :type model_name: str
    :param projection_dimension: The output latent dimension (from fixture).
    :type projection_dimension: int
    :param is_trainable: Freeze/unfreeze toggle (from fixture).
    :type is_trainable: bool

    :return: An instance of the LanguageFeatureExtractor.
    :rtype: LanguageFeatureExtractor
    """
    return LanguageFeatureExtractor(
        backbone_model_name=model_name,
        projection_dimension=projection_dimension,
        backbone_trainable=is_trainable
    )


@pytest.fixture
def batch_size() -> int:
    """
    Provides a standard batch size for NLP testing.

    :return: The number of sequences in a batch.
    :rtype: int
    """
    return 8


@pytest.fixture
def tokenizer(
    model_name: str
) -> PreTrainedTokenizer:
    """
    Loads the appropriate tokenizer for the selected language model.

    :param model_name: The model identifier (from fixture).
    :type model_name: str

    :return: A pretrained Hugging Face tokenizer.
    :rtype: PreTrainedTokenizer
    """
    return AutoTokenizer.from_pretrained(model_name)


@pytest.fixture
def input_tokens(
    batch_size: int,
    tokenizer: PreTrainedTokenizer
) -> BatchEncoding:
    """
    Generates a batch of tokenized sequences, including edge cases like empty strings.
    
    This fixture ensures uniform tensor shapes by applying padding and truncation,
    preventing common PyTorch dimensionality errors.

    :param batch_size: The number of sequences to generate (from fixture).
    :type batch_size: int
    :param tokenizer: The tokenizer to process text (from fixture).
    :type tokenizer: PreTrainedTokenizer

    :return: A dictionary-like object containing input_ids, attention_mask, etc.
    :rtype: BatchEncoding
    """
    sentences = ['Hi this is a test sentence'] * batch_size
    sentences[0] = ''  # Test edge case for empty input
    return tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )


def test_language_fe_freezing(
    is_trainable: int,
    language_fe: LanguageFeatureExtractor,
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
    :param language_fe: The initialized model wrapper under test.
    :type language_fe: LanguageFeatureExtractor

    :return: None
    :rtype: None
    """
    
    # 1. Verify Backbone is frozen
    for name, param in language_fe.backbone.named_parameters():
        assert param.requires_grad is is_trainable, f"Backbone parameter {name} should be frozen!"

    # 2. Verify Projector is still trainable (active)
    for name, param in language_fe.projection_head.named_parameters():
        assert param.requires_grad is True, f"Projection head parameter {name} must stay trainable!"


def test_language_fe_forward_pass(
    batch_size: int,
    projection_dimension: int,
    input_tokens: BatchEncoding,
    language_fe: LanguageFeatureExtractor
) -> None:
    """
    Verifies that the full LanguageFeatureExtractor can process a batch of images.

    Check: Whether a dummy image tensor can pass through the backbone and projector.
    Why: It ensures that the output dimension of the timm 
        backbone exactly matches what the ProjectHead expects.

    Check: Whether the output tensor is differentiable (connected to gradients).
    Why: If `requires_grad` is False on the output, backpropagation 
        cannot occur, and the model will never learn.

    :param batch_size: Number of images in the test batch.
    :type batch_size: int
    :param input_tokens: Dummy Input Tokens (e.g., 512).
    :type input_tokens: BatchEncoding
    :param projection_dimension: Projection Dimension
    :type projection_dimension: int
    :param language_fe: The initialized LanguageFeatureExtractor.
    :type language_fe: LanguageFeatureExtractor

    :return: None
    :rtype: None
    """
    language_features = language_fe(input_tokens)
    
    assert language_features.shape == (batch_size, projection_dimension), f'Output Shape: {language_features.shape} | Expected Shape: {(batch_size, projection_dimension)}'
    assert language_features.requires_grad is True
