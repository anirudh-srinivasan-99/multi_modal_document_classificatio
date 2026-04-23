import albumentations as A
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.data_processor.multi_modal_dataset import MultiModalDataset


@pytest.fixture(scope="session")
def tokenizer() -> PreTrainedTokenizer:
    """
    Provides a real tokenizer instance. 
    Session scope prevents reloading the model for every test, saving time.

    :return: Tokenizer
    :rtype: PreTrainedTokenizer
    """
    return AutoTokenizer.from_pretrained('bert-base-uncased')


@pytest.fixture
def mock_subset() -> list[dict[str, Image.Image | str | int]]:
    """
    Creates a mock list of tuples simulating the 'raw' data format.
    Ensures we aren't reliant on external Hugging Face downloads for unit tests.

    :return: Returns the Raw Data as a List of Image, Text and Label.
    :rtype: list[dict[str, Image.Image | str | int]]
    """
    # Creates Random Pixel Value in range 0-255, with random dimensions 50x50x3
    images = [
        Image.fromarray(np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8))
        for _ in range(3)
    ]
    # Random Texts
    texts = ['Sample invoice text', 'Receipt from grocery', 'Shipping manifest']
    # Labels
    label_ids = [0, 1, 3]
    subset = []
    for image, text, label_id in zip(images, texts, label_ids):
        subset.append(
            {
                'image': image,
                'text': text,
                'label_id': label_id
            }
        )
    return subset


@pytest.fixture
def transformations() -> A.Compose:
    """
    Provides a minimal Albumentations pipeline.
    Must include ToTensorV2 to convert NumPy arrays to PyTorch tensors.

    :return: Compose Stack to transform the Image.
    :rtype: A.Compose
    """
    return A.Compose([
        A.Resize(10, 10),
        A.ToTensorV2()
    ])


@pytest.fixture
def multimodal_dataset(
    mock_subset: list[tuple[Image.Image, str, int]], 
    tokenizer: PreTrainedTokenizer, 
    transformations: A.Compose,
) -> MultiModalDataset:
    """
    The main Dataset instance under test. 
    Injects all required dependencies via other fixtures.

    :param mock_subset: Mock Subset of data.
    :type mock_subset: list[tuple[Image.Image, str, int]]
    :param tokenizer: Tokenizer.
    :type tokenizer: PreTrainedTokenizer
    :param transformations: Transformations Stack.
    :type transformations: A.Compose

    :return: Returns the Dataset Object.
    :rtype: MultiModalDataset
    """
    return MultiModalDataset(
        subset=mock_subset,
        tokenizer=tokenizer,
        image_transformations=transformations,
        max_seq_length=16
    )


def test_dataset_length(multimodal_dataset: MultiModalDataset) -> None:
    """
    Check: Does __getitem__ return the correct triple?
    Why: To ensure the contract between Dataset and DataLoader isn't broken.

    :param multimodal_dataset: Fixture having mock multi-modal dataset.
    :type multimodal_dataset: MultiModalDataset

    :return: None
    :rtype: None
    """
    # Expect 3 here as we have initialized with a subset of size 3.
    assert len(multimodal_dataset) == 3


def test_dataset_item_types(multimodal_dataset: MultiModalDataset) -> None:
    """
    Check: Does __getitem__ return the correct triple?
    Why: To ensure the contract between Dataset and DataLoader isn't broken.
    
    Check: Is the image (C, H, W)?
    Why: PyTorch models expect (Channels, Height, Width), whereas 
    Albumentations/OpenCV defaults to (H, W, C).
    
    Check: Presence of 'input_ids' and 'attention_mask' and their sequence being max_length.
    Why: The model's forward pass will fail if the dictionary keys are wrong 
    or if the sequence length doesn't match the expected max_length.

    Check: Is label a LongTensor?
    Why: CrossEntropyLoss in PyTorch requires labels to be 'torch.long' (int64).
    
    :param multimodal_dataset: Fixture having mock multi-modal dataset.
    :type multimodal_dataset: MultiModalDataset

    :return: None
    :rtype: None
    """
   
    for image, encoded_text, label in multimodal_dataset:
        # Image checks
        assert torch.is_tensor(image)
        assert image.shape == (3, 10, 10)
        
        # Text checks
        assert 'input_ids' in encoded_text
        assert 'attention_mask' in encoded_text
        assert encoded_text['input_ids'].shape == (16,)
        assert encoded_text['attention_mask'].shape == (16,)
        
        # Label checks
        assert isinstance(label, torch.Tensor)
        assert label.dtype == torch.long
