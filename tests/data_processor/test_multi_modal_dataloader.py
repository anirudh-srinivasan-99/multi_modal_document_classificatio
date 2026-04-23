import pytest
import torch
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.data_processor.multi_modal_dataloader import MultiModalDataLoader
from src.config.env_loader import HF_CONFIG


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
def datamodule(tokenizer: PreTrainedTokenizer) -> MultiModalDataLoader:
    """
    Initializes the DataModule pointing to your actual HF repo.

    :param tokenizer: Tokenizer
    :type tokenizer: PreTrainedTokenizer
    
    :return: DataLoader Object
    :rtype: MultiModalDataLoader
    """
    return MultiModalDataLoader(
        hf_repo_id=HF_CONFIG.HF_REPO_ID,
        batch_size=4,
        image_size=(224, 224),
        max_seq_length=128,
        tokenizer=tokenizer,
        dataset_mean=(0.485, 0.456, 0.406),
        dataset_std=(0.229, 0.224, 0.225)
    )


def test_datamodule_splits_streaming(datamodule: MultiModalDataLoader) -> None:
    """
    Check 1: Verify splits exist and contain data.
    Why: To ensure that the data gets loaded properly.
    Check 2: Ensures that the split has the expected number of samples.
    Why: To ensure that th entire dataset gets loaded.

    :param datamodule: DataLoader Object
    :type datamodule: MultiModalDataLoader

    :return: None
    :rtype: None
    """
    # We use stage=None to trigger loading in your match logic
    datamodule.setup(stage="fit")
    
    assert datamodule.train_data is not None
    assert datamodule.val_data is not None
    
    # Verify we can actually get the length (if not streaming)
    # If the dataset is large, just check if the objects are initialized
    assert len(datamodule.train_data) == 2186
    assert len(datamodule.val_data) == 272

    datamodule.setup(stage='test')
    assert datamodule.test_data is not None
    assert len(datamodule.test_data) == 279


def test_dataloader_batch_integrity(datamodule):
    """
    Check 1: Try batching and ensure the correct data sizes are loaded.
    Why: To ensure the pipeline works as intended

    :param datamodule: DataLoader Object
    :type datamodule: MultiModalDataLoader

    :return: None
    :rtype: None
    """
    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()
    
    # Grab exactly one batch
    batch = next(iter(train_loader))
    images, text_enc, labels = batch

    # Verification
    # images: [batch_size, channels, h, w]
    assert images.shape == (4, 3, 224, 224)
    
    # text: [batch_size, max_seq_length]
    assert text_enc["input_ids"].shape == (4, 128)
    assert text_enc["attention_mask"].shape == (4, 128)
    
    # labels: [batch_size]
    assert labels.shape == (4,)
    assert labels.dtype == torch.long
