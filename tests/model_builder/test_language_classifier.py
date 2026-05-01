import pytest
import torch
from transformers import BatchEncoding

from src.model_builder.language_classifier import LanguageClassifier
from src.config.constants import ModelName


@pytest.fixture
def language_fe_params() -> tuple[str, int, bool, int]:
    return (ModelName.ROBERTA_BASE, 896, False, 256)


@pytest.fixture
def learning_rate() -> float:
    return 0.01


@pytest.fixture
def num_classes() -> int:
    return 10


@pytest.fixture
def language_classifier(
    language_fe_params: tuple[str, int, bool],
    learning_rate: float,
    num_classes: int
) -> LanguageClassifier:
    lmn, lpd, lbt, lmsl = language_fe_params
    return LanguageClassifier(
        language_model_name=lmn,
        language_projection_dimension=lpd,
        language_backbone_trainable=lbt,
        max_seq_len=lmsl,
        num_classes=num_classes,
        learning_rate=learning_rate
    )


@pytest.fixture
def batch_size() -> int:
    return 8


@pytest.fixture
def batch_data(
    batch_size: int,
    num_classes: int,
    language_classifier: LanguageClassifier
) -> tuple[torch.Tensor, BatchEncoding, torch.Tensor]:
    h, w = (224, 224)
    max_seq_len = language_classifier.language_fe.max_seq_len
    images = torch.randn(batch_size, 3, h, w)
    tokens = BatchEncoding(
        {
            'input_ids': torch.randint(0, 100, (batch_size, max_seq_len)),
            'attention_mask': torch.ones(batch_size, max_seq_len) 
        }
    )
    labels = torch.randint(0, num_classes, (batch_size, ))
    return images, tokens, labels


def test_classifier_forward_pass(
    batch_data: tuple[torch.Tensor, BatchEncoding],
    batch_size: int,
    num_classes: int,
    language_classifier: LanguageClassifier
) -> None:
    images, tokens, _ = batch_data

    logits = language_classifier(images, tokens)
    
    # Asserts if the shape is as expected.
    assert logits.shape == (batch_size, num_classes)

    # Asserts that the output vector is differentiable.
    assert logits.requires_grad is True


def test_training_step_logic(
    batch_data: tuple[torch.Tensor, BatchEncoding],
    language_classifier: LanguageClassifier
) -> None:
    loss = language_classifier.training_step(batch_data, 0)

    # Asserts if the loss is of expected type, differentiable.
    #   and is not NaN.
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad is True
    assert not torch.isnan(loss)
    
    # Run end of epoch to verify reset logic
    language_classifier.on_train_epoch_end()

    # Verify metrics zeroed out
    for metric in language_classifier.train_metrics.values():
        assert torch.all(metric.tp == 0)


def test_parameter_trainability(
    language_fe_params: tuple[str, int, bool],
    language_classifier: LanguageClassifier
) -> None:
    _, _, lbt, _ = language_fe_params
    
    for params in language_classifier.language_fe.backbone.parameters():
        assert params.requires_grad is lbt
    
    for params in language_classifier.language_fe.projection_head.parameters():
        assert params.requires_grad is True
    
    for params in language_classifier.classification_head.parameters():
        assert params.requires_grad is True

def test_validation_lifecycle(
    batch_size: int,
    batch_data: tuple[torch.Tensor, BatchEncoding, torch.Tensor],
    language_classifier: LanguageClassifier
) -> None:
    language_classifier.validation_step(batch_data, 0)

    val_results = language_classifier.val_metrics.compute()
    assert 0.0 <= val_results['val_MulticlassAccuracy'] <= 1.0
    
    language_classifier.on_validation_epoch_end()

    # Verify metrics zeroed out
    for metric in language_classifier.val_metrics.values():
        assert torch.all(metric.tp == 0)
    assert torch.all(language_classifier.val_cm.confmat == 0)


def test_test_lifecycle(
    batch_size: int,
    batch_data: tuple[torch.Tensor, BatchEncoding, torch.Tensor],
    language_classifier: LanguageClassifier
) -> None:
    language_classifier.test_step(batch_data, 0)

    test_results = language_classifier.test_metrics.compute()
    assert 0.0 <= test_results['test_MulticlassAccuracy'] <= 1.0
    
    language_classifier.on_test_epoch_end()

    for metric in language_classifier.train_metrics.values():
        assert torch.all(metric.tp == 0)
    assert torch.all(language_classifier.test_cm.confmat == 0)