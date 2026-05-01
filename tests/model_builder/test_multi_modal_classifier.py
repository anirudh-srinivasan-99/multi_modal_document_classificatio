import pytest
import torch
from transformers import BatchEncoding

from src.model_builder.multi_modal_classifier import MultiModalClassifier
from src.config.constants import ModelName


@pytest.fixture
def vision_fe_params() -> tuple[str, int, bool]:
    return (ModelName.MOBILE_NET_V2, 1024, False)


@pytest.fixture
def language_fe_params() -> tuple[str, int, bool, int]:
    return (ModelName.DISTILBERT_BASE, 896, False, 256)


@pytest.fixture
def learning_rate() -> float:
    return 0.01


@pytest.fixture
def num_classes() -> int:
    return 10


@pytest.fixture
def multi_modal_classifier(
    vision_fe_params: tuple[str, int, bool],
    language_fe_params: tuple[str, int, bool],
    learning_rate: float,
    num_classes: int
) -> MultiModalClassifier:
    vmn, vpd, vbt = vision_fe_params
    lmn, lpd, lbt, lmsl = language_fe_params
    return MultiModalClassifier(
        vision_model_name=vmn,
        vision_projection_dimension=vpd,
        vision_backbone_trainable=vbt,
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
    multi_modal_classifier: MultiModalClassifier
) -> tuple[torch.Tensor, BatchEncoding, torch.Tensor]:
    h, w = multi_modal_classifier.vision_fe.input_size
    max_seq_len = multi_modal_classifier.language_fe.max_seq_len
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
    multi_modal_classifier: MultiModalClassifier
) -> None:
    images, tokens, _ = batch_data

    logits = multi_modal_classifier(images, tokens)
    
    # Asserts if the shape is as expected.
    assert logits.shape == (batch_size, num_classes)

    # Asserts that the output vector is differentiable.
    assert logits.requires_grad is True


def test_training_step_logic(
    batch_data: tuple[torch.Tensor, BatchEncoding],
    multi_modal_classifier: MultiModalClassifier
) -> None:
    loss = multi_modal_classifier.training_step(batch_data, 0)

    # Asserts if the loss is of expected type, differentiable.
    #   and is not NaN.
    assert isinstance(loss, torch.Tensor)
    assert loss.requires_grad is True
    assert not torch.isnan(loss)
    
    # Run end of epoch to verify reset logic
    multi_modal_classifier.on_train_epoch_end()

    # Verify metrics zeroed out
    for metric in multi_modal_classifier.train_metrics.values():
        assert torch.all(metric.tp == 0)


def test_parameter_trainability(
    vision_fe_params: tuple[str, int, bool],
    language_fe_params: tuple[str, int, bool],
    multi_modal_classifier: MultiModalClassifier
) -> None:
    _, _, vbt = vision_fe_params
    _, _, lbt, _ = language_fe_params

    for params in multi_modal_classifier.vision_fe.backbone.parameters():
        assert params.requires_grad is vbt
    
    for params in multi_modal_classifier.vision_fe.projection_head.parameters():
        assert params.requires_grad is True
    
    for params in multi_modal_classifier.language_fe.backbone.parameters():
        assert params.requires_grad is lbt
    
    for params in multi_modal_classifier.language_fe.projection_head.parameters():
        assert params.requires_grad is True
    
    for params in multi_modal_classifier.classification_head.parameters():
        assert params.requires_grad is True

def test_validation_lifecycle(
    batch_size: int,
    batch_data: tuple[torch.Tensor, BatchEncoding, torch.Tensor],
    multi_modal_classifier: MultiModalClassifier
) -> None:
    multi_modal_classifier.validation_step(batch_data, 0)

    val_results = multi_modal_classifier.val_metrics.compute()
    assert 0.0 <= val_results['val_MulticlassAccuracy'] <= 1.0
    
    multi_modal_classifier.on_validation_epoch_end()

    # Verify metrics zeroed out
    for metric in multi_modal_classifier.val_metrics.values():
        assert torch.all(metric.tp == 0)
    assert torch.all(multi_modal_classifier.val_cm.confmat == 0)


def test_test_lifecycle(
    batch_size: int,
    batch_data: tuple[torch.Tensor, BatchEncoding, torch.Tensor],
    multi_modal_classifier: MultiModalClassifier
) -> None:
    multi_modal_classifier.test_step(batch_data, 0)

    test_results = multi_modal_classifier.test_metrics.compute()
    assert 0.0 <= test_results['test_MulticlassAccuracy'] <= 1.0
    
    multi_modal_classifier.on_test_epoch_end()

    for metric in multi_modal_classifier.train_metrics.values():
        assert torch.all(metric.tp == 0)
    assert torch.all(multi_modal_classifier.test_cm.confmat == 0)