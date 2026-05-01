import lightning as L
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MetricCollection
from transformers import BatchEncoding


from src.model_builder.classification_head import ClassificationHead
from src.model_builder.language_feature_extractor import LanguageFeatureExtractor


class LanguageClassifier(L.LightningModule):
    def __init__(
        self,
        language_model_name: str,
        language_projection_dimension: int,
        language_backbone_trainable: bool,
        max_seq_len: int,
        num_classes: int,
        learning_rate: float
    ) -> None:
        """
        Initialize the LanguageClassifier.

        :param language_model_name: Name of the language backbone (e.g., from transformers).
        :type language_model_name: str
        :param language_projection_dimension: Output dimension of the language feature extractor.
        :type language_projection_dimension: int
        :param language_backbone_trainable: Whether to update weights of the language backbone.
        :type language_backbone_trainable: bool
        :param max_seq_len: Maximum number of tokens for a sequence.
        :type max_seq_len: int
        :param num_classes: Number of target classification categories.
        :type num_classes: int
        :param learning_rate: Learning rate for the optimizer.
        :type learning_rate: float

        :return: None
        :rtype: None
        """
        super().__init__()

        self.lr: float = learning_rate

        metrics = MetricCollection([
            Accuracy(task='multiclass', num_classes=num_classes),
            F1Score(task='multiclass', num_classes=num_classes)
        ])
        self.train_metrics: MetricCollection = metrics.clone(prefix='train_')
        self.val_metrics: MetricCollection = metrics.clone(prefix='val_')
        self.test_metrics: MetricCollection = metrics.clone(prefix='test_')

        self.val_cm: ConfusionMatrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)
        self.test_cm: ConfusionMatrix = ConfusionMatrix(task='multiclass', num_classes=num_classes)

        self.language_fe: LanguageFeatureExtractor = LanguageFeatureExtractor(
            backbone_model_name=language_model_name,
            projection_dimension=language_projection_dimension,
            backbone_trainable=language_backbone_trainable,
            max_seq_len=max_seq_len
        )

        self.classification_head = ClassificationHead(
            input_dimensions=(language_projection_dimension),
            num_classes=num_classes
        )

    def forward(self, images: torch.Tensor, tokens: BatchEncoding) -> torch.Tensor:
        """
        Perform a forward pass through the multi-modal network.

        :param images: Batch of input images. Passing Images aswell as MultiModalDataset
            and MultiModalDataloader can be reused.
        :type images: torch.Tensor
        :param tokens: Tokenized language inputs.
        :type tokens: BatchEncoding

        :return: Classification logits.
        :rtype: torch.Tensor
        """
        language_features = self.language_fe(tokens)

        return self.classification_head(language_features)

    def training_step(
        self,
        batch: tuple[torch.Tensor, BatchEncoding, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        """
        Compute and log the training loss and update training metrics.

        :param batch: Tuple containing (images, tokens, labels).
        :type batch: tuple[torch.Tensor, BatchEncoding, torch.Tensor]
        :param batch_idx: Index of the current batch.
        :type batch_idx: int

        :return: The computed cross-entropy loss.
        :rtype: torch.Tensor
        """
        images, tokens, labels = batch
        logits = self(images, tokens)

        loss = F.cross_entropy(logits, labels)
        self.train_metrics.update(logits, labels)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def on_train_epoch_end(self) -> None:
        """
        Compute training metrics at the end of the epoch and reset metric states.

        :return: None
        :rtype: None
        """
        output = self.train_metrics.compute()
        if self.logger:
            self.logger.log_metrics(
                output, step=self.current_epoch
            )
        self.log_dict(
            output, on_epoch=True, on_step=False,
            logger=False, prog_bar=True
        )
        self.train_metrics.reset()

    def validation_step(
        self,
        batch: tuple[torch.Tensor, BatchEncoding, torch.Tensor],
        batch_idx: int
    ) -> None:
        """
        Compute validation loss and update validation metrics and confusion matrix.

        :param batch: Tuple containing (images, tokens, labels).
        :type batch: tuple[torch.Tensor, BatchEncoding, torch.Tensor]
        :param batch_idx: Index of the current batch.
        :type batch_idx: int

        :return: None
        :rtype: None
        """
        images, tokens, labels = batch
        logits = self(images, tokens)

        loss = F.cross_entropy(logits, labels)
        self.val_metrics.update(logits, labels)
        self.val_cm.update(logits, labels)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
    
    def on_validation_epoch_end(self) -> None:
        """
        Log validation metrics and the confusion matrix figure to the logger.

        :return: None
        :rtype: None
        """
        # Log scalar metrics (Acc, F1)
        output = self.val_metrics.compute()

        # Log Confusion Matrix as an Image to MLflow
        fig, _ = self.val_cm.plot()
        if self.logger:
            self.logger.log_metrics(
                output, step=self.current_epoch
            )
            self.logger.experiment.log_figure(
                run_id=self.logger.run_id,
                figure=fig,
                artifact_file=f"val_cm/val_cm_epoch_{self.current_epoch}.png"
            )
        self.log_dict(
            output, on_epoch=True, on_step=False,
            logger=False, prog_bar=True
        )

        plt.close(fig)
        self.val_metrics.reset()
        self.val_cm.reset()
    
    def test_step(
        self,
        batch: tuple[torch.Tensor, BatchEncoding, torch.Tensor],
        batch_idx: int
    ) -> None:
        """
        Compute test loss and update test metrics and confusion matrix.

        :param batch: Tuple containing (images, tokens, labels).
        :type batch: tuple[torch.Tensor, BatchEncoding, torch.Tensor]
        :param batch_idx: Index of the current batch.
        :type batch_idx: int

        :return: None
        :rtype: None
        """
        images, tokens, labels = batch
        logits = self(images, tokens)

        loss = F.cross_entropy(logits, labels)
        self.test_metrics.update(logits, labels)
        self.test_cm.update(logits, labels)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
    
    def on_test_epoch_end(self) -> None:
        """
        Log final test metrics and the test confusion matrix figure.

        :return: None
        :rtype: None
        """
        output = self.test_metrics.compute() # output is a dict of tensors

        # 1. Let Lightning handle the scalars (this puts them in the table & progress bar)
        self.log_dict(output, prog_bar=True)

        # 2. Manually handle the Image (since Lightning's self.log doesn't do figures)
        if self.logger:
            fig, _ = self.test_cm.plot()
            self.logger.experiment.log_figure(
                run_id=self.logger.run_id,
                figure=fig,
                artifact_file=f'test_cm/test_cm_epoch_{self.current_epoch}.png'
            )
            plt.close(fig)

        self.test_cm.reset()
        self.test_metrics.reset()

    def configure_optimizers(self):
        """
        Set up the Adam optimizer and ReduceLROnPlateau scheduler.

        :return: A dictionary containing the optimizer, lr_scheduler, and the metric to monitor.
        :rtype: dict
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,   # Drop by 2x instead of 10x
            patience=5,   # Wait 5 epochs of no improvement
            verbose=True
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'monitor': 'val_loss'
        }
