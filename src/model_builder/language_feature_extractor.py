import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, BatchEncoding

from src.model_builder.projection_head import ProjectionHead


class LanguageFeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_model_name: str,
        projection_dimension: int,
        backbone_trainable: bool
    ) -> None:
        """
        Initializes the Language Feature Extractor with different backbones.

        :param backbone_model_name: Model to be used for Image Feature Extractor.
            Refer to src.config.constants.ModelName to get the list of models being
            used.
        :type backbone_model_name: str
        :param projection_dimension: The output dimension of the final projection head.
        :type projection_dimension: int
        :param backbone_trainable: Whether the backbone parameters should be updated during training.
        :type backbone_trainable: bool

        :return: None
        :rtype: None
        """
        super().__init__()
        self.backbone: nn.Module = AutoModel.from_pretrained(backbone_model_name)
        config = AutoConfig.from_pretrained(backbone_model_name)

        self.max_seq_len = config.max_position_embeddings

        # There is a slight inconsistency with models where some store the output_dimensions
        #   in `hidden_size` and others in `dim`. Thus we check both.
        backbone_output_dimensions = getattr(config, 'hidden_size', getattr(config, 'dim', None))

        for param in self.backbone.parameters():
            param.requires_grad = backbone_trainable

        self.projection_head = ProjectionHead(
            input_dimensions=backbone_output_dimensions,
            output_dimensions=projection_dimension
        )
    
    def forward(self, tokens: BatchEncoding) -> torch.Tensor:
        """
        Performs a forward pass through the backbone and the projection head.

        :param tokens: The input image tensor.
        :type tokens: BatchEncoding

        :return: The projected feature vector.
        :rtype: torch.Tensor
        """
        # The Tokenizer returns a dictionary of input_ids and attention_masks.
        #   Both needs to be passed to the model.
        #   Once we get the last hidden state (the embedding), it is a 3-D Vector,
        #   having (batch, sequence, embedding), thus we convert it into a 2-D Vector
        #   by choosing everything from batch and embedding and the first index of sequence
        #   as these models have the summary in the first index `[CLS] token`. 
        language_features = self.backbone(**tokens)
        return self.projection_head(language_features.last_hidden_state[:, 0, :])
