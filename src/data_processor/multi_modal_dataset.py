from typing import Any, Callable, Tuple

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer


class MultiModalDataset(Dataset):
    def __init__(
        self, 
        subset: Subset,
        tokenizer: AutoTokenizer = None,
        image_augmentations: Callable[[Any], Any] | None = None,
        max_length: int = 512
    ) -> None:
        """
        A wrapper to apply transforms to a specific subset of data.

        :param subset: Subset of Data.
        :type subset: Subset
        :param tokenizer: Tokenizer to tokenize the texts
        :type tokenizer: AutoTokenizer
        :param image_augmentations: Image Augmentations that need to be applied; Defaults to None.
        :type image_augmentations: Callable[[Any], Any] | None
        :param max_length: Maximum Token Length to ensure all sequences are of same size;
            Default to 512.
        :type max_length: int

        :return: None
        :rtype: None
        """
        # It is a design choice to keep Augmentations Optional as for Val and Test we would
        #   not add any augmentations, but for Tokenization, we would be performing that
        #   irrespective of the split.
        self.subset: Subset = subset
        self.image_augmentations: Callable[[Any], Any] | None = image_augmentations
        self.tokenizer: AutoTokenizer = tokenizer
        self.max_length: int = max_length
        
    def __getitem__(self, index: int) -> Tuple[Tensor, dict[str, Tensor], int]:
        """
        Retrieves an element from the dataset.

        :param index: Index to be retrieved.
        :type index: int
        :return: Returns Image, Text and the corresponding Document Category.
        :rtype: Tuple[Tensor, dict[str, Tensor], int]
        """
        image, text, label = self.subset[index]
        # Applies Augmentations if Any
        if self.image_augmentations:
            # Albumentations only take in Numpy.
            image_np = np.array(image.convert('RGB'))
            # Albumentations return a dictionary, with keywords, bbox etc,
            #   but we are only interested in the transformed image.
            image = self.image_augmentations(image_np)['image']
        
        # Tokenizes the Text.
        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # Squeeze to remove the extra batch dimension added by return_tensors
        encoded_text['input_ids'] = encoded_text['input_ids'].squeeze(0)
        encoded_text['attention_mask'] = encoded_text['attention_mask'].squeeze(0)
        return image, encoded_text, label
        
    def __len__(self) -> int:
        """
        Returns the number of samples in the Dataset.

        :return: Number of datapoints or samples present in the Dataset.
        :rtype: int
        """
        return len(self.subset)
