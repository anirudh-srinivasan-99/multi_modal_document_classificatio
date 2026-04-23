import albumentations as A
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, Subset
from transformers import AutoTokenizer


class MultiModalDataset(Dataset):
    def __init__(
        self, 
        subset: Subset,
        tokenizer: AutoTokenizer = None,
        image_transformations: A.Compose | None = None,
        max_seq_length: int = 512,
    ) -> None:
        """
        A wrapper to apply transforms to a specific subset of data.

        :param subset: Subset of Data.
        :type subset: Subset
        :param tokenizer: Tokenizer to tokenize the texts
        :type tokenizer: AutoTokenizer
        :param image_transformations: Image Augmentations that need to be applied; Defaults to None.
        :type image_transformations: A.Compose | None
        :param max_seq_length: Maximum Token Length to ensure all sequences are of same size;
            Default to 512.
        :type max_seq_length: int

        :return: None
        :rtype: None
        """
        # It is a design choice to keep Augmentations Optional as for Val and Test we would
        #   not add any augmentations, but for Tokenization, we would be performing that
        #   irrespective of the split.
        self.subset: Subset = subset
        self.tokenizer: AutoTokenizer = tokenizer
        self.image_transformations: A.Compose | None = image_transformations
        self.max_seq_length: int = max_seq_length
        
    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor], Tensor]:
        """
        Retrieves an element from the dataset.

        :param index: Index to be retrieved.
        :type index: int
        :return: Returns Image, Text and the corresponding Document Category.
        :rtype: Tuple[Tensor, dict[str, Tensor], int]
        """
        print(f'Subset: {self.subset[index]}')
        image = self.subset[index]['image']
        text = self.subset[index]['text']
        label = self.subset[index]['label_id']
        # Applies Augmentations if Any
        if self.image_transformations:
            # Albumentations only take in Numpy.
            image_np = np.array(image.convert('RGB'))
            # Albumentations return a dictionary, with keywords, bbox etc,
            #   but we are only interested in the transformed image.
            image = self.image_transformations(image=image_np)['image']
        
        # Tokenizes the Text.
        encoded_text = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        # Squeeze to remove the extra batch dimension added by return_tensors
        encoded_text['input_ids'] = encoded_text['input_ids'].squeeze(0)
        encoded_text['attention_mask'] = encoded_text['attention_mask'].squeeze(0)

        label = torch.tensor(label, dtype=torch.long)
        return image, encoded_text, label
        
    def __len__(self) -> int:
        """
        Returns the number of samples in the Dataset.

        :return: Number of datapoints or samples present in the Dataset.
        :rtype: int
        """
        return len(self.subset)
