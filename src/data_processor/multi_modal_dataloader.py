import albumentations as A
import datasets
import lightning as L
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.config.env_loader import DefaultPaths as DP
from src.data_processor.multi_modal_dataset import MultiModalDataset


class MultiModalDataLoader(L.LightningDataModule):
    def __init__(
        self,
        hf_repo_id: str,
        batch_size: int,
        image_size: tuple[int, int],
        max_seq_length: int,
        tokenizer: AutoTokenizer,
        dataset_mean: tuple[float, float, float],
        dataset_std: tuple[float, float, float],
        hf_token: str | None = None,
        hf_force_redownload: bool = False
    ) -> None:
        """
        Constructor of the Dataloader.

        :param hf_repo_id: Repo-id of the huggingFace Repository.
        :type hf_repo_id: str
        :param batch_size: Batch Size to be used while loading Data.
        :type batch_size: int
        :param image_size: Size of the image (H, W). Dependent on the input layer of
            the Vision Backbone being used.
        :type image_size: tuple[int, int]
        :param max_seq_length: Maximum number of tokens for a sequence.
        :type max_seq_length: int
        :param tokenizer: Tokenizer used to tokenize the text. Depends on the Language Backbone
            being used.
        :type tokenizer: AutoTokenizer
        :param dataset_mean: Mean RGB values (0-1 scale) across all training images.
        :type dataset_mean: tuple[float, float, float]
        :param dataset_std: Standard deviation RGB values (0-1 scale) across all training images.
        :type dataset_std: tuple[float, float, float]
        :param hf_token: Passes HuggingFace Token for the API call to improve
            download speeds and download from private repo.
        :type hf_token: str | None
        :param hf_force_redownload: A boolean Flag to download the dataset
            even if cache is already present.
        :type hf_force_redownload: bool

        :return: None
        :rtype: None
        """
        super().__init__()
        self.__hf_repo_id: str = hf_repo_id
        self.batch_size: int = batch_size
        self.image_size: tuple[int, int] = image_size
        self.tokenizer: AutoTokenizer = tokenizer
        self.max_seq_length: int = max_seq_length
        self.dataset_mean: tuple[float, float, float] = dataset_mean
        self.dataset_std: tuple[float, float, float] = dataset_std

        self.__hf_token: str | None = hf_token if hf_token else None
        self.__hf_download_mode: str = (
            datasets.DownloadMode.FORCE_REDOWNLOAD
            if hf_force_redownload
            else datasets.DownloadMode.REUSE_DATASET_IF_EXISTS
        )

        # Train, Validation and Test Data.
        self.train_data: Dataset | None= None
        self.val_data: Dataset | None = None
        self.test_data: Dataset | None = None

    def prepare_data(self) -> None:
        """
        In-build method to handle I/O operations. In a multi-GPU setup,
        the I/O operation of downloading the dataset needs to happen once
        and gets stored in the disk and then `setup()` method loads the corresponding
        data to its respective GPUs.
        This is to ensure that all the GPUs do not download the data simultaneous resulting
        in a disk related issue.

        :return: None
        :rtype: None
        """
        # Load just the metadata/features without downloading the actual images.
        # This is very fast.
        datasets.load_dataset(
            self.__hf_repo_id, cache_dir=DP.HF_CACHE_DIR,
            token=self.__hf_token,
            download_mode=self.__hf_download_mode
        )

    def setup(self, stage: str) -> None:
        """
        In-build method used to setup data on various GPU Clusters [If many].

        :param stage: Stage with which the setup is called. Usually has value of
            fit, valid, test and None.
        :type stage: str
        :return: None
        :rtype: None
        """
        # Train Data Transformations to add some Noise to the Documents.
        h, w = self.image_size
        train_transform = A.Compose([
            # Geometric Transformations
            A.HorizontalFlip(p=0.25),
            A.Affine(rotate=(-2, 2), shear=(-2, 2), p=0.25),
            A.Resize(height=h, width=w),

            # Pixel Transformations
            A.SaltAndPepper(p=0.25),
            A.Blur(blur_limit=(3, 5), p=0.25),
            A.Normalize(
                mean=self.dataset_mean,
                std=self.dataset_std
            ),
            A.ToTensorV2()
        ])
        # We do not add much transformations for Validation and Test Data.
        #   Mostly it is Resizing, Normalization and converting to PyTorch Vectors.
        val_transform = A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(
                mean=self.dataset_mean,
                std=self.dataset_std
            ),
            A.ToTensorV2()
        ])
        test_transform = A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(
                mean=self.dataset_mean,
                std=self.dataset_std
            ),
            A.ToTensorV2()
        ])
        # In the `prepare_data()` method, data gets loaded into the disk,
        #   and in this method, data gets loaded into the memory.
        match stage:
            # Lightning requires both Train and Val data when fitting the model.
            case 'fit' | 'validate' | None:
                train_base = datasets.load_dataset(self.__hf_repo_id, split='train', cache_dir=DP.HF_CACHE_DIR)
                self.train_data = MultiModalDataset(
                    subset=train_base,
                    tokenizer=self.tokenizer,
                    image_transformations=train_transform,
                    max_seq_length=self.max_seq_length,
                )
                val_base = datasets.load_dataset(self.__hf_repo_id, split='validation', cache_dir=DP.HF_CACHE_DIR)
                self.val_data = MultiModalDataset(
                    subset=val_base,
                    tokenizer=self.tokenizer,
                    image_transformations=val_transform,
                    max_seq_length=self.max_seq_length,
                )
            case 'test':
                test_base = datasets.load_dataset(self.__hf_repo_id, split='test', cache_dir=DP.HF_CACHE_DIR)
                self.test_data = MultiModalDataset(
                    subset=test_base,
                    tokenizer=self.tokenizer,
                    image_transformations=test_transform,
                    max_seq_length=self.max_seq_length,
                )


    def train_dataloader(self) -> DataLoader:
        """
        In-build Method used to load Train Data.

        :raises ValueError: self.train_data is empty, implying train-data was not loaded. 
        :return: Train Data as a DataLoader object.
        :rtype: DataLoader 
        """
        # `pin_memory=True` essentially page-locks the data, thus resulting in a faster
        #   transfer of data between CPU and GPU.
        if not self.train_data:
            raise ValueError('Training Data not set properly !!')
        return DataLoader(
            self.train_data,
            self.batch_size,
            shuffle=True,
            pin_memory=True
        )

    def val_dataloader(self) -> DataLoader:
        """
        In-build Method used to load Validation Data.

        :raises ValueError: self.val_data is empty, implying val-data was not loaded. 
        :return: Validation Data as a DataLoader object.
        :rtype: DataLoader 
        """
        # `pin_memory=True` essentially page-locks the data, thus resulting in a faster
        #   transfer of data between CPU and GPU.
        if not self.val_data:
            raise ValueError('Validation Data not set properly !!')
        return DataLoader(
            self.val_data,
            self.batch_size,
            pin_memory=True
        )

    def test_dataloader(self) -> DataLoader:
        """
        In-build Method used to load Test Data.

        :raises ValueError: self.test_data is empty, implying test-data was not loaded. 
        :return: Test Data as a DataLoader object.
        :rtype: DataLoader 
        """
        if not self.test_data:
            raise ValueError('Test Data not set properly !!')
        # `pin_memory=True` essentially page-locks the data, thus resulting in a faster
        #   transfer of data between CPU and GPU.
        return DataLoader(
            self.test_data,
            self.batch_size,
            pin_memory=True
        )
