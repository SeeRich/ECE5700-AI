# Custom PyTorch Dataset class for subset of SA1B dataset
import os
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Tuple

import torch
import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import Dataset
import torch.nn.functional as F

from segment_anything.utils.transforms import ResizeLongestSide

import utils

logger = logging.getLogger(__name__)

T = dict[str, NDArray[np.uint8]]


class SA1BDataset(Dataset[T]):
    """SA1B dataset."""

    def __init__(self, root_dir: str, download: bool, num_samples: int):
        """
        Arguments:
        root_dir (str): Directory to store the dataset
        download (bool): Whether to download the dataset
        num_samples (int): Number of samples to download
        """
        self.root_dir = Path(root_dir)
        self.download = download
        self.num_samples = num_samples

        # Get this file folder directory
        self.file_path = Path(os.path.dirname(__file__))
        # Links filepath
        self.sam_links = self.file_path / "sam-links.txt"
        self.embeddings_dir = self.root_dir / "embeddings"

        # Dataframe stores two string columns: directory, filename
        self.data = pd.DataFrame(columns=["directory", "filename"])

        cache_fp = self.root_dir / "cache.parquet"

        # Verify that the dataset is downloaded and ready to use
        # Check for the existence of the cache file
        # If not, move on to download the dataset
        if cache_fp.exists():
            self.data = pd.read_parquet(cache_fp)
            logger.debug(f"Loaded cached data, num samples: {len(self.data)}")

            # If we do have a cache file, load any embeddings present as well.
            # This helps work around training that was interrupted.
            has_embeddings: list[bool] = []
            for _, row in self.data.iterrows():
                directory = row["directory"]
                image = row["filename"]
                embedding_fp = self.embeddings_dir / directory / f"{image}.pth"
                has_embeddings.append(embedding_fp.exists())
            self.data["embedding"] = has_embeddings
            num_missing_embeddings = self.data["embedding"].value_counts().get(False, 0)
            if num_missing_embeddings > 0:
                logger.warning(
                    "Missing %d embeddings",
                    num_missing_embeddings,
                )
        else:
            self.__load_existing_data()

        if download and not cache_fp.exists():
            logger.info("Downloading SA1B Dataset")
            # Load sam-links.txt as DataFrame, file is text file with tab-separated values of filename, url
            links = pd.read_csv(self.sam_links, sep="\t", header=0)
            # Create the directory if it doesn't exist
            self.root_dir.mkdir(parents=True, exist_ok=True)
            # Process the links and download the data
            self.__process_links(self.root_dir, links, num_samples)
            # Cache data frame to disk so we don't have to download it again
            self.data.to_parquet(cache_fp)

        # Filter the list by the number of desired samples
        if len(self.data) < num_samples:
            raise ValueError("Not enough samples to meet request")
        # self.data = self.data.sample(num_samples, random_state=42)
        logger.debug(f"Loaded data, num samples: {len(self.data)}")

    def __len__(self) -> int:
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def __getitem__(self, idx: int) -> T:
        directory = self.data.iloc[idx]["directory"]
        filename = self.data.iloc[idx]["filename"]
        has_embedding = self.data.iloc[idx]["embedding"]
        image_fp = self.root_dir / directory / f"{filename}.jpg"
        image_bgr = cv2.imread(image_fp)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sample = {
            "image": image_rgb,
            "directory": directory,
            "filename": filename,
            "embedding": has_embedding,
        }
        return sample

    def __image_path_to_dir_and_file(self, image_path: Path) -> Tuple[str, str]:
        # File path is like: "directory/blah/blah/data/train/sa_000023/sa_002340.jpg"
        # Get the directory name
        directory = image_path.parent.name
        # Get the file name
        file_name = image_path.stem
        return directory, file_name

    def __load_existing_data(self) -> None:
        # Get a list of all the files in the directory
        data_dirs = list(self.root_dir.glob("sa_*"))
        files = []
        for dir in data_dirs:
            files.extend(list([f for f in list(dir.glob("sa_*"))]))
        # Get a list of all the images
        images = [f for f in files if f.name.endswith(".jpg")]

        # Process into directory and filenames
        directories = []
        filenames = []
        for img_fp in images:
            dir, filename = self.__image_path_to_dir_and_file(img_fp)
            directories.append(dir)
            filenames.append(filename)

        # Create a dataframe to concatenate
        self.data = pd.concat(
            [
                self.data,
                pd.DataFrame({"directory": directories, "filename": filenames}),
            ]
        )

    def __process_links(
        self, data_dir: Path, links: pd.DataFrame, num_samples: int
    ) -> None:
        # Iterate over the links and download the data
        for _, link in links.iterrows():
            # Break if we have enough samples
            if len(self.data) >= num_samples:
                break

            url: str = link["cdn_link"]
            filename: str = link["file_name"]
            output_fp = data_dir / filename
            # Only do the download + untar if the directory doesn't exist
            untared_dir = output_fp.parent / output_fp.stem
            if untared_dir.exists():
                continue
            logger.debug(
                "Continuing download: %s / %s samples", len(self.data), num_samples
            )
            image_dir = self.__download_and_untar(url, output_fp)
            # Remove the tar file
            os.remove(output_fp)
            # Get a list of all the files in the directory
            files = list(image_dir.glob("*"))
            # Get a list of all the images
            images = [f for f in files if f.name.endswith(".jpg")]

            # Process into directory and filenames
            directories = []
            filenames = []
            for img_fp in images:
                dir, filename = self.__image_path_to_dir_and_file(img_fp)
                directories.append(dir)
                filenames.append(filename)

            # Create a dataframe to concatenate
            self.data = pd.concat(
                [
                    self.data,
                    pd.DataFrame({"directory": directories, "filename": filenames}),
                ]
            )

    def __download_and_untar(self, url: str, archive_fp: Path) -> Path:
        """Download a file from a URL and extract it to a directory. Returns the directory path."""
        utils.download_file(url, archive_fp)

        # Untar the file and delete the tar file
        image_dir = archive_fp.parent / archive_fp.stem
        logger.info(f"Extracting images from {archive_fp} to directory {image_dir}")
        # Remove this image_dir if it already exists
        if image_dir.exists():
            shutil.rmtree(image_dir)
        with tarfile.open(archive_fp) as tar:
            tar.extractall(path=image_dir)
        return image_dir


class SA1BImagePreprocessor:
    def __init__(self, img_size: int, device: torch.device):
        self.img_size = img_size
        self.device = device
        self.transform = ResizeLongestSide(img_size)
        self.pixel_mean = (
            torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        )
        self.pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Do transform
        x = self.transform.apply_image(x)
        # Convert to tensor
        xt = torch.as_tensor(x, device=self.device)
        # Permute to NCHW
        xt = xt.permute(2, 0, 1).contiguous()[None, :, :, :]

        # Normalize colors
        xt = (xt - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = xt.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        xt = F.pad(xt, (0, padw, 0, padh))
        return xt


class SA1BStudentDataset(Dataset[T]):
    """SA1B student dataset. This includes the images and the teacher's embeddings."""

    def __init__(
        self,
        img_size: int,
        device: torch.device,
        root_dir: str,
        num_samples: int | None = None,
    ):
        """
        Arguments:
        root_dir (str): Directory to store the dataset
        """
        self.root_dir = Path(root_dir)
        self.embeddings_dir = self.root_dir / "embeddings"

        self.device = device

        # Setup preprocessing
        self.preprocessor = SA1BImagePreprocessor(img_size, device)

        # Dataframe stores three string columns:
        # directory, image (filename), embedding (boolean)
        self.data = pd.DataFrame(columns=["directory", "image", "embedding"])

        # Cached data file for original image only dataset
        # This must exist to load the dataset
        cache_fp = self.root_dir / "cache.parquet"
        if not cache_fp.exists():
            raise ValueError("Cached data file does not exist")

        # Load the original dataset, for each image, get the embedding
        orig_data = pd.read_parquet(cache_fp)
        has_embeddings: list[bool] = []
        for _, row in orig_data.iterrows():
            directory = row["directory"]
            image = row["filename"]
            embedding_fp = self.embeddings_dir / directory / f"{image}.pth"
            has_embeddings.append(embedding_fp.exists())
        orig_data["embedding"] = has_embeddings

        # Log a warning if the not all embeddings exist, and the number missing
        # Get the number of missing embeddings
        num_missing_embeddings = orig_data["embedding"].value_counts().get(False, 0)
        if num_missing_embeddings > 0:
            logger.warning(
                "Missing %d embeddings",
                num_missing_embeddings,
            )

        # Filter out rows where embedding is False
        self.data = orig_data[orig_data["embedding"]]

        # reduce size if requested
        if num_samples is not None:
            self.data = self.data.sample(num_samples)

    def __len__(self) -> int:
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def __getitem__(self, idx: int) -> T:
        """Returns a dictionary of teacher embeddings and the image"""
        directory = self.data.iloc[idx]["directory"]
        filename = self.data.iloc[idx]["filename"]
        image_fp = self.root_dir / directory / f"{filename}.jpg"
        embedding_fp = self.embeddings_dir / directory / f"{filename}.pth"
        embedding = torch.load(embedding_fp, weights_only=True)
        image_bgr = cv2.imread(image_fp)
        # Model expects RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        # Preprocess image
        image = self.preprocessor.preprocess(image_rgb)
        # Remove the first dimension
        image = image.squeeze(0)
        embedding = embedding.squeeze(0)
        return (image, embedding)
