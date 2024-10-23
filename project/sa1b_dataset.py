# Custom PyTorch Dataset class for subset of SA1B dataset
import os
import logging
import shutil
import tarfile
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import Dataset

import utils

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

        # Dataframe stores two string columns: directory, filename
        self.data = pd.DataFrame(columns=["directory", "filename"])

        cache_fp = self.root_dir / "cache.parquet"

        # Verify that the dataset is downloaded and ready to use
        # Check for the existence of the cache file
        # If not, move on to download the dataset
        if cache_fp.exists():
            self.data = pd.read_parquet(cache_fp)
            logging.info(f"Loaded cached data, num samples: {len(self.data)}")
            return

        if download:
            # Load sam-links.txt as DataFrame, file is text file with tab-separated values of filename, url
            links = pd.read_csv("sam-links.txt", sep="\t", header=0)
            # TEMPORARY trim to first 4 rows
            links = links.iloc[:2]
            # Remove existing files in the directory
            if self.root_dir.exists():
                shutil.rmtree(self.root_dir)
            # Create the directory if it doesn't exist
            self.root_dir.mkdir(parents=True, exist_ok=True)
            # Process the links and download the data
            self.__process_links(self.root_dir, links, num_samples)
            # Cache data frame to disk so we don't have to download it again
            self.data.to_parquet(cache_fp)

    def __len__(self) -> int:
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def __getitem__(self, idx: int) -> T:
        directory = self.data.iloc[idx]["directory"]
        filename = self.data.iloc[idx]["filename"]
        image_fp = self.root_dir / directory / f"{filename}.jpg"
        image_bgr = cv2.imread(image_fp)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sample = {"image": image_rgb, "directory": directory, "filename": filename}
        return sample

    def __image_path_to_dir_and_file(self, image_path: Path) -> Tuple[str, str]:
        # File path is like: "directory/blah/blah/data/train/sa_000023/sa_002340.jpg"
        # Get the directory name
        directory = image_path.parent.name
        # Get the file name
        file_name = image_path.stem
        return directory, file_name

    def __process_links(
        self, data_dir: Path, links: pd.DataFrame, num_samples: int
    ) -> None:
        # Iterate over the links and download the data
        for _, link in links.iterrows():
            # Break if we have enough samples
            if len(self.data) >= num_samples:
                break
            url = link["cdn_link"]
            filename = link["file_name"]
            output_fp = data_dir / filename
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
        logging.info(f"Extracting images from {archive_fp} to directory {image_dir}")
        # Remove this image_dir if it already exists
        if image_dir.exists():
            shutil.rmtree(image_dir)
        with tarfile.open(archive_fp) as tar:
            tar.extractall(path=image_dir)
        return image_dir


class SA1BStudentDataset(Dataset[T]):
    """SA1B student dataset. This includes the images and the teacher's embeddings."""

    def __init__(self, root_dir: str):
        """
        Arguments:
        root_dir (str): Directory to store the dataset
        """
        self.root_dir = Path(root_dir)
        self.embeddings_dir = self.root_dir / "embeddings"

        # Dataframe stores three string columns:
        # directory, image (filename), embedding (boolean)
        self.data = pd.DataFrame(columns=["directory", "image", "embedding"])

        # Cached data file for original image only dataset
        # This must exist to load the dataset
        cache_fp = self.root_dir / "cache.parquet"
        if not cache_fp.exists():
            raise ValueError("Cached data file does not exist")

        # Load the original dataset, for each image, get the embedding
        # Log a warning if the embedding does not exist
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
            logging.warning(
                "Missing %d embeddings",
                num_missing_embeddings,
            )

        # Filter out rows where embedding is False
        self.data = orig_data[orig_data["embedding"]]

    def __len__(self) -> int:
        if self.data is None:
            raise ValueError("Dataset not loaded")
        return len(self.data)

    def __getitem__(self, idx: int) -> T:
        directory = self.data.iloc[idx]["directory"]
        filename = self.data.iloc[idx]["filename"]
        image_fp = self.root_dir / directory / f"{filename}.jpg"
        image_bgr = cv2.imread(image_fp)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        sample = {"image": image_rgb, "directory": directory, "filename": filename}
        return sample

    def __image_path_to_dir_and_file(self, image_path: Path) -> Tuple[str, str]:
        # File path is like: "directory/blah/blah/data/train/sa_000023/sa_002340.jpg"
        # Get the directory name
        directory = image_path.parent.name
        # Get the file name
        file_name = image_path.stem
        return directory, file_name
