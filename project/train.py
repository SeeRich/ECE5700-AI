import logging
from datetime import datetime
from pathlib import Path
from typing import Any, List

import torch
import torch.nn
import torch.utils.data
from segment_anything import SamPredictor, sam_model_registry

import sa1b_dataset
import utils

# Basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Teacher model pretrained weight data
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "data/weights/sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"


def collate_fn(batch: List[List[Any]] | None) -> Any:
    """Custom collate_fn so we can return numpy arrays"""
    if batch is None:
        return None
    return batch[0]


if __name__ == "__main__":
    # Download if the file does not exist
    if not Path(CHECKPOINT_PATH).exists():
        logging.info("Downloading teacher model checkpoint from %s", CHECKPOINT_URL)
        utils.download_file(CHECKPOINT_URL, Path(CHECKPOINT_PATH))

    # Create predictor
    logging.info("Creating teacher SAM model: %s", MODEL_TYPE)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(utils.get_device())
    predictor = SamPredictor(sam)

    # Load 10% of the 11M samples in the dataset
    # NOTE: if you change the number of samples, you will need to re-download the dataset from scratch
    logging.info("Loading SA1B dataset")
    dataset = sa1b_dataset.SA1BDataset("data", download=True, num_samples=int(6e3))
    seed = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(
        dataset, [0.95, 0.05], generator=seed
    )

    # Create multiprocessing dataloaders (must use batch_size=1 because the images are not all the same size)
    # NOTE: using num_workers > 0 takes longer than using num_workers=0?
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=True, generator=seed, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=1, shuffle=False, generator=seed, collate_fn=collate_fn
    )

    # Create embeddings data directory
    embeddings_dir = Path("data/embeddings")
    if not embeddings_dir.exists():
        embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Keep track of duration
    start_time = datetime.now()

    # Create embeddings using the teacher model
    logging.info("Creating embeddings using teacher model (this may take a while)")
    for i, data in enumerate(train_loader):
        if i % 100 == 0:
            now = datetime.now()
            duration_str = utils.pretty_time_delta((now - start_time).total_seconds())
            logging.info("Processed %d samples, duration: %s seconds", i, duration_str)
        image_name = data["filename"]
        image_dir = data["directory"]
        image = data["image"]
        predictor.set_image(image)
        result = predictor.get_image_embedding()
        predictor.reset_image()
        # Save the result to a file
        output_path = embeddings_dir / image_dir / f"{image_name}.pth"
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, output_path)
