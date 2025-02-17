import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, List

import torch
import torch.nn as nn
import torch.utils.data
from segment_anything import SamPredictor, sam_model_registry

import sa1b_dataset
import utils
from mobile_sam_image_encoder import SamImageEncoder

# python src/train.py --gen-embeddings --num-samples 220000

# Teacher model pretrained weight data
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "data/weights/sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

logger = logging.getLogger("ece570.train")


def setup_logging() -> logging.Logger:
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create a file handler
    file_handler = logging.FileHandler("train.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def collate_fn(batch: List[List[Any]] | None) -> Any:
    """Custom collate_fn so we can return numpy arrays"""
    if batch is None:
        return None
    return batch[0]


def gen_teacher_embeddings(num_samples: int) -> None:
    # Download if the file does not exist
    if not Path(CHECKPOINT_PATH).exists():
        logger.info("Downloading teacher model checkpoint from %s", CHECKPOINT_URL)
        utils.download_file(CHECKPOINT_URL, Path(CHECKPOINT_PATH))

    # Load 10% of the 11M samples in the dataset
    # NOTE: if you change the number of samples, you will need to re-download the dataset from scratch
    logger.info("Loading SA1B dataset")
    dataset = sa1b_dataset.SA1BDataset("data", download=True, num_samples=num_samples)
    seed = torch.Generator().manual_seed(42)

    # Create dataloaders (must use batch_size=1 because the images are not all the same size)
    # NOTE: using num_workers > 0 takes longer than using num_workers=0?
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, generator=seed, collate_fn=collate_fn
    )

    # Create embeddings data directory
    embeddings_dir = Path("data/embeddings")
    if not embeddings_dir.exists():
        embeddings_dir.mkdir(parents=True, exist_ok=True)

    # Keep track of duration
    start_time = datetime.now()

    # Create predictor
    logger.info("Creating teacher SAM model: %s", MODEL_TYPE)
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(utils.get_device())
    predictor = SamPredictor(sam)

    # Create embeddings using the teacher model
    logger.info("Creating embeddings using teacher model (this may take a while)")
    for i, data in enumerate(train_loader):
        if i % 100 == 0:
            now = datetime.now()
            duration_str = utils.pretty_time_delta((now - start_time).total_seconds())
            logger.info("Processed %d samples, duration: %s seconds", i, duration_str)
        if "embedding" in data and data["embedding"]:
            # logger.debug("Skipping")
            continue
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

    logger.info(
        "Finished creating embeddings: %s seconds",
        utils.pretty_time_delta((datetime.now() - start_time).total_seconds()),
    )


def train(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    data_loader: torch.utils.data.DataLoader,
    epoch: int,
    start_time: datetime,
    device: torch.device,
) -> List[float]:
    """Train function for the student model (single epoch)."""
    # Set model to training mode
    model.train()

    train_loss = []

    for batch_idx, (images, targets) in enumerate(data_loader):
        # move the data to the device
        images, targets = images.to(device), targets.to(device)
        # Model forward evaluation
        output = model(images)
        # Calculate loss
        loss = criterion(output, targets)
        # Loss backward propagation
        loss.backward()
        # Gradient evaluation and backward propagation
        optimizer.step()

        train_loss.append(
            loss.item()
        )  # item() is to get the value of the tensor directly
        if batch_idx % 10 == 0:  # We log our output every 100 batches
            now = datetime.now()
            duration_str = utils.pretty_time_delta((now - start_time).total_seconds())
            logger.info(
                f"Epoch {epoch}: [{batch_idx*len(images)}/{len(data_loader.dataset)}], Loss: {loss.item()}, Duration: {duration_str}"
            )
    # ----------- <End Your code> ---------------
    assert len(train_loss) == len(train_loader)
    return train_loss


def test(
    model: nn.Module,
    criterion: nn.modules.loss._Loss,
    data_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
) -> None:
    # Set the model to evaluation mode (i.e. not training)
    model.eval()

    test_loss = 0

    with torch.no_grad():
        for images, targets in data_loader:
            # Move the data to the device
            images, targets = images.to(device), targets.to(device)
            # Evaluate the model on the batch
            output = model(images)
            # Sum up batch loss
            test_loss += criterion(output, targets).item() * images.size(0)

    # Number of total test samples
    total_num = len(data_loader.dataset)
    test_loss /= total_num

    print(
        f"Test result on epoch {epoch}: total samples: {total_num}, Loss: {test_loss:.3f}"
    )


if __name__ == "__main__":
    logger = setup_logging()

    # CLI arguments
    parser = argparse.ArgumentParser(description="Train a student SAM model")
    parser.add_argument(
        "--gen-embeddings", action="store_true", help="Generate embeddings"
    )
    parser.add_argument(
        "--num-samples", type=int, default=300, help="Number of samples"
    )
    args = parser.parse_args()

    ######################## Generate embeddings ########################
    if args.gen_embeddings:
        gen_teacher_embeddings(num_samples=args.num_samples)

    ######################## Train the student model ########################
    # Create the student model
    logger.info("Creating student model")
    sam_image_encoder = SamImageEncoder()
    sam_image_encoder.to(device=utils.get_device())

    logger.info("Loading SA1B student dataset")
    dataset = sa1b_dataset.SA1BStudentDataset(
        sam_image_encoder.get_image_size(), utils.get_device(), "data"
    )
    seed = torch.Generator().manual_seed(42)
    train_set, test_set = torch.utils.data.random_split(
        dataset, [0.95, 0.05], generator=seed
    )

    # Create data loaders for training and testing
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=8, shuffle=True, generator=seed
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=False)

    num_train = len(train_set)
    num_test = len(test_set)
    logger.info(
        "Training student model with %s training samples, %s test samples, %s total samples",
        num_train,
        num_test,
        num_train + num_test,
    )

    # Train the student model for n_epochs
    n_epochs = 2
    start_time = datetime.now()
    optimizer = torch.optim.Adam(sam_image_encoder.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    for epoch in range(n_epochs):
        train(
            model=sam_image_encoder,
            criterion=nn.MSELoss(),
            optimizer=optimizer,
            data_loader=train_loader,
            epoch=epoch,
            start_time=start_time,
            device=utils.get_device(),
        )
        test(
            model=sam_image_encoder,
            criterion=nn.MSELoss(),
            data_loader=test_loader,
            epoch=epoch,
            device=utils.get_device(),
        )
        scheduler.step()

        # Save the student model checkpoint (just do this each epoch)
        checkpoint_path = f"data/weights/student_model_epoch_{epoch}.pth"
        logger.info("Saving student model checkpoint to %s", checkpoint_path)
        sam_image_encoder.cpu()
        torch.save(sam_image_encoder.state_dict(), checkpoint_path)
        sam_image_encoder.to(utils.get_device())
