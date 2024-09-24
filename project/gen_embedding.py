import argparse
import logging
import pathlib
import sys
from pathlib import Path
from typing import List

import cv2
import torch
from segment_anything import SamPredictor, sam_model_registry

# Basic configuration for logging
logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Checkpoint details
MODEL_TYPE = "vit_h"
CHECKPOINT_PATH = "weights/sam_vit_h_4b8939.pth"


# Define a custom argument type for a list of filepaths
def list_of_filepaths(arg: str) -> List[Path]:
    result = []
    # Split the argument by commas
    filepaths = arg.split(",")
    # Check if all the filepaths exist, log a warning if not
    for filepath in filepaths:
        fp = Path(filepath)
        if not fp.exists():
            logging.warning("Filepath does not exist: %s", filepath)
            continue
        result.append(fp)
    return result


if __name__ == "__main__":
    # Print torch device setup (i.e. cuda)
    if not torch.cuda.is_available():
        logging.error("Torch CUDA is not available")
        sys.exit(1)
    logging.info("Torch CUDA version: %s", torch.version.cuda)

    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=list_of_filepaths)
    parser.add_argument("--image-dir", type=pathlib.Path)
    args = parser.parse_args()

    torch_device = torch.device("cuda:0")
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=torch_device)

    predictor = SamPredictor(sam)

    # Build a list of images from arguments
    images = []
    if args.images:
        images.extend(args.images)
    if args.image_dir:
        images.extend(args.image_dir.glob("*.jpg"))

    # Loop over the images
    for i, image_path in enumerate(images):
        image_bgr = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        predictor.set_image(image_rgb)
        result = predictor.get_image_embedding()
        predictor.reset_image()
        # logging.info("Result: %s", result.size())
        # Save the result to a file (new parent directory "embeddings")
        output_path = Path("data/embeddings") / (image_path.stem + ".pth")
        print(f"Saving to {output_path}, count: {i}")
        torch.save(result, output_path)
