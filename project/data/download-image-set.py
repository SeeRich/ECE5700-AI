import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def download_and_untar(url: str, output_fp: Path) -> None:
    filename = output_fp.name
    output_dir = output_fp.parent
    # Download the file with progress bar using rich
    with Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(
            f"Downloading {filename}", filename=filename, total=100
        )

        def reporthook(block_num: int, block_size: int, total_size: int) -> None:
            progress.update(task, total=total_size, completed=block_num * block_size)

        urllib.request.urlretrieve(url, output_fp, reporthook)
    print(f"Downloaded {filename}")

    # Untar the file
    with tarfile.open(output_fp) as tar:
        tar.extractall(path=output_dir)
        print(f"Extracted {filename}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and extract tar files from a list of URLs."
    )
    parser.add_argument(
        "-f,--file",
        type=str,
        dest="tar_file",
        required=True,
        help="filename of the tar file to download",
    )
    args = parser.parse_args()

    tar_filename = args.tar_file

    # Get the filepath of this script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    input_file = Path(script_dir) / "sam-links.txt"
    output_dir = Path(script_dir) / "images"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file, "r", encoding="UTF-8") as file:
        # Skip the header row
        next(file)

        for line in file:
            filename, url = line.strip().split()
            if tar_filename != filename:
                continue
            # print(f"Downloading {filename} from {url}")
            output_fp = output_dir / filename
            download_and_untar(url, output_fp)

            # Once downloaded, remove the tar file
            os.remove(output_fp)


if __name__ == "__main__":
    main()
