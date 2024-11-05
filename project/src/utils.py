import urllib.request
from pathlib import Path

import torch
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def download_file(url: str, fullpath: Path) -> None:
    if not fullpath.parent.exists():
        fullpath.parent.mkdir(parents=True, exist_ok=True)
    filename = fullpath.name
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
            progress.update(task, total=total_size, completed=block_num * block_size, refresh=True)

        urllib.request.urlretrieve(url, fullpath, reporthook)


def pretty_time_delta(seconds: float) -> str:
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days > 0:
        return "%dd%dh%dm%ds" % (days, hours, minutes, seconds)
    elif hours > 0:
        return "%dh%dm%ds" % (hours, minutes, seconds)
    elif minutes > 0:
        return "%dm%ds" % (minutes, seconds)
    else:
        return "%ds" % (seconds,)
