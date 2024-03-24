import shutil

import pngquant

from .resize import TelegramStickerUtils
from .custom import IGNORE_CHECKS


def check_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError(
            "ffmpeg doesn't seem to be installed, or its path is not added to the PATH environment variable."
        )


def check_pngquant():
    if not shutil.which("pngquant"):
        raise FileNotFoundError(
            "pngquant doesn't seem to be installed, or its path is not added to the PATH environment variable."
        )


if not IGNORE_CHECKS:
    check_ffmpeg()
    check_pngquant()
