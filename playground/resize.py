import sys

from loguru import logger

from telegram_sticker_utils import ImageProcessor
from telegram_sticker_utils.core.const import add_emoji_rule

logger.remove(0)
handler_id = logger.add(
    sys.stderr,
    format="<level>[{level}]</level> | <level>{message}</level> | "
           "<cyan>{name}:{function}:{line}</cyan> <yellow>@{time}</yellow>",
    colorize=True,
    backtrace=True,
    enqueue=True,
    level="TRACE",
)
add_emoji_rule("sad", "😢")
sticker = ImageProcessor.make_sticker(
    input_name='私密马赛',
    input_data=open("酷酷.gif", 'rb').read(),
    scale=512,
    master_edge='width',
)
print(sticker.sticker_type)
print(sticker.emojis)
with open(f"output.{sticker.file_extension}", 'wb') as f:
    f.write(sticker.data)

import mimetypes


def validate_webm(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type != 'video/webm':
        raise ValueError(f"Expected 'video/webm', but got '{mime_type}'")
    print("The file is a valid WebM video.")


# Example usage
validate_webm('output.webm')
