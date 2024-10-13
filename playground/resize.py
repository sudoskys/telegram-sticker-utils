import sys

from loguru import logger

from telegram_sticker_utils import ImageProcessor

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
sticker = ImageProcessor.make_sticker(
    input_name='sad',
    input_data=open("st22.mp4", 'rb').read(),
    scale=512,
    master_edge='width',
)
print(sticker.sticker_type)
print(sticker.emojis)
with open(f"output.{sticker.file_extension}", 'wb') as f:
    f.write(sticker.data)
