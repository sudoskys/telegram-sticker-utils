# 📦 Telegram Sticker Utils SDK

[![PyPI version](https://badge.fury.io/py/telegram-sticker-utils.svg)](https://badge.fury.io/py/telegram-sticker-utils)
[![Downloads](https://pepy.tech/badge/telegram-sticker-utils)](https://pepy.tech/project/telegram-sticker-utils)

This SDK provides a set of utilities for working with Telegram stickers.

- Convert image formats without losing transparency.
- Auto optimize output size for sticker, make it valid for Telegram.

## 🛠 Supported Pack Types

- [x] Video Sticker
- [x] Static Sticker
- [ ] Animated Sticker

## 🚀 Installation

You need install [ffmpeg](https://ffmpeg.org/download.html) and [pngquant](https://pngquant.org/) before using this SDK.

```shell
apt install ffmpeg
apt install pngquant
pip install telegram-sticker-utils
```

## 📝 Usage

Here is a brief overview of the classes and methods provided by this SDK:

```python
from telegram_sticker_utils import TelegramStickerUtils

# Create an instance of the class
utils = TelegramStickerUtils()
GIF_PATH = "path_to_your_image.gif"
PNG_PATH = "path_to_your_image.png"

# Check if an image is an animated GIF
is_animated = utils.is_animated_gif('path_to_your_image.gif')  # It will return True if the image is a TRUE GIF
print(is_animated)

bytes_io, suffix = utils.make_video_sticker(GIF_PATH, scale=512, master_edge="width")  # or PNG_PATH
print(suffix)
with open("512.webm", "wb") as f:
    f.write(bytes_io.getvalue())

bytes_io, suffix2 = utils.make_static_sticker(PNG_PATH, scale=512, master_edge="width")  # or PNG_PATH
print(suffix2)
with open("512.png", "wb") as f:
    f.write(bytes_io.getvalue())
```

### 📚 TelegramStickerUtils API

This is the main class that provides all the functionality. It includes the following methods:

- `is_animated_gif(image_path: str) -> bool`: Checks if an image is an animated GIF.
- `resize_gif(input_path: str, output_path: str, new_width: int, new_height: int = -1) -> pathlib.Path`: Resizes a GIF
  file using ffmpeg.
- `resize_png(input_path: str, output_path: str, new_width: int, new_height: int = -1) -> pathlib.Path`: Resizes an
  image file using ffmpeg.
- `convert_gif_png(input_path: str, output_path: str, width: int, height: int) -> pathlib.Path`: Converts a GIF file to
  a PNG file.
- `convert_gif_webm(input_path, sec: float = 2.9, bit_rate: str = '2000k', crf: int = 10, cpu_used: int = 2) -> BytesIO`:
  Converts a GIF file to a WebM file.
- `make_static_sticker(input_path: str, *, scale: int = 512, master_edge: Literal["width", "height"] = "width", size_limit=500 * 1024, max_iterations=3) -> Tuple[BytesIO, str]`:
  Resizes a PNG file and optimizes it using pngquant.
- `make_video_sticker(input_path: str, *, scale: int = 512, master_edge: Literal["width", "height"] = "width", size_limit=256 * 1024) -> Tuple[BytesIO, str]`:
  Resizes a gif/png file and optimizes it using pngquant.

