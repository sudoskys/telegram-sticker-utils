# 📦 Telegram Sticker Utils SDK

[![PyPI version](https://badge.fury.io/py/telegram-sticker-utils.svg)](https://badge.fury.io/py/telegram-sticker-utils)
[![Downloads](https://pepy.tech/badge/telegram-sticker-utils)](https://pepy.tech/project/telegram-sticker-utils)

If you are not a developer, you can use the Telegram Sticker [CLI](https://github.com/sudoskys/tsticker) (developed by
this SDK) for
user-friendly operations.

This SDK provides a set of utilities for working with Telegram stickers.

- Convert image formats without losing transparency.
- Auto optimize output size for sticker, make it valid for Telegram.
- Auto-detect sticker type and emojis.

## 🛠 Supported Pack Types

- [x] Video Sticker
- [x] Static Sticker
- [ ] Animated Sticker(Tgs)

## 🚀 Installation

You need install **[ImageMagick](https://github.com/imagemagick/imagemagick)** and 
**[ffmpeg](https://www.ffmpeg.org/download.html)** before using this SDK.

Install Guide: https://docs.wand-py.org/en/0.6.12/guide/install.html

```shell
apt install ffmpeg
pip3 install telegram-sticker-utils
```

## 📖 Usage

```python
from telegram_sticker_utils import ImageProcessor
from telegram_sticker_utils import is_animated_gif

print(is_animated_gif('test.gif'))  # Path to the image file or a file-like object.

for sticker_file in ["happy.webp", "sad.png", "高兴.jpg", "悲伤.gif"]:
    sticker = ImageProcessor.make_sticker(
        input_name=sticker_file,
        input_data=open(sticker_file, 'rb').read(),
        scale=512
    )
    print(sticker.sticker_type)
    print(sticker.emojis)
    with open(f"{sticker_file}.{sticker.file_extension}", 'wb') as f:
        f.write(sticker.data)
```