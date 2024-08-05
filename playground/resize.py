import os
import pathlib
from typing import Union, IO

from wand.image import Image


def resize_gif(
        input_data: Union[str, bytes, os.PathLike, IO[bytes]],
        new_width: int,
        new_height: int = -1
) -> bytes:
    """
    Resize a GIF file using wand.

    :param input_data: Path to the input GIF file or binary data.
    :param new_width: New width of the GIF file.
    :param new_height: New height of the GIF file. Default is -1.
    :return: Resized GIF as binary data.
    """
    if isinstance(input_data, (str, os.PathLike)):
        input_path = pathlib.Path(input_data)
        assert input_path.exists(), FileNotFoundError(f"Input file {input_path} does not exist")
        with open(input_path, 'rb') as f:
            input_data = f.read()

    assert isinstance(input_data, (bytes, IO)), f"Invalid input_data type: {type(input_data)}"
    assert isinstance(new_width, int) and new_width >= -1, f"Invalid new width {new_width}"
    assert isinstance(new_height, int) and new_height >= -1, f"Invalid new height {new_height}"
    assert not (new_width == -1 and new_height == -1), "Both new width and new height cannot be -1"

    # 使用 wand 来调整 GIF 大小
    with Image(blob=input_data) as img:
        original_width, original_height = img.width, img.height
        if new_height == -1:
            new_height = int((new_width / original_width) * original_height)
        elif new_width == -1:
            new_width = int((new_height / original_height) * original_width)
        img.resize(new_width, new_height)
        resized_gif_data = img.make_blob(format='gif')
    assert resized_gif_data and len(resized_gif_data) > 0, "Failed to resize GIF"
    return resized_gif_data


bytes_out = resize_gif("300_ori.gif", 500)
with open("300_resized.gif", 'wb') as f:
    f.write(bytes_out)
