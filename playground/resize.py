import os
import pathlib
from io import BytesIO
from typing import Union, IO

from wand.image import Image


def convert_gif_to_webm(
        input_data: Union[str, bytes, os.PathLike, IO[bytes]],
        new_width: int,
        new_height: int = -1
) -> bytes:
    if isinstance(input_data, (str, os.PathLike)):
        input_path = pathlib.Path(input_data)
        assert input_path.exists(), FileNotFoundError(f"Input file {input_path} does not exist")
        with open(input_path, 'rb') as file:
            input_data = file.read()

    with Image(blob=input_data) as img:
        img.transform(resize=f"{new_width}x{new_height}") if new_height > 0 else img.transform(resize=f"{new_width}")
        # Apply the optimizations
        img.optimize_layers()
        img.color_fuzz = "10%"
        img.optimize_transparency()
        # Convert to WEBM
        img.options['webm:cq'] = '30'  # Optimize for quality, change value as needed (0-63)
        img.format = 'webm'
        optimized_blob = BytesIO()
        img.save(file=optimized_blob)
        optimized_blob.seek(0)
        return optimized_blob.read()


bytes_out = convert_gif_to_webm("300_resized.png", 512)
with open("301png.webm", 'wb') as f:
    f.write(bytes_out)
