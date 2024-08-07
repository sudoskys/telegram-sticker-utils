import os
import pathlib
import tempfile
from dataclasses import dataclass
from io import BytesIO
from typing import Literal, Tuple
from typing import Union, IO

import wand.image as w_image
from PIL import Image as PilImage
from ffmpy import FFmpeg
from loguru import logger
from moviepy.video.io.VideoFileClip import VideoFileClip

from telegram_sticker_utils.core.const import get_random_emoji_from_text


def is_animated_gif(
        image: Union[str, bytes, os.PathLike, IO[bytes]]
) -> bool:
    """
    Check if an image is an animated GIF.
    :param image: Path to the image file or a file-like object.
    :return: True if the image is an animated GIF, False otherwise.
    :raises ValueError: If the image is not a valid GIF file.
    """
    # Load the image data
    if isinstance(image, (str, os.PathLike)):
        image_path = pathlib.Path(image)
        if not image_path.exists():
            raise FileNotFoundError(f"Input file {image_path} does not exist")
        with open(image_path, 'rb') as f:
            image_data = f.read()
    elif isinstance(image, IO):
        image_data = image.read()
    elif isinstance(image, bytes):
        image_data = image
    else:
        raise TypeError("image_path must be a string, bytes, os.PathLike, or file-like object")

    # Check animation using Wand
    try:
        from wand.image import Image as WImage  # noqa
        with WImage(blob=image_data) as img:
            return img.animation
    except ImportError:
        pass  # Wand is not available

    # Check animation using PIL
    try:
        from PIL import Image  # noqa
        with Image.open(BytesIO(image_data)) as img:
            try:
                img.seek(1)  # Try to move to the second frame
                return True
            except EOFError:
                return False
    except ImportError:
        pass  # PIL is not available

    raise ValueError("Unable to process the image file. Ensure the file is a valid GIF.")


@dataclass
class Sticker:
    data: bytes
    file_extension: str
    emojis: list[str]
    sticker_type: Union[Literal["static", "video"], str]


class ImageProcessor(object):

    @staticmethod
    def resize_image(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            new_width: int,
            new_height: int = -1,
            output_format: str = 'png'
    ) -> bytes:
        """
        Resize an image file using wand and optimize PNG if necessary.

        :param input_data: Path to the input image file or binary data.
        :param new_width: New width of the image file.
        :param new_height: New height of the image file. Default is -1.
        :param output_format: Output image format. Supported formats: 'gif', 'png'.
        :return: Resized image as binary data.
        """
        if isinstance(input_data, (str, os.PathLike)):
            input_path = pathlib.Path(input_data)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file {input_path} does not exist")
            with open(input_path, 'rb') as f:
                input_data = f.read()

        if isinstance(input_data, IO):
            input_data = input_data.read()

        if not isinstance(input_data, (bytes,)):
            raise TypeError(f"Invalid input_data type: {type(input_data)}")

        if new_width < -1:
            raise ValueError(f"Invalid new width {new_width}")

        if new_height < -1:
            raise ValueError(f"Invalid new height {new_height}")

        if new_width == -1 and new_height == -1:
            raise ValueError("Both new width and new height cannot be -1")

        design_formats = ['gif', 'png']
        if output_format not in design_formats:
            logger.warning(f"Unexpected output format: {output_format}")

        with w_image.Image(blob=input_data) as img:
            original_width, original_height = img.width, img.height

            if new_height == -1:
                new_height = int((new_width / original_width) * original_height)
            elif new_width == -1:
                new_width = int((new_height / original_height) * original_width)

            img.resize(new_width, new_height)
            resized_image_data = img.make_blob(format=output_format)

        if output_format == 'png':
            resized_image_data = ImageProcessor._optimize_png(resized_image_data)

        if not resized_image_data or len(resized_image_data) == 0:
            raise RuntimeError("Failed to resize image")

        return resized_image_data

    @staticmethod
    def resize_image_with_scale(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            scale: int,
            output_format: str = 'png'
    ) -> bytes:
        """
        Resize an image file using wand and ensure the longest side does not exceed the given scale.

        :param input_data: Path to the input image file or binary data.
        :param scale: Maximum length of the longest side of the image file.
        :param output_format: Output image format. Supported formats: 'gif', 'png'.
        :return: Resized image as binary data.
        """
        if scale <= 0:
            raise ValueError(f"Invalid scale value: {scale}. Scale must be positive.")

        if isinstance(input_data, (str, os.PathLike)):
            input_path = pathlib.Path(input_data)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file {input_path} does not exist")
            with open(input_path, 'rb') as f:
                input_data = f.read()

        if isinstance(input_data, IO):
            input_data = input_data.read()

        if not isinstance(input_data, bytes):
            raise TypeError(f"Invalid input_data type: {type(input_data)}")

        with w_image.Image(blob=input_data) as img:
            original_width, original_height = img.width, img.height

            if original_width > original_height:
                new_width = scale
                new_height = -1
            else:
                new_height = scale
                new_width = -1

        return ImageProcessor.resize_image(input_data, new_width, new_height, output_format)

    @staticmethod
    def _optimize_png(png_data: bytes) -> bytes:
        """
        Optimize PNG image to ensure its size is under 500kb.

        :param png_data: PNG image data.
        :return: Optimized PNG image data.
        """
        target_size = 500 * 1024  # 500kb in bytes
        quality = 100  # Start with the highest quality

        while len(png_data) > target_size and quality > 10:
            with w_image.Image(blob=png_data) as img:
                img.compression_quality = quality
                png_data = img.make_blob(format='png')

            quality -= 5  # Reduce quality stepwise

        if len(png_data) > target_size:
            raise RuntimeError("Failed to optimize PNG to be under 500kb")

        return png_data

    @staticmethod
    def convert_gif_to_png(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            scale: int
    ) -> bytes:
        """
        Convert a GIF file to a PNG file.

        :param input_data: Path to the input GIF file or binary data.
        :param scale: The desired maximum size for the longest side of the PNG file.
        :return: PNG file in binary form.
        :raises FileNotFoundError: If the input file does not exist.
        """
        if isinstance(input_data, (str, os.PathLike)):
            input_path = pathlib.Path(input_data)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file {input_path} does not exist")
            with open(input_path, 'rb') as f:
                input_data = f.read()
        elif isinstance(input_data, IO):
            input_data = input_data.read()

        with w_image.Image(blob=input_data) as img:
            original_width = img.width
            original_height = img.height

            # Compute the new size while maintaining aspect ratio
            if original_width > original_height:
                new_width = scale
                new_height = int(original_height * (scale / original_width))
            else:
                new_height = scale
                new_width = int(original_width * (scale / original_height))

            # Resize image
            img.resize(new_width, new_height)
            img.format = 'png'

            return img.make_blob()

    @staticmethod
    def convert_to_webm_ffmpeg(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            scale: int,
            *,
            frame_rate: Union[int, None] = None,
            duration: Union[int, None] = None
    ) -> bytes:
        """
        Convert image or video data to optimized WEBM format, resizing as necessary.

        :param input_data: Path to the input file or the input file data.
        :param scale: Desired maximum size for the longest side of the output video.
        :param frame_rate: Desired frame rate of the output video. If None, frame rate is not adjusted.
        :param duration: Desired duration of the output video. If None, duration is not adjusted.
        :return: Bytes of the optimized WEBM file.
        :raises FileNotFoundError: If the input file does not exist.
        :raises ValueError: If the encoded video exceeds 256 KB size limit.
        """

        def process_video(_input_path, _output_path, _scale, _frame_rate=None, _duration=None):
            output_options = [
                '-c:v', 'libvpx-vp9',  # VP9 codec for WEBM
                '-vf', f"scale={_scale}:-1",  # Scaling
                '-an',  # No audio stream
                '-loop', '1',  # Loop the video
                '-deadline', 'realtime',  # Speed/quality tradeoff setting
                '-b:v', '1M',  # Bitrate
                '-v', 'error',  # Silence ffmpeg output
            ]

            if _frame_rate is not None:
                output_options.extend(['-r', str(_frame_rate)])  # FPS setting

            if _duration is not None:
                output_options.extend(['-t', str(_duration)])

            ff = FFmpeg(
                inputs={_input_path: None},
                outputs={_output_path: output_options}
            )
            logger.debug(f"Calling ffmpeg command: {ff.cmd}")
            ff.run()

        # Create a temporary directory to hold the files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save input data to a temporary file if it is not already a path
            if isinstance(input_data, (str, os.PathLike)):
                input_path = pathlib.Path(input_data)
                if not input_path.exists():
                    raise FileNotFoundError(f"Input file {input_path} does not exist")
            else:
                with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_input_file:
                    temp_input_file.write(input_data)
                    input_path = temp_input_file.name

            # Initial temporary output file path
            output_path = os.path.join(temp_dir, "output_initial.webm")
            process_video(input_path, output_path, scale, frame_rate, duration)

            with open(output_path, 'rb') as output_file:
                optimized_webm = output_file.read()

            # Validate and adjust properties if needed
            video = VideoFileClip(output_path)

            if video.fps > 30 or video.duration > 3:
                adjusted_output_path = os.path.join(temp_dir, "output_adjusted.webm")
                frame_rate = 24 if video.fps > 30 else frame_rate
                duration = 2 if video.duration > 3 else duration
                logger.debug("Reprocessing video to fit requirements")
                process_video(input_path, adjusted_output_path, scale, frame_rate, duration)

                with open(adjusted_output_path, 'rb') as output_file:
                    optimized_webm = output_file.read()

            # Ensure the size does not exceed 256 KB
            if len(optimized_webm) > 256 * 1024:
                logger.warning("Encoded video exceeds 256 KB size limit")

            return optimized_webm

    @staticmethod
    def convert_to_webm(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            scale: int,
            *,
            strict: bool = True
    ) -> bytes:
        """
        Convert image or video data to optimized WEBM format, resizing as necessary.

        :param input_data: Path to the input file or the input file data.
        :param scale: Desired maximum size for the longest side of the output video.
        :param strict: Some images may have wrong metadata, set this to True to fall back to ffmpeg.
        :return: Bytes of the optimized WEBM file.
        :raises FileNotFoundError: If the input file does not exist.
        :raises ValueError: If the image dimensions change after optimization.
        """
        # Load input data
        if isinstance(input_data, (str, os.PathLike)):
            input_path = pathlib.Path(input_data)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file {input_path} does not exist")
            with open(input_path, 'rb') as file:
                input_data = file.read()

        with w_image.Image(blob=input_data) as img:
            # Compute the new size while maintaining aspect ratio
            if img.width > img.height:
                new_width = scale
                new_height = int(img.height * (scale / img.width))
            else:
                new_height = scale
                new_width = int(img.width * (scale / img.height))

            if img.format == "GIF":
                # Use Pillow to get the image dimensions
                with BytesIO(input_data) as img_byte_io:
                    pil_image = PilImage.open(img_byte_io)
                    pil_width, pil_height = pil_image.size
                    pil_image.close()
                    # Check if dimensions match between wand and Pillow
                if (img.width, img.height) != (pil_width, pil_height):
                    if strict:
                        # Use ffmpeg for conversion if dimensions do not match
                        return ImageProcessor.convert_to_webm_ffmpeg(input_data, scale)
                    raise ValueError(f"Image dimensions unknown error occurred")
            # Resize image/video
            img.transform(resize=f"{new_width}x{new_height}!")
            # Apply the optimizations
            # img.color_fuzz = "10%"
            # img.optimize_transparency()

            # Convert to WEBM with quality optimizations
            # img.options['webm:lossy'] = 'true'  # Use lossy compression for smaller size
            img.options['webm:method'] = '6'  # Method 6 provides good quality and compression

            if img.width != new_width or img.height != new_height:
                raise ValueError(f"Sticker Dimensions changed after optimization {img.width}x{img.height}")

            img.format = 'webm'
            optimized_blob = BytesIO()
            img.save(file=optimized_blob)
            optimized_blob.seek(0)
            sticker_data = optimized_blob.read()
            if len(sticker_data) > 256 * 1024:
                logger.warning("Encoded video exceeds 256 KB size limit")
            return sticker_data

    @staticmethod
    def make_raw_sticker(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            *,
            scale: int = 512,
            master_edge: Literal["width", "height"] = "width",
            strict: bool = True
    ) -> Tuple[bytes, str]:
        """
        Process the image. If the image is animated, convert it to WebM.
        If the image is static, resize it to the specified dimensions.

        :param input_data: Path to the input image file or binary data.
        :param scale: New size of the image file.
        :param master_edge: Which dimension (width or height) to scale
        :param strict: Some images may have wrong metadata, set this to True to fall back to ffmpeg.
        :return: Processed image as binary data.
        """
        if isinstance(input_data, (str, os.PathLike)):
            input_path = pathlib.Path(input_data)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file {input_path} does not exist")
            with open(input_path, 'rb') as f:
                input_data = f.read()
        elif isinstance(input_data, IO):
            input_data = input_data.read()

        with w_image.Image(blob=input_data) as img:
            if img.animation:
                # Convert to webm if image is animated
                if strict:
                    try:
                        return ImageProcessor.convert_to_webm_ffmpeg(input_data=input_data, scale=scale), "video"
                    except Exception as exc:
                        logger.error(f"ffmpeg error {exc}, using wand")
                        return ImageProcessor.convert_to_webm(input_data, scale=scale), "video"
                else:
                    return ImageProcessor.convert_to_webm(input_data, scale=scale), "video"
            else:
                # Convert to PNG if image is static
                if master_edge == "width":
                    return ImageProcessor.resize_image_with_scale(
                        input_data,
                        scale=scale,
                        output_format='png'
                    ), "static"
                else:
                    return ImageProcessor.resize_image_with_scale(
                        input_data,
                        scale=scale,
                        output_format='png'
                    ), "static"

    @staticmethod
    def make_sticker(
            input_name: str,
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            *,
            scale: int = 512,
            master_edge: Literal["width", "height"] = "width",
            strict: bool = True
    ) -> Sticker:
        """
        Process the image. If the image is animated, convert it to WebM.
        If the image is static, resize it to the specified dimensions.

        :param input_name: Name of the input image file.
        :param input_data: Path to the input image file or binary data.
        :param scale: New size of the image file.
        :param master_edge: Which dimension (width or height) to scale
        :param strict: Some images may have wrong metadata, set this to True to fall back to ffmpeg.
        :return: Processed image as binary data.
        """
        if isinstance(input_data, (str, os.PathLike)):
            input_path = pathlib.Path(input_data)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file {input_path} does not exist")
            with open(input_path, 'rb') as f:
                input_data = f.read()
        elif isinstance(input_data, IO):
            input_data = input_data.read()
        sticker_data, sticker_type = ImageProcessor.make_raw_sticker(
            input_data,
            scale=scale,
            master_edge=master_edge,
            strict=strict
        )
        emoji_item = [get_random_emoji_from_text(input_name)]
        file_extension = "png" if sticker_type == "static" else "webm"
        return Sticker(
            data=sticker_data,
            file_extension=file_extension,
            emojis=emoji_item,
            sticker_type=sticker_type
        )
