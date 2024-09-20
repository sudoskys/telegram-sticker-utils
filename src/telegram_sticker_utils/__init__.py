import os
import pathlib
import tempfile
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Literal
from typing import Union, IO

import wand.image as w_image
from PIL import Image as PilImage
from ffmpy import FFmpeg
from loguru import logger
from magika import Magika
from moviepy.video.io.VideoFileClip import VideoFileClip

from telegram_sticker_utils.core.const import get_random_emoji_from_text

mimetype_detector = Magika()


class BadInput(Exception):
    pass


class StickerType(Enum):
    STATIC = "static"
    VIDEO = "video"


@dataclass
class Sticker:
    data: bytes
    file_extension: str
    emojis: list[str]
    sticker_type: Union[Literal["static", "video"], str]


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


class ImageProcessor(object):

    @staticmethod
    def _read_input_data(input_data: Union[str, bytes, os.PathLike, IO[bytes]]) -> bytes:
        """Helper function to read input data from different formats."""
        if isinstance(input_data, (str, os.PathLike)):
            input_path = pathlib.Path(input_data)
            if not input_path.exists():
                raise FileNotFoundError(f"Input file {input_path} does not exist")
            with open(input_path, 'rb') as f:
                return f.read()
        elif isinstance(input_data, IO):
            return input_data.read()
        if not isinstance(input_data, bytes):
            raise TypeError(f"Invalid input_data type: {type(input_data)}")
        return input_data

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
    def _resize_image(
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
        input_data = ImageProcessor._read_input_data(input_data)
        with w_image.Image(blob=input_data) as img:
            original_width, original_height = img.width, img.height
            if original_width > original_height:
                new_width = scale
                new_height = -1
            else:
                new_height = scale
                new_width = -1
        return ImageProcessor._resize_image(input_data, new_width, new_height, output_format)

    @staticmethod
    def _process_animated_image(input_data: bytes, scale: int) -> tuple[bytes, StickerType]:
        """Helper function to process animated images."""
        try:
            return WebmHelper.convert_to_webm_ffmpeg(input_data=input_data, scale=scale), StickerType.VIDEO
        except Exception as exc:
            logger.error(f"ffmpeg error {exc}, using wand")
            return WebmHelper.convert_to_webm_wand(input_data, scale=scale), StickerType.VIDEO

    @staticmethod
    def _resize_static_image(input_data: bytes, scale: int) -> tuple[bytes, StickerType]:
        """Helper function to resize static images."""
        return ImageProcessor.resize_image_with_scale(
            input_data,
            scale=scale,
            output_format='png'
        ), StickerType.STATIC

    @staticmethod
    def make_raw_sticker(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            *,
            scale: int = 512
    ) -> tuple[bytes, StickerType]:
        """
        Process the image. If the image is animated, convert it to WebM.
        If the image is static, resize it to the specified dimensions.

        :param input_data: Path to the input image file or binary data.
        :param scale: New size of the image file.
        :return: Processed image as binary data.
        """
        input_data = ImageProcessor._read_input_data(input_data)
        file_type = mimetype_detector.identify_bytes(input_data).output.ct_label
        if file_type == "webm":
            return ImageProcessor._process_animated_image(input_data, scale)

        if file_type in ["gif"]:
            with w_image.Image(blob=input_data) as img:
                if img.animation:
                    return ImageProcessor._process_animated_image(input_data, scale)
                return ImageProcessor._resize_static_image(input_data, scale)

        if file_type in ["png", "jpeg", "jpg"]:
            return ImageProcessor._resize_static_image(input_data, scale)

        try:
            with w_image.Image(blob=input_data) as img:
                if img.animation:
                    return ImageProcessor._process_animated_image(input_data, scale)
                return ImageProcessor._resize_static_image(input_data, scale)
        except Exception as exc:
            logger.warning(f"Unsupported file type: {file_type}")
            raise BadInput(
                f"An Error happened!Unsupported file type @{file_type}."
                f"If you believe this is an error, please report it at "
                f"https://github.com/sudoskys/telegram-sticker-utils/issues/new"
            ) from exc

    @staticmethod
    def make_sticker(
            input_name: str,
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            *,
            scale: int = 512,
            **kwargs
    ) -> Sticker:
        """
        Process the image. If the image is animated, convert it to WebM.
        If the image is static, resize it to the specified dimensions.

        :param input_name: Name of the input image file.
        :param input_data: Path to the input image file or binary data.
        :param scale: New size of the image file.
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

        # Process the image
        sticker_data, sticker_type = ImageProcessor.make_raw_sticker(
            input_data,
            scale=scale
        )
        # Get random emoji from the input name
        emoji_item = [get_random_emoji_from_text(input_name)]
        # Output file extension
        file_extension = "png" if sticker_type == "static" else "webm"
        return Sticker(
            data=sticker_data,
            file_extension=file_extension,
            emojis=emoji_item,
            sticker_type=sticker_type.value
        )


class WebmHelper(object):
    MAX_SIZE = 256 * 1024  # 256 KB

    @staticmethod
    def _optimize_webm(
            webm_data: bytes,
            scale: int,
            origin_file_type: str
    ) -> bytes:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Try different compression levels if necessary
            input_temp_path = os.path.join(temp_dir, "current_input.webm")
            output_temp_path = os.path.join(temp_dir, "current_output.webm")
            # If the initial webm data is already under 256 KB, return it
            if len(webm_data) <= WebmHelper.MAX_SIZE:
                return webm_data
            # Write the initial webm data to a file
            with open(input_temp_path, 'wb') as temp_input_file:
                temp_input_file.write(webm_data)

            crf_values = [30, 40, 45, 50, 55, 60]
            while len(webm_data) > WebmHelper.MAX_SIZE and crf_values:
                crf = crf_values.pop(0)
                WebmHelper.process_video(
                    input_temp_path,
                    output_temp_path,
                    scale=scale,
                    crf=crf,
                    file_type=origin_file_type
                )
                with open(output_temp_path, 'rb') as output_file:
                    webm_data = output_file.read()
        if len(webm_data) > WebmHelper.MAX_SIZE:
            raise BadInput("Encoded video exceeds 256 KB size limit even after optimization")
        return webm_data

    @staticmethod
    def process_video(input_path, output_path, scale, file_type: str, frame_rate=None, duration=None, crf=None):
        output_options = [
            '-c:v', 'libvpx-vp9',  # VP9 codec for WEBM
            # '-pix_fmt', 'yuva420p',  # Pixel format
            '-vf', f"scale={scale}:-1",  # Scaling
            '-an',  # No audio stream
            '-loop', '1',  # Loop the video
            '-deadline', 'realtime',  # Speed/quality tradeoff setting
            '-b:v', '1M',  # Bitrate
            '-v', 'error',  # Silence ffmpeg output
        ]
        if file_type == "webm":
            input_options = [
                '-c:v', 'libvpx-vp9',  # VP9 codec for WEBM
            ]
        else:
            input_options = []
        if frame_rate is not None:
            output_options.extend(['-r', str(frame_rate)])  # FPS setting

        if duration is not None:
            output_options.extend(['-t', str(duration)])  # Duration setting

        if crf is not None:
            output_options.extend(['-crf', str(crf)])  # Constant Rate Factor

        ff = FFmpeg(
            global_options=['-y'],  # Overwrite output file if it exists
            inputs={input_path: input_options},
            outputs={output_path: output_options}
        )
        logger.trace(f"Calling ffmpeg command: {ff.cmd}")
        ff.run()

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
        try:
            file_type = mimetype_detector.identify_bytes(input_data).output.ct_label
        except Exception as exc:
            raise BadInput("Failed to infer file type") from exc
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

            # Process video and optimize
            output_path = os.path.join(temp_dir, "output_initial.webm")
            WebmHelper.process_video(
                input_path=input_path,
                output_path=output_path,
                scale=scale,
                file_type=file_type,
                frame_rate=frame_rate,
                duration=duration
            )

            with open(output_path, 'rb') as output_file:
                optimized_webm = output_file.read()

            # Validate and adjust properties if needed
            video = VideoFileClip(output_path)

            if video.fps > 30 or video.duration > 3:
                adjusted_output_path = os.path.join(temp_dir, "output_adjusted.webm")
                frame_rate = 24 if video.fps > 30 else frame_rate
                duration = 2 if video.duration > 3 else duration
                logger.trace("Reprocessing video to fit requirements")
                WebmHelper.process_video(
                    input_path=input_path,
                    output_path=adjusted_output_path,
                    scale=scale,
                    file_type=file_type,
                    frame_rate=frame_rate,
                    duration=duration
                )
                if not os.path.exists(adjusted_output_path):
                    raise FileNotFoundError("Failed to create adjusted video")
                with open(adjusted_output_path, 'rb') as output_file:
                    optimized_webm = output_file.read()

            # Optimize the WEBM file to be under 256 KB
            optimized_webm = WebmHelper._optimize_webm(
                optimized_webm,
                scale=scale,
                origin_file_type=file_type
            )

            # Ensure the size does not exceed 256 KB
            if len(optimized_webm) > 256 * 1024:
                raise BadInput(
                    "Encoded video exceeds 256 KB size limit"
                    "But Telegram thinks it's too big!"
                    "Please check this file."
                )
            return optimized_webm

    @staticmethod
    def convert_to_webm_wand(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            scale: int,
            *,
            strict: bool = True
    ) -> bytes:
        """
        Convert image or video data to optimized WEBM format, resizing as necessary.
        !!!Warning: This method may cause the GIF size to be distorted!!!

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
                        return WebmHelper.convert_to_webm_ffmpeg(input_data, scale)
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
                raise BadInput(
                    "Encoded video exceeds 256 KB size limit"
                    "But Telegram thinks it's too big!"
                    "Please check this file."
                )
            return sticker_data
