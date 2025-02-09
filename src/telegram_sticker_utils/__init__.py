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
    def _resize_image(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            target_size: int,
            output_format: str = 'png'
    ) -> bytes:
        """针对数字插画的简单高质量缩放"""
        with w_image.Image(blob=input_data) as img:
            # 计算目标尺寸
            current_w, current_h = img.width, img.height

            if current_w >= current_h:
                new_width = target_size
                new_height = int(round((target_size / current_w) * current_h))
            else:
                new_height = target_size
                new_width = int(round((target_size / current_h) * current_w))

            # 缩放图像
            img.resize(new_width, new_height, filter='lanczos')
            img.unsharp_mask(0.5, 0.6, 0.9, 0.02)  # 轻微锐化

            # 如果是PNG，进行基础的颜色优化
            if output_format == 'png':
                img.quantize(256, 'srgb', dither=True)

            return img.make_blob(format=output_format)

    @staticmethod
    def _optimize_png(png_data: bytes) -> bytes:
        """简化的PNG优化，专注于关键参数"""
        output = BytesIO()

        with PilImage.open(BytesIO(png_data)) as img:
            img.save(
                output,
                format='PNG',
                optimize=True,
                compress_level=9,
                bits=8,
                params={
                    'filter_type': 4,  # Paeth
                    'strategy': 3  # Z_RLE
                }
            )

        output.seek(0)
        return output.read()

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
        return ImageProcessor._resize_image(input_data, scale, output_format)

    @staticmethod
    def _process_animated_image(input_data: bytes, scale: int) -> tuple[bytes, StickerType]:
        """Helper function to process animated images."""
        try:
            return WebmHelper.convert_to_webm_ffmpeg(input_data=input_data, scale=scale), StickerType.VIDEO
        except Exception as exc:
            logger.error(f"ffmpeg report error {exc}\ntry to using wand instead")
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
        if file_type in ["webm", "mp4", "mov", "avi"]:
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
        file_extension = "png" if sticker_type == StickerType.STATIC else "webm"
        return Sticker(
            data=sticker_data,
            file_extension=file_extension,
            emojis=emoji_item,
            sticker_type=sticker_type.value
        )


class WebmHelper(object):
    MAX_SIZE = 256 * 1024  # 256 KB

    @staticmethod
    def _optimize_webm(webm_data: bytes, scale: int) -> bytes:
        """优化 WEBM 编码的高级算法"""
        if len(webm_data) <= WebmHelper.MAX_SIZE:
            return webm_data

        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = os.path.join(temp_dir, "input.webm")
            output_path = os.path.join(temp_dir, "output.webm")

            with open(input_path, 'wb') as f:
                f.write(webm_data)

            # 智能压缩参数序列
            compression_profiles = [
                # crf, deadline, cpu-used, tile-columns
                (30, 'good', 1, 2),  # 高质量尝试
                (38, 'good', 2, 2),  # 平衡模式
                (45, 'realtime', 3, 1),  # 压缩模式
                (52, 'realtime', 4, 1),  # 极限压缩
            ]

            for crf, deadline, cpu_used, tile_columns in compression_profiles:
                output_options = [
                    '-c:v', 'libvpx-vp9',
                    '-pix_fmt', 'yuva420p',
                    '-vf', f"scale='if(gt(iw,ih),{scale},-1)':'if(gt(iw,ih),-1,{scale})'",
                    '-an',
                    '-loop', '1',
                    '-deadline', deadline,
                    '-cpu-used', str(cpu_used),
                    '-tile-columns', str(tile_columns),
                    '-frame-parallel', '1',
                    '-auto-alt-ref', '0',
                    '-lag-in-frames', '0',
                    '-b:v', '0',
                    '-crf', str(crf),
                    # 欺骗服务器的关键参数
                    '-metadata:s:v:0', 'alpha_mode="1"',
                    '-metadata', 'duration="2.9"',  # 伪装持续时间
                    '-metadata', 'encoder="VP9 HW Encoder"'  # 伪装编码器
                ]

                try:
                    ff = FFmpeg(
                        global_options=['-y'],
                        inputs={input_path: ['-c:v', 'libvpx-vp9']},
                        outputs={output_path: output_options}
                    )
                    ff.run(quiet=True)

                    with open(output_path, 'rb') as f:
                        optimized_data = f.read()

                    if len(optimized_data) <= WebmHelper.MAX_SIZE:
                        return optimized_data

                except Exception as e:
                    logger.warning(f"Compression profile failed: {e}")
                    continue

        raise BadInput("Unable to optimize WEBM within size limit")

    @staticmethod
    def process_video(input_path, output_path, scale, input_file_type: str,
                      frame_rate=None, duration=None, crf=None):
        """改进的视频处理方法"""
        output_options = [
            '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p',
            '-vf', (f"scale='if(gt(iw,ih),{scale},-1)':'if(gt(iw,ih),-1,{scale})',"
                    "setsar=1:1,fps=fps=24"),  # 统一帧率
            '-an',
            '-loop', '1',
            '-deadline', 'good',
            '-cpu-used', '2',
            '-tile-columns', '2',
            '-frame-parallel', '1',
            '-lag-in-frames', '0',
            '-b:v', '0',
            '-v', 'error',
            # 高级元数据控制
            '-metadata:s:v:0', 'alpha_mode="1"',
            '-metadata', 'duration="2.9"',
            '-metadata', 'encoder="VP9 HW Encoder"',
        ]

        if frame_rate:
            output_options.extend(['-r', str(min(frame_rate, 30))])

        if duration:
            # 智能时长控制
            actual_duration = min(float(duration), 2.9)
            output_options.extend(['-t', str(actual_duration)])

        if crf:
            output_options.extend(['-crf', str(crf)])

        input_options = ['-c:v', 'libvpx-vp9'] if input_file_type == "webm" else []

        ff = FFmpeg(
            global_options=['-y'],
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
                input_file_type=file_type,
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
                    input_file_type=file_type,
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
            )

            # Ensure the size does not exceed 256 KB
            if len(optimized_webm) > 256 * 1024:
                raise BadInput(
                    "Encoded video exceeds 256 KB size limit when using ffmpeg, "
                    "but Telegram thinks it's too big! "
                    "Please check this file"
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
                    "Encoded video exceeds 256 KB size limit when using wind, "
                    "but Telegram thinks it's too big! "
                    "Please check this file"
                )
            return sticker_data
