import os
import pathlib
import tempfile
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from typing import Literal
from typing import Union, IO
import json
import subprocess

import wand.image as w_image
from PIL import Image as PilImage
from ffmpy import FFmpeg
from loguru import logger
from magika import Magika

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
    MAX_STATIC_IMAGE_SIZE_BYTES = 500 * 1024  # 500 KB

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
        """针对二次元风格图片的高质量缩放"""
        # 提前导入，避免重复导入
        from PIL.ImageEnhance import Sharpness
        
        input_buffer = BytesIO(input_data)
        with PilImage.open(input_buffer) as img:
            # 优化透明度处理
            if img.mode not in ('RGBA', 'LA'):
                img = img.convert('RGBA')
            
            # 计算新尺寸，确保最长边等于target_size
            current_w, current_h = img.size
            if current_w >= current_h:
                new_width = target_size
                new_height = int(round((target_size / current_w) * current_h))
            else:
                new_height = target_size
                new_width = int(round((target_size / current_h) * current_w))

            # 优化重采样策略
            if target_size <= 100:
                # 小图优化：直接使用高质量重采样
                resized = img.resize(
                    (new_width, new_height),
                    PilImage.Resampling.LANCZOS,
                    reducing_gap=2.0,
                    box=None  # 显式指定完整区域
                )
                # 更温和的锐化
                resized = Sharpness(resized).enhance(1.2)
            else:
                # 大图优化：渐进式重采样
                current_size = img.size
                steps = []
                
                # 计算渐进式缩放步骤
                while (current_size[0] / 1.5 > new_width or 
                       current_size[1] / 1.5 > new_height):
                    current_size = (
                        max(int(current_size[0] / 1.5), new_width),
                        max(int(current_size[1] / 1.5), new_height)
                    )
                    steps.append(current_size)
                
                # 渐进式重采样
                current = img
                for size in steps:
                    current = current.resize(
                        size,
                        PilImage.Resampling.BICUBIC,
                        reducing_gap=3.0
                    )
                
                # 最终重采样
                resized = current.resize(
                    (new_width, new_height),
                    PilImage.Resampling.LANCZOS,
                    reducing_gap=2.0
                )
                
                # 根据缩放比例调整锐化程度
                scale_ratio = min(new_width/img.size[0], new_height/img.size[1])
                sharpen_amount = 1.1 if scale_ratio < 0.5 else 1.05
                resized = Sharpness(resized).enhance(sharpen_amount)

            # 优化输出质量
            output = BytesIO()
            save_params = {
                'format': 'PNG',
                'optimize': True,
                'compress_level': 9,
                'bits': 8,
            }
            
            # 根据是否有透明通道优化保存参数
            if resized.mode == 'RGBA' and not any(resized.getchannel('A').getdata()):
                resized = resized.convert('RGB')
            
            resized.save(output, **save_params)
            initial_output_data = output.getvalue()
            if len(initial_output_data) > ImageProcessor.MAX_STATIC_IMAGE_SIZE_BYTES:
                optimized_data = ImageProcessor._optimize_png(initial_output_data)
                
                if len(optimized_data) > ImageProcessor.MAX_STATIC_IMAGE_SIZE_BYTES:
                     logger.warning(
                        f"Optimized image size is still too large: {len(optimized_data) / 1024:.2f}KB "
                        f"(limit: {ImageProcessor.MAX_STATIC_IMAGE_SIZE_BYTES / 1024:.2f}KB)."
                    )
                return optimized_data
            
            # 如果初始大小在限制内，直接返回
            return initial_output_data

    @staticmethod
    def _optimize_png(png_data: bytes) -> bytes:
        """优化PNG输出质量，主要用于后处理"""
        output = BytesIO()
        
        with PilImage.open(BytesIO(png_data)) as img:
            # 确保 RGBA 模式
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # 应用颜色量化，保持透明度
            if img.mode == 'RGBA':
                # 分离 alpha 通道
                rgb = img.convert('RGB')
                alpha = img.split()[3]
                
                # 对 RGB 通道进行量化
                quantized = rgb.quantize(colors=256, method=2)  # method=2 使用中位切分算法
                
                # 重新组合 alpha 通道
                quantized = quantized.convert('RGBA')
                quantized.putalpha(alpha)
                
                img = quantized
            
            img.save(
                output,
                format='PNG',
                optimize=True,
                compress_level=9,
                quality=95,
                bits=8
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
    def convert_to_webm_ffmpeg(
            input_data: Union[str, bytes, os.PathLike, IO[bytes]],
            scale: int,
            *,
            frame_rate: Union[int, None] = None,
            duration: Union[float, None] = None
    ) -> bytes:
        """
        将输入数据转换为优化的WebM格式动态贴纸。
        
        参数:
            input_data: 输入文件路径或二进制数据
            scale: 输出视频最长边的目标尺寸
            frame_rate: 可选的目标帧率，默认为原始帧率（最高30fps）
            duration: 可选的目标时长，默认为原始时长（最长2.9秒）
            
        返回:
            bytes: 优化后的WebM格式数据
            
        异常:
            BadInput: 当文件类型推断失败或视频处理失败时
        """
        try:
            file_type = mimetype_detector.identify_bytes(input_data).output.ct_label
        except Exception as exc:
            raise BadInput("Failed to infer file type") from exc

        with tempfile.TemporaryDirectory() as temp_dir:
            # 准备输入文件
            input_path = os.path.join(temp_dir, f"input.{file_type}")
            with open(input_path, 'wb') as f:
                f.write(input_data)

            # 获取输入视频信息
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration',
                '-of', 'json',
                input_path
            ]
            try:
                probe = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
                video_info = json.loads(probe.stdout)
                
                # 确保streams存在且不为空
                if 'streams' not in video_info or not video_info['streams']:
                    raise ValueError("No video streams found in the input file")
                    
                stream_info = video_info['streams'][0]
                
                # 计算原始帧率
                fps_num, fps_den = map(int, stream_info.get('r_frame_rate', '24/1').split('/'))
                original_fps = fps_num / fps_den
                
                # 智能帧率控制
                target_fps = min(original_fps, 30)
                if frame_rate:
                    target_fps = min(frame_rate, 30)
                
                # 智能时长控制
                orig_duration = float(stream_info.get('duration', '3'))
                target_duration = min(orig_duration, 2.9)
                if duration:
                    target_duration = min(duration, 2.9)
                    
                # 获取原始尺寸
                width = int(stream_info.get('width', 512))
                height = int(stream_info.get('height', 512))
            except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to get video info: {e}, using default values")
                target_fps = 24
                target_duration = 2.9
                width = height = 512

            output_path = os.path.join(temp_dir, "output.webm")
            
            # 计算缩放参数，确保最长边等于scale
            if width >= height:
                scale_filter = f"scale={scale}:-2:flags=lanczos"
            else:
                scale_filter = f"scale=-2:{scale}:flags=lanczos"
            
            # 基础编码参数
            base_options = [
                # 视频编码器设置
                '-c:v', 'libvpx-vp9',
                '-pix_fmt', 'yuva420p',
                # 尺寸控制 - 修正为确保最长边是scale
                '-vf', f"{scale_filter},setsar=1:1",
                # 移除音频
                '-an',
                # 循环设置
                '-loop', '0',
                # 质量控制
                '-deadline', 'good',
                '-cpu-used', '2',
                # 并行处理
                '-tile-columns', '2',
                '-frame-parallel', '1',
                '-auto-alt-ref', '1',
                '-lag-in-frames', '16',
                # 比特率控制
                '-b:v', '0',
                '-crf', '30',
                # 速度控制
                '-speed', '2',
                # 帧率控制
                '-r', f'{target_fps}',
                # 时长控制
                '-t', f'{target_duration}',
                # 元数据（用于欺骗服务器）
                '-metadata:s:v:0', 'alpha_mode="1"',
                '-metadata', f'duration="{target_duration}"',
                '-metadata', 'encoder="VP9 HW Encoder"',
                # 错误处理
                '-v', 'error'
            ]

            try:
                # 第一次编码尝试
                ff = FFmpeg(
                    global_options=['-y', '-hide_banner'],
                    inputs={input_path: None},
                    outputs={output_path: base_options}
                )
                ff.run()

                # 检查文件大小并优化
                if os.path.getsize(output_path) > WebmHelper.MAX_SIZE:
                    # 压缩配置序列
                    compression_configs = [
                        {'crf': 35, 'cpu-used': 2, 'speed': 2},
                        {'crf': 40, 'cpu-used': 3, 'speed': 3},
                        {'crf': 45, 'cpu-used': 4, 'speed': 4},
                        {'crf': 50, 'deadline': 'realtime', 'cpu-used': 4, 'speed': 4}
                    ]

                    for i, config in enumerate(compression_configs):
                        try:
                            options = base_options.copy()
                            # 更新压缩参数
                            for param, value in config.items():
                                if f'-{param}' in options:
                                    param_index = options.index(f'-{param}') + 1
                                    options[param_index] = str(value)
                                else:
                                    # 如果参数不存在，添加它
                                    options.extend([f'-{param}', str(value)])
                            
                            temp_output = os.path.join(temp_dir, f"output_compressed_{i}.webm")
                            ff = FFmpeg(
                                global_options=['-y', '-hide_banner'],
                                inputs={input_path: None},
                                outputs={temp_output: options}
                            )
                            ff.run()

                            if os.path.getsize(temp_output) <= WebmHelper.MAX_SIZE:
                                with open(temp_output, 'rb') as f:
                                    return f.read()
                        except Exception as e:
                            logger.warning(f"Compression config {config} failed: {e}")
                            continue

                    # 如果所有压缩配置都失败，但原始输出文件存在，返回它
                    if os.path.exists(output_path) and os.path.getsize(output_path) <= WebmHelper.MAX_SIZE:
                        logger.warning("Using original output as all compression attempts failed")
                        with open(output_path, 'rb') as f:
                            return f.read()
                            
                    raise BadInput("Failed to compress animated sticker within size limit")

                with open(output_path, 'rb') as f:
                    return f.read()
                    
            except Exception as e:
                logger.error(f"FFmpeg processing failed: {e}")
                raise BadInput(f"Video processing failed: {e}") from e

    @staticmethod
    def convert_to_webm_wand(
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
            # 计算新尺寸，确保最长边等于scale
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
                    raise ValueError("Image dimensions unknown error occurred")
                    
            # 使用高质量重采样方法
            img.resize(new_width, new_height, filter='lanczos')
            
            # 应用轻微锐化以提高清晰度
            img.sharpen(radius=0, sigma=0.8)

            # 优化透明度
            img.alpha_channel = True
            
            # 转换为WebM并设置高质量参数
            img.options['webm:method'] = '6'  # 高质量压缩方法
            img.options['webm:thread-level'] = '2'  # 并行处理
            
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
