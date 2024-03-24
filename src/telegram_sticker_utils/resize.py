import os
import pathlib
import tempfile
from io import BytesIO
from typing import Literal, Tuple

import ffmpy
import pngquant
from loguru import logger
from PIL import Image

from .custom import FFMPEG_EXECUTABLE, PNGQUANT_EXECUTABLE

pngquant.config(PNGQUANT_EXECUTABLE)


class TelegramStickerUtils(object):
    @staticmethod
    def is_animated_gif(image_path: str) -> bool:
        """
        Check if an image is an animated GIF.
        :param image_path:
        :return:
        """
        assert os.path.exists(image_path), FileNotFoundError(f"Input File {image_path} does not exist")
        gif = Image.open(image_path)
        try:
            gif.seek(1)
        except EOFError:
            return False
        else:
            return True

    @staticmethod
    def resize_gif(input_path: str, output_path: str, new_width: int, new_height: int = -1) -> pathlib.Path:
        """
        Resize a GIF file using ffmpeg.
        :param input_path:  Path to the input GIF file.
        :param output_path: Path to the output GIF file.
        :param new_width:  New width of the GIF file.
        :param new_height: New height of the GIF file. Default is -1.
        :return:  output_path
        """
        assert os.path.exists(input_path), FileNotFoundError(f"Input File {input_path} does not exist")
        assert new_width >= -1, f"Invalid new width {new_width}"
        assert new_height >= -1, f"Invalid new height {new_height}"
        assert not (new_width == -1 and new_height == -1), "Both new width and new height cannot be -1"
        # 使用ffmpy调整GIF大小
        ff = ffmpy.FFmpeg(
            executable=FFMPEG_EXECUTABLE,
            inputs={
                input_path: None
            },
            outputs={
                output_path:
                    '-vf "scale={}:{},split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -y -loglevel 0'.format(
                        new_width, new_height)}
        )
        ff.run()
        assert os.path.exists(output_path), f"Failed to create {output_path}"
        assert os.path.getsize(output_path) > 1, f"File {output_path} is empty"
        return pathlib.Path(output_path)

    @staticmethod
    def resize_png(input_path: str, output_path: str, new_width: int, new_height: int = -1) -> pathlib.Path:
        """
        Resize an image file using ffmpeg.
        :param input_path:  Path to the input image file.
        :param output_path:  Path to the output image file.
        :param new_width:  New width of the image file.
        :param new_height:  New height of the image file. Default is -1.
        :return: output_path
        """
        assert os.path.exists(input_path), FileNotFoundError(f"Input File {input_path} does not exist")
        assert new_width >= -1, f"Invalid new width {new_width}"
        assert new_height >= -1, f"Invalid new height {new_height}"
        assert not (new_width == -1 and new_height == -1), "Both new width and new height cannot be -1"
        scale = f'{new_width}:{new_height}'
        try:
            ff = ffmpy.FFmpeg(
                executable=FFMPEG_EXECUTABLE,
                inputs={input_path: None},
                outputs={output_path: f'-vf scale={scale} -y -loglevel 0'}
            )
            ff.run()
        except Exception as e:
            if "234" not in str(e):
                raise e
        assert os.path.exists(output_path), f"Failed to create {output_path}"
        assert os.path.getsize(output_path) > 1, f"File {output_path} is empty"
        return pathlib.Path(output_path)

    @staticmethod
    def convert_gif_png(input_path: str, output_path: str, width: int, height: int) -> pathlib.Path:
        """
        Convert a GIF file to a PNG file.
        :param input_path:  Path to the input GIF file.
        :param output_path:  Path to the output PNG file.
        :param width:  Width of the PNG file.
        :param height:  Height of the PNG file.
        :return:
        """
        assert os.path.exists(input_path), FileNotFoundError(f"Input File {input_path} does not exist")
        if os.path.exists(output_path):
            logger.warning(f"Output File {output_path} already exists, overwriting")
        assert not (width == -1 and height == -1), "Both new width and new height cannot be -1"
        img = Image.open(input_path)
        # 将 -1 调整为比例尺寸
        if width == -1:
            width = int(img.width * height / img.height)
        if height == -1:
            height = int(img.height * width / img.width)
        # 将 GIF 转换为 PNG
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        img.save(output_path, 'PNG')
        assert os.path.exists(output_path), f"Failed to create {output_path}"
        assert os.path.getsize(output_path) > 1, f"File {output_path} is empty"
        return pathlib.Path(output_path)

    @staticmethod
    def convert_gif_webm(
            input_path,
            sec: float = 2.9,
            bit_rate: str = '2000k',
            crf: int = 10,
            cpu_used: int = 2) -> BytesIO:
        """
        Convert a GIF file to a WebM file.
        :param input_path: Path to the input GIF file.
        :param sec: Duration of the WebM file. Default is 2.9 seconds.
        :param bit_rate: Bit rate of the WebM file. Default is 2000k.
        :param crf: Constant Rate Factor of the WebM file. Default is 10.
        :param cpu_used: CPU used of the WebM file. Default is 2.
        :return: BytesIO
        """
        assert os.path.exists(input_path), FileNotFoundError(f"The file {input_path} doesn't exist")
        assert sec > 0, ValueError("Duration must be greater than 0")

        attempts = 0
        bit_rates = [bit_rate, '1000k', '500k', '320k']  # List of bit_rates to try

        while sec > 1 and attempts < 10:
            for br in bit_rates:
                with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp:
                    output_filename = temp.name
                    output = {
                        output_filename: f'-c:v libvpx-vp9 -b:v {br} -crf {crf} -cpu-used {cpu_used} -an -pix_fmt yuva420p -auto-alt-ref 0 -metadata:s:v:0 alpha_mode="1"'
                    }
                    ff = ffmpy.FFmpeg(
                        executable=FFMPEG_EXECUTABLE,
                        inputs={input_path: f'-y -loglevel 0 -ss 00:00:0.0 -t {sec}'},
                        outputs=output
                    )
                    ff.run()

                    _size = os.path.getsize(output_filename)
                    if _size <= 256 * 1024:
                        with open(output_filename, 'rb') as f:
                            output_bytes = BytesIO(f.read())
                            pathlib.Path(output_filename).unlink(missing_ok=True)
                            return output_bytes
                    pathlib.Path(output_filename).unlink(missing_ok=True)
            sec = round(sec - 0.5, 1)
            logger.warning(f"File size {_size} exceeds 256 KB limit, retry with {sec} seconds")
            attempts += 1

        logger.error(f"File Size {_size}")
        raise ValueError(f'Failed to generate a video: size exceeds 256 KB limit after {attempts} attempts.')

    def make_static_sticker(self,
                            input_path: str,
                            *,
                            scale: int = 512,
                            master_edge: Literal["width", "height"] = "width",
                            size_limit=500 * 1024,
                            max_iterations=3
                            ) -> Tuple[BytesIO, str]:
        """
        Resize a PNG file and optimize it using pngquant.
        :param input_path:  Path to the input PNG file.
        :param scale:  New width or height of the PNG file.
        :param master_edge:  Master edge to scale. Default is "width".
        :param size_limit:  Maximum size of the PNG file in bytes. Default is 500 KB.
        :param max_iterations:  Maximum number of iterations to optimize the PNG file. Default is 3.
        :return: BytesIO, File Extension
        """
        # Check if the input file exists
        assert os.path.exists(input_path), f"Input File {input_path} does not exist"
        # Check if the scale is valid
        assert scale > 0, f"Invalid scale {scale}"
        # Check if the file size is within the limit
        assert os.path.getsize(input_path) <= size_limit * 25, f"File {input_path} too large"
        # Calculate the new width and height
        new_width, new_height = (scale, -1) if master_edge == "width" else (-1, scale)
        # Get the file extension
        suffix = pathlib.Path(input_path).suffix
        # Ensure the input is not an animated GIF
        if self.is_animated_gif(input_path):
            raise ValueError("Input cannot be an animated GIF")
        # Temporary file to store the output
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            output_path = temp.name
            self.resize_png(input_path, output_path, new_width, new_height)
            for i in range(max_iterations):
                filesize = os.path.getsize(output_path)
                if filesize <= size_limit:
                    break
                logger.debug(f"Optimizing {input_path} PNG FILESIZE iteration {i + 1}")
                try:
                    codes, bytes_image = pngquant.quant_image(output_path)
                except Exception as e:
                    logger.error(f"Error while zip PNG using pngquant: {e}")
                    raise e
                with open(output_path, 'wb') as f:
                    f.write(bytes_image)
            with open(output_path, 'rb') as f:
                bytes_io = BytesIO(f.read())
        os.remove(output_path)
        return bytes_io, suffix

    def make_video_sticker(self,
                           input_path: str,
                           *,
                           scale: int = 512,
                           master_edge: Literal["width", "height"] = "width",
                           size_limit=256 * 1024,
                           ) -> Tuple[BytesIO, str]:
        """
        Resize a gif/png file and optimize it using pngquant.
        :param input_path:  Path to the input video file.
        :param scale:  New width or height of the video file.
        :param master_edge:  Master edge to scale. Default is "width".
        :param size_limit:  Maximum size of the video file in bytes. Default is 256 KB.
        :return: BytesIO, File Extension
        """
        # Check if the input file exists
        assert os.path.exists(input_path), f"Input File {input_path} does not exist"
        # Check if the scale is valid
        assert scale > 0, f"Invalid scale {scale}"
        # Check if the file size is within the limit
        assert os.path.getsize(input_path) <= size_limit * 25, f"File {input_path} too large"
        # Calculate the new width and height
        new_width, new_height = (scale, -1) if master_edge == "width" else (-1, scale)
        # 判断input 是不是真的 GIF
        if not self.is_animated_gif(input_path):
            logger.debug(f"Input File {input_path} is not a GIF")
            # 转换为 PNG，同时变换input_path
            input_path = self.convert_gif_png(input_path, input_path, new_width, new_height)
            input_path = str(input_path)
        # Get the file extension
        suffix = pathlib.Path(input_path).suffix
        # Temporary file to store the output
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            output_path = temp.name
            if suffix == '.gif':
                self.resize_gif(input_path, output_path, new_width, new_height)
            else:
                self.resize_png(input_path, output_path, new_width, new_height)
            if suffix in ['.gif', '.png']:
                # 转换为 WebM
                bytes_io = self.convert_gif_webm(output_path)
            else:
                raise ValueError(f"Unsupported file type {suffix}")
        os.remove(output_path)
        return bytes_io, ".webm"
