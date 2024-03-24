# executable='ffmpeg'
import shutil

FFMPEG_EXECUTABLE = 'ffmpeg'
IGNORE_CHECKS = False
PNGQUANT_EXECUTABLE = shutil.which("pngquant") or "pngquant"
