from pathlib import Path

from telegram_sticker_utils import TelegramStickerUtils

OUTPUT_DIR = Path(__file__).parent.joinpath("output")
OUTPUT_DIR.mkdir(exist_ok=True)

utils = TelegramStickerUtils()
GIF512 = Path(__file__).parent / "512_ori.gif"
bytes_io, suffix = utils.make_video_sticker(str(GIF512.absolute()), scale=512, master_edge="width")
assert suffix == ".webm", "Suffix should be .webm"
with OUTPUT_DIR.joinpath(GIF512.stem + suffix).open("wb") as f:
    f.write(bytes_io.getvalue())

GIF300 = Path(__file__).parent / "300_ori.gif"
bytes_io, suffix = utils.make_video_sticker(str(GIF300.absolute()), scale=512, master_edge="width")
assert suffix == ".webm", "Suffix should be .webm"
with OUTPUT_DIR.joinpath(GIF300.stem + suffix).open("wb") as f:
    f.write(bytes_io.getvalue())

PNG512 = Path(__file__).parent / "512_ori.png"
bytes_io, suffix = utils.make_static_sticker(str(PNG512.absolute()), scale=512, master_edge="width")
assert suffix == ".png", "Suffix should be .png"
with OUTPUT_DIR.joinpath(PNG512.stem + suffix).open("wb") as f:
    f.write(bytes_io.getvalue())

PNG300 = Path(__file__).parent / "300_ori.png"

bytes_io, suffix = utils.make_static_sticker(str(PNG300.absolute()), scale=512, master_edge="width")
assert suffix == ".png", "Suffix should be .png"
with OUTPUT_DIR.joinpath(PNG300.stem + suffix).open("wb") as f:
    f.write(bytes_io.getvalue())
