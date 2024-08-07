from telegram_sticker_utils import ImageProcessor

sticker = ImageProcessor.make_sticker(
    input_name='sad',
    input_data=open("报警.gif", 'rb').read(),
    scale=512,
    master_edge='width',
)
print(sticker.sticker_type)
print(sticker.emojis)
with open(f"output.{sticker.file_extension}", 'wb') as f:
    f.write(sticker.data)
