import os
from PIL import Image

img_dir = r".\\0"
for filename in os.listdir(img_dir):
    filepath = os.path.join(img_dir, filename)
    with Image.open(filepath) as im:
        x, y = im.size
    if (x < 100 and y < 100):
        os.remove(filepath)
