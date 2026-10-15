import numpy as np
from PIL import Image

configurations = [
    ("image1.jpg", 512,  "image1_512.raw"),
    ("image2.jpg", 1024, "image2_1024.raw"),
    ("image3.jpg", 2048, "image3_2048.raw")
]

for src_path, target_size, out_path in configurations:
    img = Image.open(src_path).convert("L")
    img_resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)
    arr = np.array(img_resized, dtype=np.uint8)
    arr.tofile(out_path)
