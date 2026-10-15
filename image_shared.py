import ctypes
import numpy as np
import time

# shared library
lib = ctypes.cdll.LoadLibrary("./image_cuda.so") 

# argument types
lib.convolution.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags="C_CONTIGUOUS"),
    ctypes.c_int,
    ctypes.c_int
]

# images
image_files = ["image1_512.raw", "image2_1024.raw", "image3_2048.raw"]
image_sizes = [512, 1024, 2048]

# filters
edge = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1], dtype=np.float32)

blur = np.array([
    1/256, 4/256, 6/256, 4/256, 1/256,
    4/256, 16/256, 24/256, 16/256, 4/256,
    6/256, 24/256, 36/256, 24/256, 6/256,
    4/256, 16/256, 24/256, 16/256, 4/256,
    1/256, 4/256, 6/256, 4/256, 1/256
], dtype=np.float32)

sharpen = np.array([
    -1, -2, -3, -4, -3, -2, -1,
    -2, -4, -6, -8, -6, -4, -2,
    -3, -6, -12, -16, -12, -6, -3,
    -4, -8, -16, 217, -16, -8, -4,
    -3, -6, -12, -16, -12, -6, -3,
    -2, -4, -6, -8, -6, -4, -2,
    -1, -2, -3, -4, -3, -2, -1
], dtype=np.float32)

filters = [edge, blur, sharpen]
filter_sizes = [3, 5, 7]
filter_names = ["Edge", "Blur", "Sharpen"]

# loop through each image
for img_idx in range(3):
    filename = image_files[img_idx]
    M = image_sizes[img_idx]
    pixels = M * M

    with open(filename, "rb") as f:
        img_bytes = f.read()

    h_image = np.frombuffer(img_bytes, dtype=np.uint8).copy()
    h_convoluted = np.zeros_like(h_image)

    # loop through each filter
    for f_idx in range(3):
        f_data = filters[f_idx]
        f_size = filter_sizes[f_idx]
        f_name = filter_names[f_idx]

        # call convolution function
        start_time = time.time()
        lib.convolution(h_image, h_convoluted, f_data, M, f_size)
        elapsed_ms = (time.time() - start_time) * 1000.0

        print(f"Image: {filename} (M={M}) | Filter: {f_name} ({f_size}x{f_size}) | Time: {elapsed_ms:.3f} ms")

        # save image
        out_filename = f"out_py_img{img_idx + 1}_{f_name}.pgm"
        with open(out_filename, "wb") as out:
            out.write(f"P5\n{M} {M}\n255\n".encode())
            out.write(h_convoluted.tobytes())

