
import ctypes
import time

import numpy as np
import sys

# Load the compiled CUDA shared library.
lib = ctypes.CDLL("./libconvolution.so")

# Describe the NumPy array types expected by the C function.
ImageArray = np.ctypeslib.ndpointer(
    dtype=np.uint8,
    ndim=1,
    flags="C_CONTIGUOUS"
)

FilterArray = np.ctypeslib.ndpointer(
    dtype=np.float32,
    ndim=1,
    flags="C_CONTIGUOUS"
)

# Match the C function signature:
# int gpu_convolve(input, output, M, filter, N)
lib.gpu_convolve.argtypes = [
    ImageArray,
    ImageArray,
    ctypes.c_int,
    FilterArray,
    ctypes.c_int
]
lib.gpu_convolve.restype = ctypes.c_int


def convolve_cpu(image, kernel):
    """CPU reference with the same padding and output rules."""
    M = image.shape[0]
    N = kernel.shape[0]
    pad = N // 2

    padded = np.pad(image, pad, mode="constant")
    output = np.zeros((M, M), dtype=np.uint8)

    for y in range(M):
        for x in range(M):
            region = padded[y:y + N, x:x + N]

            # Flip the kernel for mathematical convolution.
            value = np.sum(
                region.astype(np.float32)
                * kernel[::-1, ::-1]
            )
            value = abs(float(value))
            value = min(255.0, max(0.0, value))

            output[y, x] = int(value + 0.5)

    return output


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    rng = np.random.default_rng(0)
    image = rng.integers(
        0, 256, size=(M, M), dtype=np.uint8
    )

    kernel = np.zeros((N, N), dtype=np.float32)
    center = N // 2
    kernel[center, center] = 4.0

    if center > 0:
        kernel[center - 1, center] = -1.0
        kernel[center + 1, center] = -1.0
        kernel[center, center - 1] = -1.0
        kernel[center, center + 1] = -1.0

    # Prepare contiguous, flattened arrays for ctypes.
    input_flat = np.ascontiguousarray(image.ravel())
    filter_flat = np.ascontiguousarray(kernel.ravel())
    output_flat = np.zeros(M * M, dtype=np.uint8)

    # First call may include one-time CUDA initialization overhead.
    status = lib.gpu_convolve(
        input_flat, output_flat, M, filter_flat, N
    )

    if status != 0:
        raise RuntimeError("CUDA convolution failed")

    # Measure repeated end-to-end calls.
    repetitions = 10
    start = time.perf_counter()

    for _ in range(repetitions):
        status = lib.gpu_convolve(
            input_flat, output_flat, M, filter_flat, N
        )
        if status != 0:
            raise RuntimeError("CUDA convolution failed")

    elapsed_ms = (
        (time.perf_counter() - start)
        * 1000.0 / repetitions
    )

    gpu_output = output_flat.reshape(M, M)

    if M <= 256:
        cpu_output = convolve_cpu(image, kernel)
        mismatches = np.count_nonzero(gpu_output != cpu_output)
        print(f"Verification: {'PASS' if mismatches == 0 else 'FAIL'}")
        print(f"Mismatched pixels: {mismatches}")
    else:
        mismatches = None
        print("CPU reference comparison skipped for this large test.")

    print(f"Image size: {M} x {M}")
    print(f"Filter size: {N} x {N}")
    print(f"Average Python-to-CUDA call: {elapsed_ms:.3f} ms")
    if mismatches is not None:
        print(f"Verification: {'PASS' if mismatches == 0 else 'FAIL'}")
        print(f"Mismatched pixels: {mismatches}")

    # Save the output for later inspection.
    with open("edge_python_output.pgm", "wb") as f:
        f.write(f"P5\n{M} {M}\n255\n".encode("ascii"))
        f.write(gpu_output.tobytes())

    print("Output saved to edge_python_output.pgm")


if __name__ == "__main__":
    main()