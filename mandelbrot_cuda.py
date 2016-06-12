#! /home/locky/miniconda3/envs/cuda/bin/python

import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer


from numbapro import cuda
from numba import *


@autojit
def mandel(x, y, max_iters):
    """Given the real and imaginary parts of a complex number,
    determine if it is a candidate for membership in the mandelbrot
    set, given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real*z.real + z.imag*z.imag) >= 4:
            return i

    return max_iters

@autojit
def create_fractal (min_x, max_x, min_y, max_y, image, iters):
    """iterates over all the pixels in the image, computing the
    complex coordinates from the pixel coordinates and
    calls the mandel function at each pixel.
    the return value of mandel is used to color the pixel
    """
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color


mandel_gpu = cuda.jit(restype=uint32, argtypes=[f8, f8, uint32], device=True)(mandel)

@cuda.jit(argtypes=[f8, f8, f8, f8, uint8[:,:], uint32])
def mandel_kernel (min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) /width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX = cuda.gridDim.x * cuda.blockDim.x
    gridY = cuda.gridDim.y * cuda.blockDim.y

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            image[y, x] = mandel_gpu(real, imag, iters)



#create a 1024/1024 pixel image as a numpy array of bytes

gimage = np.zeros((30000, 20000), dtype = np.uint8)
blockdim = (32, 8)
griddim = (32, 16)


start = timer()
d_image = cuda.to_device(gimage)
mandel_kernel[griddim, blockdim](-2.0, 1, -1, 1, d_image, 1000)
d_image.to_host()
dt = timer() - start

print("Cuda: Mandelbrot created in %f s" % dt)
imshow(gimage)
show()
