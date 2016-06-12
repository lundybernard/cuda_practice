import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer


#from numba import autojit

#@autojit
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

#@autojit
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

#create a 1024/1024 pixel image as a numpy array of bytes

image = np.zeros((2160, 4096), dtype = np.uint8)
start = timer()
create_fractal(-2.0, 1, -1, 1, image, 100)
dt = timer() - start

print("Python3: Mandelbrot created in %f s" % dt)
imshow(image)
show()
