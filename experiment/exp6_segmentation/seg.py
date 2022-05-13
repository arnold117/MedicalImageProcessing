import cv2
import numpy as np

if __name__ == '__main__':
    # original input
    img = cv2.imread('./Lena.png')
    cv2.imwrite('./out/origin.png', img)

    # need gray image not rgb
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('./out/gray_img.png', gray_img)

    # gauss smoothed
    gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    cv2.imwrite('./out/gaussian_img.png', gaussian_img)

    # roberts operator
    kernel_x = np.array(
        [
            [-1, 0],
            [0, 1]
        ],
        dtype=int
    )
    kernel_y = np.array(
        [
            [0, -1],
            [1, 0]
        ],
        dtype=int
    )
    x = cv2.filter2D(gaussian_img, cv2.CV_16S, kernel_x)
    y = cv2.filter2D(gaussian_img, cv2.CV_16S, kernel_y)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imwrite('./out/roberts.png', roberts)

    # prewitt operator
    kernel_x = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ],
        dtype=int
    )
    kernel_y = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ],
        dtype=int
    )
    x = cv2.filter2D(gaussian_img, cv2.CV_16S, kernel_x)
    y = cv2.filter2D(gaussian_img, cv2.CV_16S, kernel_y)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imwrite('./out/prewitt.png', prewitt)

    # sobel operator
    x = cv2.Sobel(gaussian_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gaussian_img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imwrite('./out/sobel.png', sobel)

    # laplace operator
    dst = cv2.Laplacian(gaussian_img, cv2.CV_16S, ksize=3)
    laplace = cv2.convertScaleAbs(dst)
    cv2.imwrite('./out/laplace.png', laplace)

    # canny operator
    canny = cv2.Canny(gaussian_img, 100, 200, 5)
    cv2.imwrite('./out/canny.png', canny)
