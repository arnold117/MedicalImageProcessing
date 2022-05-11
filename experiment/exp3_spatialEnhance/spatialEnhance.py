import cv2
import numpy as np

if __name__ == '__main__':
    # original input
    img = cv2.imread('./Lena.png')
    cv2.imwrite('./out/origin.png', img)

    # dim by 50
    dim = cv2.subtract(
        img,
        np.ones(
            img.shape,
            dtype='uint8'
        ) * 50
    )
    cv2.imwrite('./out/dim50.png', dim)

    # logical not
    # need gray image not rgb
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('./out/gray_img.png', gray_img)
    gray_not = cv2.bitwise_not(img)
    cv2.imwrite('./out/gray_not.png', gray_not)

    # mean filtering
    mean_filtered = cv2.blur(gray_img, (10, 10))
    cv2.imwrite('./out/mean_filtered.png', mean_filtered)

    # medium filtering
    medium_filtered = cv2.medianBlur(gray_img, 3)
    cv2.imwrite('./out/medium_filtered.png', medium_filtered)

    # gauss filtering
    gauss_filtered = cv2.GaussianBlur(gray_img, (7, 7), 1.8)
    cv2.imwrite('./out/gauss_filtered.png', gauss_filtered)

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
    x = cv2.filter2D(gray_img, cv2.CV_16S, kernel_x)
    y = cv2.filter2D(gray_img, cv2.CV_16S, kernel_y)
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
    x = cv2.filter2D(gray_img, cv2.CV_16S, kernel_x)
    y = cv2.filter2D(gray_img, cv2.CV_16S, kernel_y)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imwrite('./out/prewitt.png', prewitt)

    # sobel operator
    x = cv2.Sobel(gray_img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(gray_img, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    cv2.imwrite('./out/sobel.png', sobel)

    # laplace operator
    dst = cv2.Laplacian(gray_img, cv2.CV_16S, ksize=3)
    laplace = cv2.convertScaleAbs(dst)
    cv2.imwrite('./out/laplace.png', sobel)
