import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('./Lena.png')
    cv2.imwrite('./out/origin.png', img)
    # need gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('./out/gray_img.png', gray_img)
    # high-pass filter 30
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)

    rows, cols = gray_img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
    ishift = np.fft.ifftshift(fshift)
    iimg = np.abs(np.fft.ifft2(ishift))
    cv2.imwrite('./out/hp_30.png', iimg)

    # high-pass filter 1
    f = np.fft.fft2(gray_img)
    fshift = np.fft.fftshift(f)

    rows, cols = gray_img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    fshift[crow - 1:crow + 1, ccol - 1:ccol + 1] = 0
    ishift = np.fft.ifftshift(fshift)
    iimg = np.abs(np.fft.ifft2(ishift))
    cv2.imwrite('./out/hp_1.png', iimg)

    # low-pass filter 30
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    rows, cols = gray_img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    fi = fshift * mask
    ishift = np.fft.ifftshift(fi)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    cv2.imwrite('./out/lp_30.png', res)

    # low-pass filter 1
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    fshift = np.fft.fftshift(dft)

    rows, cols = gray_img.shape
    crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 1:crow + 1, ccol - 1:ccol + 1] = 1
    fi = fshift * mask
    ishift = np.fft.ifftshift(fi)
    iimg = cv2.idft(ishift)
    res = cv2.magnitude(iimg[:, :, 0], iimg[:, :, 1])
    cv2.imwrite('./out/lp_1.png', res)

