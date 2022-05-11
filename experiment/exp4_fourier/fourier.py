import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('./Lena.png')
    cv2.imwrite('./out/origin.png', img)
    # need gray image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('./out/gray_img.png', gray_img)
    # fft
    fshit = np.fft.fftshift(
        np.fft.fft2(gray_img)
    )
    fft_img = np.log(
        np.abs(
            fshit
        )
    )
    cv2.imwrite('./out/fft_res.png', fft_img)
    # ifft
    ifft_img = np.abs(
        np.fft.ifft2(
            np.fft.ifftshift(fshit)
        )
    )
    cv2.imwrite('./out/ifft_img.png', ifft_img)
