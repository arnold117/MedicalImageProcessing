import cv2
import numpy as np


def sampling(read_file, digits, out_dir, out_name):
    img = cv2.imread(read_file)
    msg = out_dir + 'sampling_origin.png'
    cv2.imwrite(msg, img)

    height = img.shape[0]
    width = img.shape[1]

    height_dig = int(height / digits)
    width_dig = int(width / digits)
    new_img = np.zeros((height, width, 3), np.uint8)

    for i in range(digits):
        y = i * height_dig
        for j in range(digits):
            x = j * width_dig

            b = img[y, x][0]
            g = img[y, x][1]
            r = img[y, x][2]

            for n in range(height_dig):
                for m in range(width_dig):
                    new_img[y + n, x + m][0] = np.uint8(b)
                    new_img[y + n, x + m][1] = np.uint8(g)
                    new_img[y + n, x + m][2] = np.uint8(r)
    msg = out_dir + out_name
    cv2.imwrite(msg, new_img)
    return new_img


def quantize(read_file, out_dir, out_name):
    img = cv2.imread(read_file)
    msg = out_dir + 'quantize_origin.png'
    cv2.imwrite(msg, img)

    new_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        np.uint8
    )

    # quantization
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(3):
                if img[i, j][k] < 128:
                    gray = 0
                else:
                    gray = 128
                new_img[i, j][k] = np.uint8(gray)

    msg = out_dir + out_name
    cv2.imwrite(msg, new_img)
    return new_img


def dim_pic(read_file, dim_point, out_dir, out_name):
    img = cv2.imread(read_file)
    msg = out_dir + 'dime_origin.png'
    cv2.imwrite(msg, img)
    
    new_img = cv2.subtract(
        img,
        np.ones(img.shape, dtype='uint8')*dim_point
    )
    
    msg = out_dir + out_name
    cv2.imwrite(msg, new_img)
    return new_img
    

if __name__ == '__main__':
    org_pic = './Lena.png'
    out_dir = './out/'
    sampled = sampling(org_pic, 16, out_dir, 'sampled.png')
    quantized = quantize(org_pic, out_dir, 'quantized.png')
    dim = dim_pic(org_pic, 50, out_dir, 'dim.png')

    # change pixel
    img = cv2.imread(org_pic)
    test = img[88, 142]
    print('read:', test)
    img[88, 142] = [255, 255, 255]
    print('changed:', test)



