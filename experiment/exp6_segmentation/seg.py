import cv2
import numpy as np


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y


def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint.x, currentPoint.y]) - int(img[tmpPoint.x, tmpPoint.y]))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0),
                    Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape
    seedMark = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
    label = 1
    connects = selectConnects(p)
    while len(seedList) > 0:
        currentPoint = seedList.pop(0)

        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
    return seedMark


# 判断方框是否需要再次拆分为四个
def judge(w0, h0, w, h):
    a = img[h0: h0 + h, w0: w0 + w]
    ave = np.mean(a)
    std = np.std(a, ddof=1)
    count = 0
    total = 0
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
            # 注意！我输入的图片数灰度图，所以直接用的img[j,i]，RGB图像的话每个img像素是一个三维向量，不能直接与avg进行比较大小。
            if abs(img[j, i] - ave) < 1 * std:
                count += 1
            total += 1
    if (count / total) < 0.95:  # 合适的点还是比较少，接着拆
        return True
    else:
        return False


# 将图像将根据阈值二值化处理，在此默认125
def draw(w0, h0, w, h):
    for i in range(w0, w0 + w):
        for j in range(h0, h0 + h):
            if img[j, i] > 125:
                img[j, i] = 255
            else:
                img[j, i] = 0


def block_split(w0, h0, w, h):
    if judge(w0, h0, w, h) and (min(w, h) > 5):
        block_split(w0, h0, int(w / 2), int(h / 2))
        block_split(w0 + int(w / 2), h0, int(w / 2), int(h / 2))
        block_split(w0, h0 + int(h / 2), int(w / 2), int(h / 2))
        block_split(w0 + int(w / 2), h0 + int(h / 2), int(w / 2), int(h / 2))
    else:
        draw(w0, h0, w, h)


def watershed(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二值化
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 去除噪音(要不然最终成像会导致过度分割)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # 确定非对象区域
    sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 进行膨胀操作

    # 确定对象区域
    dist_transform = cv2.distanceTransform(opening, 1, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # 寻找未知的区域
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)  # 非对象区域减去对象区域就是不确定区域

    # 为对象区域类别标记
    ret, markers = cv2.connectedComponents(sure_fg)
    # 为所有的标记加1，保证非对象是0而不是1
    markers = markers + 1
    # 现在让所有的未知区域为0
    markers[unknown == 255] = 0

    # 执行分水岭算法
    markers = cv2.watershed(img, markers)
    img[markers == -1] = [255, 0, 0]

    # 解决中文显示问题
    return img


if __name__ == '__main__':
    # original input
    org = cv2.imread('./Lena.png')
    cv2.imwrite('./out/origin.png', org)

    # need gray image not rgb
    gray_img = cv2.cvtColor(org, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('./out/gray_img.png', gray_img)

    # gauss smoothed
    gaussian_img = cv2.GaussianBlur(gray_img, (3, 3), 0)
    cv2.imwrite('./out/gaussian_img.png', gaussian_img)

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

    seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
    region_grow = regionGrow(gaussian_img, seeds, 10)
    cv2.imwrite('./out/region_grow.png', region_grow)

    img = gaussian_img
    height, width = img.shape
    block_split(0, 0, width, height)
    cv2.imwrite('./out/block_split.png', img)

    water = watershed(org)
    cv2.imwrite('./out/wartershed.png', water)
