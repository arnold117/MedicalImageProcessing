import cv2
import matplotlib.pyplot as plt

if __name__ == '__main__':
    src = cv2.imread('./Lena.png')

    # single channel hist
    plt.hist(src.ravel(), bins=256, density=1, facecolor='black', alpha=0.75)
    plt.savefig('./out/singleHist.png')
    plt.close()

    # 3 channel hist
    b, g, r = cv2.split(src)
    plt.hist(b.ravel(), bins=256, density=1, facecolor='b', edgecolor='b', alpha=0.75)
    plt.hist(g.ravel(), bins=256, density=1, facecolor='g', edgecolor='g', alpha=0.75)
    plt.hist(r.ravel(), bins=256, density=1, facecolor='r', edgecolor='r', alpha=0.75)
    plt.savefig('./out/c3Hist.png')
    plt.close()

    # gray hist equalization
    gray_img = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    equ = cv2.equalizeHist(gray_img)
    cv2.imwrite('./out/gray_img.png', equ)

    # RGB hist equalization
    b, g, r = cv2.split(src)
    bHist = cv2.equalizeHist(b)
    gHist = cv2.equalizeHist(g)
    rHist = cv2.equalizeHist(r)

    rgbEqu = cv2.merge((bHist, gHist, rHist))
    cv2.imwrite('./out/rgb_img.png', rgbEqu)

    plt.hist(bHist.ravel(), bins=256, facecolor='b', edgecolor='b')
    plt.hist(gHist.ravel(), bins=256, facecolor='g', edgecolor='g')
    plt.hist(rHist.ravel(), bins=256, facecolor='r', edgecolor='r')
    plt.savefig('./out/rgbHist.png')

