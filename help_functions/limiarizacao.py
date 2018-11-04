import sys
from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank, threshold_local
from skimage.util import img_as_ubyte
from skimage import data
from skimage import io
from skimage.color import rgb2gray
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import math
from PIL import Image
import numpy as np
# to work openCV on python3
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2

def limiarizacao(path, img_cor):
    #src = cv2.imread(path + img_cor, 0)  # pass 0 to convert into gray level
    img = io.imread(path + "/" + img_cor)#img_as_ubyte(data.page())
    img = rgb2gray(img)

    radius = 15
    selem = disk(radius)

    local_otsu = rank.otsu(img, selem)
    threshold_global_otsu = threshold_otsu(img)
    global_otsu = img >= threshold_global_otsu

    fig, axes = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    plt.tight_layout()

    fig.colorbar(ax[0].imshow(img, cmap=plt.cm.gray),
                 ax=ax[0], orientation='horizontal')
    ax[0].set_title('Original')
    ax[0].axis('off')

    fig.colorbar(ax[1].imshow(local_otsu, cmap=plt.cm.gray),
                 ax=ax[1], orientation='horizontal')
    ax[1].set_title('Local Otsu (radius=%d)' % radius)
    ax[1].axis('off')

    ax[2].imshow(img >= local_otsu, cmap=plt.cm.gray)
    ax[2].set_title('Original >= Local Otsu' % threshold_global_otsu)
    ax[2].axis('off')

    ax[3].imshow(global_otsu, cmap=plt.cm.gray)
    ax[3].set_title('Global Otsu (threshold = %d)' % threshold_global_otsu)
    ax[3].axis('off')

    plt.show()


def limiarizacao2(path, img_cor):
    image = io.imread(path + "/" + img_cor)  # img_as_ubyte(data.page())
    image = rgb2gray(image)
    # image = data.page()

    global_thresh = threshold_otsu(image)
    binary_global = image > global_thresh

    block_size = 35
    local_thresh = threshold_local(image, block_size, offset=10)
    binary_local = image > local_thresh

    fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
    ax = axes.ravel()
    plt.gray()

    ax[0].imshow(image)
    ax[0].set_title('Original')

    ax[1].imshow(binary_global)
    ax[1].set_title('Global thresholding')

    ax[2].imshow(binary_local)
    ax[2].set_title('Local thresholding')

    for a in ax:
        a.axis('off')

    plt.show()


def limiarizacao3(path, image):
    img = cv2.imread(path + "/" + image)
    thresh = 0
    maxValue = 255
    th, dst = cv2.threshold(img, thresh, maxValue, cv2.THRESH_BINARY)
    cv2.imshow("original", img)
    cv2.imshow("threshold", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------OTSU-----------------
threshold_values = {}


def Hist(img, show_=False):
    row, col = img.shape
    y = np.zeros(256)
    for i in range(0, row):
        for j in range(0, col):
            y[img[i, j]] += 1
    if show_:
        x = np.arange(0, 256)
        plt.bar(x, y, color='b', width=5, align='center', alpha=0.25)
        plt.show()
    return y


def regenerate_img(img, threshold):
    row, col = img.shape
    y = np.zeros((row, col))
    for i in range(0, row):
        for j in range(0, col):
            if img[i, j] >= threshold:
                y[i, j] = 255
            else:
                y[i, j] = 0
    return y


def countPixel(h_):
    cnt = 0
    for i in range(0, len(h_)):
        if h_[i] > 0:
            cnt += h_[i]
    return cnt


def wieght(s, e, h_):
    w = 0
    for i in range(s, e):
        w += h_[i]
    return w


def mean(s, e, h_):
    m = 0
    w = wieght(s, e, h_)
    for i in range(s, e):
        m += h_[i] * i

    return m / float(w)


def variance(s, e, h_):
    v = 0
    m = mean(s, e, h_)
    w = wieght(s, e, h_)
    for i in range(s, e):
        v += ((i - m) ** 2) * h_[i]
    v /= w
    return v


def threshold(h_):
    cnt = countPixel(h_)
    for i in range(1, len(h_)-1):
        vb = variance(0, i, h_)
        wb = wieght(0, i, h_) / float(cnt)
        mb = mean(0, i, h_)

        vf = variance(i, len(h_), h_)
        wf = wieght(i, len(h_), h_) / float(cnt)
        mf = mean(i, len(h_), h_)

        V2w = wb * (vb) + wf * (vf)
        V2b = wb * wf * (mb - mf) ** 2

        fw = open("trace.txt", "a")
        fw.write('T=' + str(i) + "\n")

        fw.write('Wb=' + str(wb) + "\n")
        fw.write('Mb=' + str(mb) + "\n")
        fw.write('Vb=' + str(vb) + "\n")

        fw.write('Wf=' + str(wf) + "\n")
        fw.write('Mf=' + str(mf) + "\n")
        fw.write('Vf=' + str(vf) + "\n")

        fw.write('within class variance=' + str(V2w) + "\n")
        fw.write('between class variance=' + str(V2b) + "\n")
        fw.write("\n")

        if not math.isnan(V2w):
            threshold_values[i] = V2w


def get_optimal_threshold():
    min_V2w = min(threshold_values.values())
    optimal_threshold = [k for k, v in threshold_values.items() if v == min_V2w]
    print('optimal threshold: ', optimal_threshold[0])
    return optimal_threshold[0]


def otsu(path, img, show_=False):
    image = Image.open(path + "/" + img).convert("L")
    img = np.asarray(image)

    h = Hist(img)
    threshold(h)
    op_thres = get_optimal_threshold()

    res = regenerate_img(img, op_thres)
    if show_:
        plt.imshow(res, cmap='gray')
        plt.show()
    return res