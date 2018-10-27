import sys
from os import listdir
from os.path import isfile, join, isdir, exists
import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
from texttable import Texttable
# to work openCV on python3
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2


def list_files_from_directory(my_path):
    my_list = []
    if isdir(my_path):
        my_list = [f for f in listdir(my_path) if isfile(join(my_path, f))]
    else:
        print("ERROR!: The directory({0}) does not exist ...".format(my_path))
    return my_list


# printing a pretty matrix (only in type int, float)
def print_matrix(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="   ")
        print("")


def show_image_properties(img):
    shape = img.shape

    table = Texttable()
    table.set_deco(Texttable.HEADER)
    table.set_cols_dtype(['t',   # text
                          'a'])  # automatic
    table.set_cols_align(["l", "r"])
    table.add_rows([["properties", "values"],
                    ["width", shape[1]],
                    ["height", shape[0]],
                    ["channels", shape[2]],
                    ["# of pixels", img.size],
                    ["data type", img.dtype]])
    print(table.draw())


def image_read(path, image):
    img_path = path + "/" + image
    img = [[]]
    if exists(img_path):
        img = mplimg.imread(img_path)   # RGB
        # img = cv2.imread(img_path)    # BGR
    else:
        print("ERROR!: The image({0}) does not exist ...".format(img_path))
    return img


def display_multiple_images(list_imgs, list_titles, rows, columns, show_axis=True, cmap_gray=False):
    total_imgs = len(list_imgs)
    cont = 0
    fig = plt.figure(figsize=(8, 8))
    for i in range(1, columns * rows + 1):
        img = list_imgs[cont]
        fig.add_subplot(rows, columns, i)
        if cmap_gray:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(list_titles[cont])
        if not show_axis:
            plt.axis('off')

        cont += 1
        if cont >= total_imgs:
            break
    plt.show()


def create_image(shape, nchanels, dtype):
    w = shape[1]
    h = shape[0]
    img = np.zeros([h, w, nchanels], dtype=dtype)
    if nchanels==3:
        r, g, b = cv2.split(img)
        img = cv2.merge([r, g, b])
    return img


def single_band_representation(img, show=True):
    img_b = create_image(img.shape, 3, img.dtype)
    img_g = create_image(img.shape, 3, img.dtype)
    img_r = create_image(img.shape, 3, img.dtype)
    w = img.shape[1]
    h = img.shape[0]
    color_titles = ('Banda Vermelha(Red)', 'Banda verde(Green)', 'Banda Azul(Blue)')
    for x in range(0, w):
        for y in range(0, h):
            cor = img[y, x]
            img_b[y, x] = [cor[2], cor[2], cor[2]]
            img_g[y, x] = [cor[1], cor[1], cor[1]]
            img_r[y, x] = [cor[0], cor[0], cor[0]]
    if show:
        display_multiple_images([img_r, img_g, img_b], color_titles, 1, 3, show_axis=False)
    return img_r, img_g, img_b


def show_histogram_rgb(img, title="Title"):
    color = ('r', 'g', 'b')
    min_cor = 0
    max_cor = 256
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [max_cor], [min_cor, max_cor])
        plt.plot(histr, color=col)
        plt.xlim([min_cor, max_cor])
    plt.title(title)
    plt.show()


def histograms_equalization(img_r, img_g, img_b, compare_hist=True):
    img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY)
    img_g = cv2.cvtColor(img_g, cv2.COLOR_BGR2GRAY)
    img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    equ_r = cv2.equalizeHist(img_r)
    equ_g = cv2.equalizeHist(img_g)
    equ_b = cv2.equalizeHist(img_b)
    equ_titles = ('Banda Vermelha(Red): Imagem original(left), Imagen equalizada(right)',
                  'Banda verde(Green): Imagem original(left), Imagen equalizada(right)',
                  'Banda Azul(Blue): Imagem original(left), Imagen equalizada(right)')

    res_r = np.hstack((img_r, equ_r))  # stacking images side-by-side
    res_g = np.hstack((img_g, equ_g))  # stacking images side-by-side
    res_b = np.hstack((img_b, equ_b))  # stacking images side-by-side
    display_multiple_images([res_r, res_g, res_b],equ_titles, 3, 1, show_axis=False, cmap_gray=True)

    if compare_hist:
        compare_histograms(img_r, equ_r, "Histograma Vermelha(Red)", "Histograma Equalizado", [0], 'r', 'k')
        compare_histograms(img_g, equ_g, "Histograma Verde(Green)", "Histograma Equalizado", [0], 'g', 'k')
        compare_histograms(img_b, equ_b, "Histograma Azul(Blue)", "Histograma Equalizado", [0], 'b', 'k')


def compare_histograms(img1, img2, title1, title2, channel, cor1, cor2):
    min_cor = 0
    max_cor = 256

    hist1 = cv2.calcHist([img1], channel, None, [max_cor], [min_cor, max_cor])
    plt.subplot(1, 2, 1)
    plt.plot(hist1, color=cor1)
    plt.xlim([min_cor, max_cor])
    plt.title(title1)

    hist2 = cv2.calcHist([img2], channel, None, [max_cor], [min_cor, max_cor])
    plt.subplot(1, 2, 2)
    plt.plot(hist2, color=cor2)
    plt.xlim([min_cor, max_cor])
    plt.title(title2)

    plt.show()
