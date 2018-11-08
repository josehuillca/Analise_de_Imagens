is_jupyter_hist = True
if is_jupyter_hist:
    from help_functions.HelpFunctions import *
else:
    from HelpFunctions import *
import sys
import matplotlib.pyplot as plt
import time
import numpy as np
# to work openCV on python3
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2


# ----------------- FUNCIONES REPETIDAS
def create_image(shape, nchanels, dtype):  # QUE RARO ***
    w = shape[1]
    h = shape[0]
    img = np.zeros([h, w, nchanels], dtype=dtype)
    if nchanels==3:
        r, g, b = cv2.split(img)
        img = cv2.merge([r, g, b])
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
        plt.tight_layout()
        if not show_axis:
            plt.axis('off')

        cont += 1
        if cont >= total_imgs:
            break
    plt.show()
# ------------------


def single_band_representation_(img, show_=True):
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
    if show_:
        display_multiple_images([img_r, img_g, img_b], color_titles, 1, 3, show_axis=False)
    return img_r, img_g, img_b


def show_histogram_rgb(img, title="Title", color=('r', 'g', 'b'), min_cor=0, max_cor=256):
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [max_cor], [min_cor, max_cor])
        plt.plot(histr, color=col)
        plt.xlim([min_cor, max_cor])
    plt.title(title)
    plt.tight_layout()


def histograms_equalization(img_r, img_g, img_b, show=True, compare_hist=True):
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
    if show:
        display_multiple_images([res_r, res_g, res_b],equ_titles, 3, 1, show_axis=False, cmap_gray=True)

    if compare_hist:
        compare_histograms(img_r, equ_r, "Histograma Vermelha(Red)", "Histograma Equalizado", [0], 'r', 'k')
        compare_histograms(img_g, equ_g, "Histograma Verde(Green)", "Histograma Equalizado", [0], 'g', 'k')
        compare_histograms(img_b, equ_b, "Histograma Azul(Blue)", "Histograma Equalizado", [0], 'b', 'k')
    return  equ_r, equ_g, equ_b

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


def show_rgb_equalized(image, show_=True):
    channels = cv2.split(image)
    eq_channels = []
    for ch, color in zip(channels, ['R', 'G', 'B']):
        eq_channels.append(cv2.equalizeHist(ch))

    eq_image = cv2.merge(eq_channels)
    if show_:
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title("Imagem Original")

        # eq_image = cv2.cvtColor(eq_image, cv2.COLOR_BGR2RGB)  # Ya esta en RGB->por el image_read()
        plt.subplot(1, 2, 2)
        plt.imshow(eq_image)
        plt.axis('off')
        plt.title("Imagem Equalizada")
        plt.show()

        # ---------------------- Show Histograms
        plt.subplot(2, 1, 1)
        show_histogram_rgb(image, "Imagem Original")

        plt.subplot(2, 1, 2)
        show_histogram_rgb(eq_image, "Imagem Equalizada")

        plt.show()
    return eq_image


def show_grayscale_equalized(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_grayscale_image = cv2.equalizeHist(grayscale_image)
    display_multiple_images([grayscale_image, eq_grayscale_image], ["Imagem cinza", "Imagem cinza equalizada"], 1, 2, False, True)
    compare_histograms(grayscale_image, eq_grayscale_image, "Hist. cinza", "Hist. cinza equalizada", [0], 'm', 'k')

# https://lmcaraig.com/image-histograms-histograms-equalization-and-histograms-comparison/
def show_hsv_equalized(image, show_=True):
    r, g, b = cv2.split(image)
    image_rgb = cv2.merge([b, g, r])
    H, S, V = cv2.split(cv2.cvtColor(image_rgb, cv2.COLOR_BGR2HSV))
    eq_V = cv2.equalizeHist(V)
    eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2RGB)
    if show_:
        plt.subplot(2, 1, 1)
        plt.imshow(eq_image, cmap='gray')
        plt.title("HSV Equalizada")
        plt.tight_layout()

        plt.subplot(2, 1, 2)
        show_histogram_rgb(eq_image, "Histograma HSV")

        plt.show()
    return  eq_image



def img_grayscale(img_rgb):
    r, g, b = cv2.split(img_rgb)
    image_bgr = cv2.merge([b, g, r])
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return gray_image


def rgb_to_bgr(im):
    r, g, b = cv2.split(im)
    image_bgr = cv2.merge([b, g, r])
    return  image_bgr

def negativo_color(im_cor):
    im5 = create_image(im_cor.shape, 3, im_cor.dtype)
    i = 0
    while i < im_cor.shape[0]:
        j = 0
        while j < im_cor.shape[1]:
            r, g, b = im_cor[i,j]
            rn = 255 - r
            gn = 255 - g
            bn = 255 - b
            im5[i,j] = [rn, gn, bn]
            j+=1
        i+=1
    return im5


def negativo_grises(im):
    im_gray = img_grayscale(im)
    im5 = create_image(im.shape, 3, im.dtype)
    i = 0
    while i < im_gray.shape[0]:
        j = 0
        while j < im_gray.shape[1]:
            gris = im_gray[i,j]  # como es gris no importa
            valor = 255 - gris
            im5[i, j] = [valor, valor, valor]
            j+=1
        i+=1
    return im5


def negative_images(im):
    tiempoIn = time.time()
    neg_cor = negativo_color(im)
    tiempoFin = time.time()
    print('Negative cor demoro: ', tiempoFin - tiempoIn, 'Segundos')
    neg_gray = negativo_grises(im)
    tiempoFin = time.time()
    print('Negative gris demoro: ', tiempoFin - tiempoIn, 'Segundos')
    display_multiple_images([neg_cor, neg_gray], ["Negative Cor", "Negative gris"], 1, 2, show_axis=False, cmap_gray=True)

    # ---------------------- Show Histograms
    compare_histograms(neg_cor, neg_gray, "Negative cor", "Negative gris", [0], 'c', 'k')

    plt.show()

# https://guilhermekfreitaslab.wordpress.com/2015/09/16/amostragem-e-quantizacao/
def quantizacao(img, n = 4):
    m = np.amax(img)+1
    a = np.uint8(img/(m/float(n)))
    b = np.uint8((a/(n-1.))*255) #transforma de volta pra 0-255 (para exibir a imagem)

    return b

def quantizacao_4_8_32(img):
    img4 = quantizacao(img, 4)
    img8 = quantizacao(img, 8)
    img32 = quantizacao(img, 32)
    titles = ["Quantização 4 tons", "Quantização 8 tons", "Quantização 32 tons"]
    display_multiple_images([img4, img8, img32], titles, 1, 3, show_axis=False, cmap_gray=True)
    # ---------------------- Show Histograms
    plt.subplot(3, 1, 1)
    show_histogram_rgb(img4, titles[0])
    plt.subplot(3, 1, 2)
    show_histogram_rgb(img8, titles[1])
    plt.subplot(3, 1, 3)
    show_histogram_rgb(img32, titles[2])

    plt.show()