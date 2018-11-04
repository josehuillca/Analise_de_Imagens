is_jupyter_help = True
if is_jupyter_help:
    from help_functions.histogramas import *
    from help_functions.limiarizacao import *
    from help_functions.canny import *
else:
    from histogramas import *
    from limiarizacao import *
    from canny import *
import sys
from os import listdir
from os.path import isfile, join, isdir, exists
import numpy as np
import matplotlib.image as mplimg
import matplotlib.pyplot as plt
from texttable import Texttable
import copy
import time
# to work openCV on python3
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2


def is_image(f):
    l = f.split(".")
    if l[len(l)-1] == 'jpg' or l[len(l)-1] == 'png':
        return True
    return False

def list_files_from_directory(my_path):
    my_list = []
    if isdir(my_path):
        my_list = [f for f in listdir(my_path) if isfile(join(my_path, f)) and is_image(f)]
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
        #img = cv2.imread(img_path)    # BGR
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
        plt.tight_layout()
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


def use_erode_dilate(image):
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    #erode = cv2.erode(image, kernel_erode)
    dilate = cv2.dilate(image, kernel_dilate)
    erode = cv2.erode(dilate, kernel_erode)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    dilate = cv2.dilate(erode, kernel_dilate)

    return dilate


# --- AQUI SEGMENTAMOS EL CIELO --------
def segment_sky(image_base_path, img, show_=True, img_save_neg="negative.jpg", img_save_otsu="/segment_sky.jpg"):
    my_img = image_read(image_base_path, img)
    neg_cor = negativo_grises(my_img)
    plt.imsave(image_base_path + "/" + img_save_neg, neg_cor)
    result = otsu(image_base_path, img_save_neg)
    erode_dilate = use_erode_dilate(result)
    cv2.imwrite(image_base_path + img_save_otsu, erode_dilate)
    if show_:
        display_multiple_images([my_img, neg_cor], ["Imagen original", "Imagen negatica"], 1, 2, False, True)
        display_multiple_images([result, erode_dilate], ["Otsu", "Segment sky(Dilate + Erode)"], 1, 2, False, True)


def get_contourns_mountain(path_base, image_base, img_sky="/segment_sky.jpg", img_temp="/img_absdiff_contourns_mountain.jpg"):
    img = cv2.imread(path_base + img_sky)
    neg_img = negativo_grises(img)
    my_img = cv2.imread(path_base + "/" + image_base)  # 2

    # ------- Calculamos la diferencia absoluta de las dos imagenes
    diff_total = cv2.absdiff(my_img, neg_img)
    cv2.imwrite(path_base + img_temp, diff_total)

    result = get_canny(path_base, img_temp, 65)  # 34

    result = use_erode_dilate(result)  # dilate, luego erode, se modifico eso
    cv2.imwrite(path_base + "/result_contours_mountain.jpg", result)
    display_multiple_images([diff_total, result], ["Diferencia absoluta","Contornos da mountain"], 1, 2, False, True)


# --------------- aqui sumamos el cielo con otra imagen
def merge_sky(path, image, path_base, img_otsu="/segment_sea.jpg", img_temp="/img_temp_diff.jpg", show_=True):
    img = cv2.imread(path_base + img_otsu)
    neg_img = negativo_grises(img)
    my_img = cv2.imread(path + "/" + image)#2

    #------- Calculamos la diferencia absoluta de las dos imagenes
    diff_total = cv2.absdiff(my_img, neg_img)
    cv2.imwrite(path_base + img_temp, diff_total)
    if show_:
        display_multiple_images([neg_img, diff_total], ["Negative image(segment sea)", "absdiff"], 1, 2, False, True)


# union de todo
def get_ship_2(path, image, path_base ):
    # segment_sky(image_base_path, "DSC_000002905.jpg")
    merge_sky(path, image, path_base)

    # main_canny(image_path , "/" + my_images[3], 46)
    result = get_canny(path_base, "/img_temp_diff.jpg", 65)  # 34

    result = use_erode_dilate(result)  # dilate, luego erode, se modifico eso
    cv2.imwrite(path_base + "/result_ships.jpg", result)


def remove_noise(path_base, image="/mediacolor.jpg", limiar=60, show_=True):
    result = get_canny(path_base, image, limiar)
    countour = cv2.imread(path_base + "/result_contours_mountain.jpg")
    rest_countour_mountain = copy.copy(result)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    # erode = cv2.erode(image, kernel_erode)
    countour = cv2.dilate(countour, kernel_dilate)

    black = 0
    white = 255
    for x in range(0, result.shape[1]):
        for y in range(0, result.shape[0]):
            if countour[y, x][0] == white:
                rest_countour_mountain[y, x] = black
    if show_:
        display_multiple_images([result, rest_countour_mountain], ["canny", "rest contourn mountain"], 1, 2, False, True)
    return  rest_countour_mountain


def segment_sea(path_base, image='/result_contours_mountain.jpg', show_=True):
    img = cv2.imread(path_base + image)
    limit_y = 300  # obtenido manualmente
    white = 255
    img_result = create_image(img.shape, 3, img.dtype)

    for x in range(0, img.shape[1], 1):
        for y in range(img.shape[0]-1, 0, -1):
            val = img[y, x][0] #como es un imagen en black-white no importa cual tomar
            if val==white and y<=limit_y:
                break
            img_result[y, x] = [white, white, white]

    neg = negativo_grises(img_result)
    cv2.imwrite(path_base + "/segment_sea.jpg", neg)
    if show_:
        display_multiple_images([img, neg], ["contourns da mountain", "Segment sea"], 1, 2, False, True)


def segment_mountain(path_base):
    img_sky = cv2.imread(path_base + "/segment_sky.jpg")
    img_sea = cv2.imread(path_base + "/segment_sea.jpg")
    # la interseccion de las dos imagenes son las montanias
    img_bwa = cv2.bitwise_and(img_sky, img_sea)
    # img_bwo = cv2.bitwise_or(img_sky, img_sea)
    img_bwx = cv2.bitwise_xor(img_sky, img_sea)

    display_multiple_images([img_bwa, img_bwx],
                            ["AND of 'Segment sky' and 'Segment sea'","XOR of 'Segment sky' and 'Segment sea'"],
                            1, 2, False, True)


def get_media_color(path_base, img_base):
    img = cv2.imread(path_base + "/segment_sea.jpg")
    my_img = cv2.imread(path_base + "/" + img_base)  # 2

    # ------- Calculamos la diferencia absoluta de las dos imagenes
    diff_total = cv2.absdiff(my_img, img)
    ini_y = 300
    m = 9
    arr_b = []
    arr_g = []
    arr_r = []

    for x in range(0, m):
        for y in range(ini_y, ini_y + m):
            arr_b.append(diff_total[y, x][0])
            arr_g.append(diff_total[y, x][1])
            arr_r.append(diff_total[y, x][2])
    b = np.sum(arr_b)/len(arr_b)
    g = np.sum(arr_g)/len(arr_g)
    r = np.sum(arr_r)/len(arr_r)

    return [b, g, r]


def change_sky_mountain(path_base, img_diff_total, bgr, show_=True):
    img = cv2.imread(path_base + "/segment_sea.jpg")
    diff_total = cv2.imread(path_base + img_diff_total)
    diff_total_temp = copy.copy(diff_total)

    # cambiar el cielo y montañas de color
    for x in range(0, img.shape[1]):
        for y in range(0, img.shape[0]):
            if img[y, x][0] == 255:
                diff_total_temp[y, x] = bgr

    cv2.imwrite(path_base + "/mediacolor.jpg", diff_total_temp)
    if show_:
        display_multiple_images([diff_total, diff_total_temp], ["absdiff", "adbsdiff result"], 1, 2, False, True)
    '''cv2.imshow("diff", diff_total)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''


def get_ship(image_path, name_my_image, image_base_path, limiar_=60):
    merge_sky(image_path, name_my_image, image_base_path, "/segment_sea.jpg", show_=False)
    change_sky_mountain(image_base_path, "/img_temp_diff.jpg", [0, 0, 0], show_=False)
    result_ships = remove_noise(image_base_path, limiar=limiar_, show_=False)

    original = cv2.imread(image_path + "/" + name_my_image)
    get_ship_contourn(result_ships, original, rgb_=False)


def get_ship_contourn(img, original, rgb_=True):
    result = use_erode_dilate(img)
    original_copy = original.copy()

    cnts = cv2.findContours(result.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(original, cnts, -1, (0, 255, 0), 3)
    if not rgb_:
        original_copy = rgb_to_bgr(original_copy)
        original = rgb_to_bgr(original)
    display_multiple_images([original_copy, original], ["original", "Get ship contour"], 1, 2, False, True)