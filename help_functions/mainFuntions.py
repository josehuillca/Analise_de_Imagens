import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2, numpy as np, matplotlib.pyplot as plt
from canny import my_auto_canny, my_get_canny
from histogramas import show_histogram_rgb
from HelpFunctions import list_files_from_directory, my_print
import copy
from random import randint
from scipy.spatial import distance


SEGMENT_SEA_BIN = '../data/base_real/segment_sea.jpg'  # image that stores the segmented sea(binary image)
ONLY_SEA_IMAGE = '../data/base_real/only_sea.jpg'  # image
PATH_CLASS_SHIP = '../data/classes/'
TEMPORAL_IMAGE_OTSU = '../data/base_real/temp_otsu_calculate.jpg'  # image temporal to calculate the optimal threshold
PATH_SAVE_SEGMENTED_SHIPS = '../data/segmented_ships_real/'   # path to save segmented ships
CV2_WINDOW_RESIZE_WIDTH = 1200  # resize image to show on screen with openCV
CV2_WINDOW_RESIZE_HEIGHT = 800  # resize image to show on screen with openCV
IMAGE_SIZE_WIDTH = 6000         # image of the beach in pixels
IMAGE_SIZE_HEIGHT = 4000        # image of the beach in pixels
MIN_LEN_W_BOAT = 80             # Minimum perimeter that a boat can take
MAX_LEN_W_BOAT = 1200          # Minimum perimeter that a boat can take
MIN_LEN_H_BOAT = 80             # Maximum perimeter that a boat can take
MAX_LEN_H_BOAT = 1200          # Maximum perimeter that a boat can take
AXI_Y_CROP = 2190
# minimo 65 h, ver realacion entre h y w
REFERENCE_POINT = [3000, 2100]  # coordinates x, y
CENTROID_REAL_CLASS_1 = [255, 0, 0]  # [r, g, b]
CENTROID_REAL_CLASS_2 = [0, 255, 0]  # [r, g, b]
CENTROID_REAL_CLASS_3 = [0, 255, 200]  # [r, g, b]
CENTROID_REAL_CLASS_4 = [255, 0, 255]  # [r, g, b]
CENTROID_REAL_CLASS_5 = [255, 255, 0]  # [r, g, b]
CENTROID_REAL_CLASS_6 = [0, 0, 255]  # [r, g, b]
LIST_CLASSES_COR = [CENTROID_REAL_CLASS_1, CENTROID_REAL_CLASS_2, CENTROID_REAL_CLASS_3,
                    CENTROID_REAL_CLASS_4, CENTROID_REAL_CLASS_5, CENTROID_REAL_CLASS_6]


def get_random_color():
    """ Get a random color RGB or BGR, they are just random numbers
    :return: a array with the color [r,g,b] or [b,g,r]
    """
    return (randint(0, 255), randint(0, 255), randint(0, 255))


def get_min_xy_and_wh(box_points):
    """ We use this function to crop image with point's start min_xy
    :param box_points: list with four coordinates that represent a box
    :return: two coordinates that represent the minimum coordinates and length of width and height respectively
    """
    list_x = [box_points[0][0], box_points[1][0], box_points[2][0], box_points[3][0]]
    list_y = [box_points[0][1], box_points[1][1], box_points[2][1], box_points[3][1]]
    min_xy = [min(list_x), min(list_y)]
    max_xy = [max(list_x), max(list_y)]
    len_wh = [max_xy[0]-min_xy[0], max_xy[1]-min_xy[1]]  # getting length of width and height
    return min_xy, len_wh


def midpoint(xy1, xy2):
    """
    :param xy1: first coordinate x, y
    :param xy2: second coordinate x, y
    :return: mid point
    """
    return [(xy1[0] + xy2[0])/2.0, (xy1[1] + xy2[1])/2.0]


def get_min_length_wh(box_points):
    """
    :param box_points: list with four coordinates that represent a box
    :return: distance between mid points
    """
    p_mid_sup = midpoint(box_points[0], box_points[1])
    p_mid_inf = midpoint(box_points[2], box_points[3])
    p_mid_left = midpoint(box_points[1], box_points[2])
    p_mid_right = midpoint(box_points[0], box_points[3])
    min_len_w = distance.euclidean(p_mid_sup, p_mid_inf)
    min_len_h = distance.euclidean(p_mid_left, p_mid_right)
    return min_len_w, min_len_h


def show_image_on_screen(window_name, image):
    """ Show image of openCV on screen
    :param window_name: window name
    :param image: image read of openCV
    :return: none
    """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, CV2_WINDOW_RESIZE_WIDTH, CV2_WINDOW_RESIZE_HEIGHT)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


def absdiff_img_sea(image, show_=True):
    """ cv2.absdiff : We calculate the per-element absolute difference between two arrays
    :param image: image read of openCV
    :param show_: bool = show result (image) on screen
    :return: Absolute difference between two arrays when they have the same size and type
    """
    img_sea = cv2.imread(SEGMENT_SEA_BIN)
    result = cv2.absdiff(image, img_sea)
    if show_:
        window_name = "absdiff-img-sea"
        show_image_on_screen(window_name, result)
    return result


def get_only_sea(image, show_=True):
    """ We use the segmented image of the sea in binary, to only stay with the part of the sea
    :param image: image read of openCV
    :param show_: bool = show result (image) on screen
    :return: image with sky and mountains in white color
    """
    img_sea_bin = cv2.imread(SEGMENT_SEA_BIN)
    result = cv2.bitwise_or(img_sea_bin, image)  # Calculates the per-element bit-wise disjunction of two arrays

    if show_:
        window_name = "segment-sea"
        show_image_on_screen(window_name, result)
    return result


def get_edges_noise_images(image, show_=True):
    """ We remove noise and get the edges of the image
    :param image: image read of openCV
    :param show_: bool = show result (image) on screen
    :return: Resulting image applied canny and resulting image by applying filter to remove noise
    """
    image_copy = copy.copy(image)
    no_noise_image = image_copy

    result_canny = my_auto_canny(no_noise_image)
    if show_:
        window_name = "noisy-sea"
        show_image_on_screen(window_name, result_canny)
    return result_canny, no_noise_image


def remove_bad_contours(contours):
    """ remove small and large contours
    :param contours: array with all contours
    :return: array with contours between min and max perimeter
    """
    new_contours = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        len_w, len_h = get_min_length_wh(box)
        bool_len_w = MIN_LEN_W_BOAT < len_w < MAX_LEN_W_BOAT
        bool_len_h = MIN_LEN_H_BOAT < len_h < MAX_LEN_H_BOAT
        if bool_len_w and bool_len_h:
            new_contours.append(cnt)
    return new_contours


def find_contours(image_edges, image_color, show_=True, kernel_d=(37, 37)):
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_d)
    img_edges_contour = cv2.dilate(image_edges, kernel_dilate)
    contours = cv2.findContours(img_edges_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    image_color_all_cnts = copy.copy(image_color)
    for cnt in contours:
        cv2.drawContours(image_color_all_cnts, [cnt], -1, get_random_color(), 12)
    new_contours = remove_bad_contours(contours)

    image_color_new_cnts = copy.copy(image_color)

    # for each contour
    for cnt in new_contours:

        # draw it in red color
        cv2.drawContours(image_color_new_cnts, [cnt], -1, get_random_color(), 5)

    if show_:
        window_name = "contour-result"
        show_image_on_screen(window_name, image_color_new_cnts)
    return new_contours


def crop_image(image, contours):
    image_color = copy.copy(image)
    i = 0
    list_imgs = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        min_xy, len_wh = get_min_xy_and_wh(box)
        y = min_xy[1] if (min_xy[1] - 10)<0 else min_xy[1] - 10
        x = min_xy[0] if (min_xy[0] - 10)<0 else min_xy[0] - 10
        w = len_wh[0] if (len_wh[0] + 20)<image.shape[1] else len_wh[0] + 20
        h = len_wh[1] if (len_wh[1] + 20)<image.shape[0] else len_wh[1] + 20
        crop_img = image_color[y:y + h, x:x + w]
        list_imgs.append(crop_img)
        i = i + 1
    return list_imgs


def save_image_contours(image, contours, file_name):
    name = file_name.split('.')[0] + '_crop_'
    list_imgs = crop_image(image, contours)
    i = 0
    for crop_img in list_imgs:
        cv2.imwrite(PATH_SAVE_SEGMENTED_SHIPS + name + str(i) + '.jpg', crop_img)
        i = i + 1


def extract_feature_crop_img(img, kernel_d=(27, 27)):
    image_edges = my_auto_canny(img)
    new_contours = find_contours(image_edges, img, show_=False, kernel_d=kernel_d)
    img_bin = np.zeros((img.shape[0], img.shape[1]))
    cv2.fillPoly(img_bin, pts=new_contours, color=(255, 255, 255))
    if len(new_contours)<=0:
        return [[0,0], 0, 0, 0], new_contours

    cnt = new_contours[0]
    M = cv2.moments(cnt)
    cx = int(M['m10'] / M['m00']) / img.shape[1]
    cy = int(M['m01'] / M['m00']) / img.shape[0]

    area = cv2.contourArea(cnt) / (img.shape[1]*img.shape[0])
    perimeter = cv2.arcLength(cnt, True) / (img.shape[1]+img.shape[0])
    aspect_ratio = float(img.shape[1]) / img.shape[0]
    return [cx, cy, area, perimeter, aspect_ratio], new_contours


def extract_feature_of_classes(show_=True):
    my_images = list_files_from_directory(PATH_CLASS_SHIP)
    list_features = []
    for i in range(0, len(my_images)):
        my_img_name = PATH_CLASS_SHIP + "class" + str(i+1) + ".jpg"
        img = cv2.imread(my_img_name)

        f, new_contours = extract_feature_crop_img(img)
        if show_:
            for cnt in new_contours:
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                cv2.drawContours(img, [box], 0, LIST_CLASSES_COR[i], 5)
            show_image_on_screen("bin", img)
        list_features.append(f)
    return list_features


def extract_features(image, show_=True):
    result_sea = get_only_sea(image, False)
    result_canny, no_noise_image = get_edges_noise_images(result_sea, False)
    contours = find_contours(result_canny, no_noise_image, True)
    # binarizar, rellenar contorno
    img_bin = np.zeros((image.shape[0], image.shape[1]), dtype=image.dtype)
    cv2.fillPoly(img_bin, pts=contours, color=(255, 255, 255))

    list_images_crop = crop_image(img_bin, contours)
    if show_:
        l_crop = crop_image(image, contours)
        for i in range(len(l_crop)):
            show_image_on_screen("crop", l_crop[i])

    list_f = []
    for j in range(len(contours)):
        f, _ = extract_feature_crop_img(list_images_crop[j], kernel_d=(27, 27))
        list_f.append(f)

    my_print(['cx', 'cy', 'area', 'perimeter', 'aspect_ratio'], np.array(list_f), title="first-features")
    return list_f


def compare_features(features_classes, features):
    list_result = []
    for i in range(len(features)):
        r = []
        for j in range(len(features_classes)):
            dist_c = distance.euclidean(features_classes[j], features[i])
            r.append(dist_c)
        list_result.append(r)

    for i in range(len(list_result)):
        my_print(['tipo1', 'tipo2', 'tipo3', 'tipo4', 'tipo5', 'tipo6'], np.array(list_result), title="second-features:" + str(i))
