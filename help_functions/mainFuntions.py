import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2, numpy as np, matplotlib.pyplot as plt
from canny import my_auto_canny, my_get_canny
from limiarizacao import get_limiar_otsu
import copy
from random import randint
from scipy.spatial import distance


SEGMENT_SEA_BIN = '../data/base_real/segment_sea.jpg'  # image that stores the segmented sea(binary image)
ONLY_SEA_IMAGE = '../data/base_real/only_sea.jpg'  # image
TEMPORAL_IMAGE_OTSU = '../data/base_real/temp_otsu_calculate.jpg'  # image temporal to calculate the optimal threshold
PATH_SAVE_SEGMENTED_SHIPS = '../data/segmented_ships_real/'   # path to save segmented ships
CV2_WINDOW_RESIZE_WIDTH = 1200  # resize image to show on screen with openCV
CV2_WINDOW_RESIZE_HEIGHT = 800  # resize image to show on screen with openCV
IMAGE_SIZE_WIDTH = 6000         # image of the beach in pixels
IMAGE_SIZE_HEIGHT = 4000        # image of the beach in pixels
MIN_LEN_W_BOAT = 80             # Minimum perimeter that a boat can take
MAX_LEN_W_BOAT = 300*4          # Minimum perimeter that a boat can take
MIN_LEN_H_BOAT = 80             # Maximum perimeter that a boat can take
MAX_LEN_H_BOAT = 700*2          # Maximum perimeter that a boat can take
AXI_Y_CROP = 2190
# minimo 65 h, ver realacion entre h y w


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
    #img_sea_bin = cv2.imread(SEGMENT_SEA_BIN)
    #result = cv2.bitwise_or(img_sea_bin, image)  # Calculates the per-element bit-wise disjunction of two arrays

    y = AXI_Y_CROP
    x = 0
    w = IMAGE_SIZE_WIDTH
    h = IMAGE_SIZE_HEIGHT - y
    result = image[y:y+h, x:x+w]  # image crop

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
    # cv2.fastNlMeansDenoising :Perform image denoising using 'Non-local Means Denoising' algorithm  with
    # several computational optimizations. Noise expected to be a gaussian white noise
    # cv2.fastNlMeansDenoisingColored(src[, dst[, h[, hColor[, templateWindowSize[, searchWindowSize]]]]])
    no_noise_image = image_copy #cv2.fastNlMeansDenoisingColored(image_copy, None, 7, 15, 7, 21)

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


def find_contours(image_edges, image_color, show_=True):
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
    # contour = cv2.morphologyEx(image_edges, cv2.MORPH_CLOSE, kernel)
    img_edges_contour = cv2.dilate(image_edges, kernel_dilate)
    # kernel = np.ones((5, 5), np.uint8)
    # contour = cv2.erode(contour, kernel, iterations=1)
    contours = cv2.findContours(img_edges_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    image_color_all_cnts = copy.copy(image_color)
    for cnt in contours:
        cv2.drawContours(image_color_all_cnts, [cnt], -1, get_random_color(), 3)
    new_contours = remove_bad_contours(contours)

    image_color_new_cnts = copy.copy(image_color)

    # for each contour
    for cnt in new_contours:
        # get convex hull
        hull = cv2.convexHull(cnt)

        # draw it in red color
        cv2.drawContours(image_color_new_cnts, [hull], -1, get_random_color(), 5)

    if show_:
        window_name = "contour-result"
        show_image_on_screen(window_name, img_edges_contour)
    return new_contours


def save_image_contours(image, contours, file_name):
    name = file_name.split('.')[0] + '_crop_'
    image_color = copy.copy(image)
    i = 0
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # cv2.drawContours(image_color, [box], 0, get_random_color(), 2)
        min_xy, len_wh = get_min_xy_and_wh(box)
        y = min_xy[1]
        x = min_xy[0]
        w = len_wh[0]
        h = len_wh[1]
        crop_img = image_color[y:y + h, x:x + w]
        window_name = "contour-result"
        # show_image_on_screen(window_name, crop_img)
        cv2.imwrite(PATH_SAVE_SEGMENTED_SHIPS + name + str(i) + '.jpg', crop_img)
        i = i + 1