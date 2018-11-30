import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2, numpy as np, matplotlib.pyplot as plt
from canny import my_auto_canny, my_get_canny
from limiarizacao import get_limiar_otsu
import copy
from random import randint


SEGMENT_SEA_BIN = '../data/base/segment_sea.jpg'  # image that stores the segmented sea(binary image)
TEMPORAL_IMAGE_OTSU = '../data/base/temp_otsu_calculate.jpg'  # image temporal to calculate the optimal threshold
CV2_WINDOW_RESIZE_WIDTH = 1200  # resize image to show on screen with openCV
CV2_WINDOW_RESIZE_HEIGHT = 800  # resize image to show on screen with openCV
IMAGE_SIZE_WIDTH = 1500   # image of the beach in pixels
IMAGE_SIZE_HEIGHT = 1000  # image of the beach in pixels
MIN_PERIMETER_BOAT = 89   # Perimetro minimo que puede tomar un barco
MAX_PERIMETER_BOAT = 700  # Perimetro maximo que puede tomar un barco


def get_random_color():
    """ Get a random color RGB or BGR, they are just random numbers
    :return: a array with the color [r,g,b] or [b,g,r]
    """
    return (randint(0, 255), randint(0, 255), randint(0, 255))


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
    result = cv2.bitwise_or(img_sea_bin, image) # Calculates the per-element bit-wise disjunction of two arrays
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
    no_noise_image = cv2.fastNlMeansDenoisingColored(image_copy, None, 7, 15, 7, 21)

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
        perimeter = cv2.arcLength(cnt, True)
        if MIN_PERIMETER_BOAT < perimeter < MAX_PERIMETER_BOAT:
            new_contours.append(cnt)
    return new_contours


def find_contours(image_edges, image_color, show_=True):
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    contour = cv2.dilate(image_edges, kernel_dilate)
    contours = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    new_contours = remove_bad_contours(contours)

    #cv2.drawContours(image_color, contours, -1, (0, 255, 0), 3)
    # for each contour
    for cnt in new_contours:
        # get convex hull
        hull = cv2.convexHull(cnt)
        # draw it in red color
        cv2.drawContours(image_color, [hull], -1, get_random_color(), 2)

        '''rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(image_color, [box], 0, get_random_color(), 2)'''
    if show_:
        window_name = "contour-result"
        show_image_on_screen(window_name, image_color)
    print("# contours: ", len(new_contours))


def find_and_colored_contours(image_edges, image_color, show_=True):
    cnts = cv2.findContours(image_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(image_color, cnts, -1, get_random_color(), 3)
    if show_:
        window_name = "contours"
        show_image_on_screen(window_name, image_color)