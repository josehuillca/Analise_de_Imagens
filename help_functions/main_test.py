from HelpFunctions import *
from limiarizacao import *
from histogramas import *
from feature_detection import *
from canny import *
from DetectColor import DetectColor
from mainFuntions import *

import sys
# to work openCV on python3
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2


if __name__ == "__main__":
    image_path = "../data/scale_images"
    image_base_path = "../data/base"
    image_path_feature = "../data/"
    my_images = list_files_from_directory(image_path)
    # print("# images: ",len(my_images))
    # my_img = image_read(image_path, my_images[3])
    # show_image_properties(my_img)

    i = 3
    # get_ship(image_path, my_images[i], image_base_path, limiar_=80)


    # ------------------------ Actual ----------------------------------
    img = cv2.imread(image_path+'/'+my_images[i])
    # result_absdiff = absdiff_img_sea(img, True)
    result_sea = get_only_sea(img, False)
    result_canny, no_noise_image = get_edges_noise_images(result_sea, False)
    find_contours(result_canny, no_noise_image)
    #find_and_colored_contours(result_canny, no_noise_image)

    # ----------------------------------------------------------
    # segment_sky(image_base_path, "/DSC_000009279.jpg")
    # get_contourns_mountain(image_base_path, "/DSC_000009279.jpg")
    # segment_sea(image_base_path)

    print("Hello world!")

