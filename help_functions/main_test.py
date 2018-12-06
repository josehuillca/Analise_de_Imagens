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
    image_path = "../data/images"
    image_base_path = "../data/base_real"
    image_path_feature = "../data/"
    my_images = list_files_from_directory(image_path)
    # print("# images: ",len(my_images))
    # my_img = image_read(image_path, my_images[3])
    # show_image_properties(my_img)

    i = 10
    # get_ship(image_path, my_images[i], image_base_path, limiar_=80)

    # ------------------------ Actual ----------------------------------
    #for i in range(0, len(my_images)):
    img = cv2.imread(image_path+'/'+my_images[i])
    result_sea = get_only_sea(img, False)
    result_canny, no_noise_image = get_edges_noise_images(result_sea, False)
    contours = find_contours(result_canny, no_noise_image, True)
    #save_image_contours(img, contours, my_images[i])
    #print("finished image: ", my_images[i])

    # ----------------------------------------------------------
    # segment_sky(image_base_path, "/DSC_000009279.jpg")
    # get_contourns_mountain(image_base_path, "/DSC_000002905.jpg")
    # segment_sea(image_base_path)

    # my_segment_sea(image_base_path + "/DSC_000000380.jpg")
    #img = cv2.imread(image_base_path + "/DSC_000009279.jpg")
    #show_histogram_rgb(img, title="Title", color=('k'))
    #plt.show()
    print("finished...!")

