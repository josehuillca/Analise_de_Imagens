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
    image_path = "../data/images/imgs"
    image_base_path = "../data/base_real"
    image_path_feature = "../data/"
    my_images = list_files_from_directory(image_path)
    # print("# images: ",len(my_images))
    # my_img = image_read(image_path, my_images[3])
    # show_image_properties(my_img)

    i = 4
    # get_ship(image_path, my_images[i], image_base_path, limiar_=80)
    # ------------------------ Actual ----------------------------------
    # for i in range(0, len(my_images)):
    '''img = cv2.imread(image_base_path + '/' + my_images[i])
    show_image_on_screen("ciza", cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    print("Name image:", my_images[i])
    result_sea = get_only_sea(img, True)
    result_canny, no_noise_image = get_edges_noise_images(result_sea, False)
    contours = find_contours(result_canny, no_noise_image, False)
    save_image_contours(img, contours, my_images[i])
    print("finished image: ", my_images[i])'''


    #img = cv2.imread(image_path + "/DSC_000025585.jpg")
    #show_image_on_screen("title", img)
    list_feautures_classes = extract_feature_of_classes(show_=False)
    my_print(['centroid', 'area', 'perimeter', 'aspect_ratio'], np.array(list_feautures_classes), title="list_feautures")

    '''img = cv2.imread(image_path + '/' + my_images[i])
    print("name: ", my_images[i])
    features = extract_features(img, show_=False)

    compare_features(list_feautures_classes, features)'''

    # ----------------------------------------------------------
    # segment_sky(image_base_path, "/DSC_000009279.jpg")
    # get_contourns_mountain(image_base_path, "/DSC_000002905.jpg")
    # segment_sea(image_base_path)

    # my_segment_sea(image_base_path + "/DSC_000000380.jpg")
    #img = cv2.imread(image_base_path + "/DSC_000009279.jpg")
    #show_histogram_rgb(img, title="Title", color=('k'))
    #plt.show()
    print("finished...!")

