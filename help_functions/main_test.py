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
    i = 4
    list_feautures_classes = extract_feature_of_classes(show_=False)
    my_print(['cx', 'cy', 'area', 'perimeter', 'aspect_ratio'], np.array(list_feautures_classes), title="list_feautures")

    for i in range(10):
        img = cv2.imread(image_path + '/' + my_images[i])
        print("name: ", my_images[i])
        features = extract_features(img, show_=False)

        compare_features(list_feautures_classes, features)

    print("finished...!")

