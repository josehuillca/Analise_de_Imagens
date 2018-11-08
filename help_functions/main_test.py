from HelpFunctions import *
from limiarizacao import *
from histogramas import *
from canny import *
import matplotlib.pyplot as plt
from DetectColor import DetectColor

import sys
# to work openCV on python3
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2


if __name__ == "__main__":
    image_path = "../data/scale_images"
    image_base_path = "../data/base"
    my_images = list_files_from_directory(image_path)
    print("# images: ",len(my_images))
    my_img = image_read(image_path, my_images[3])
    show_image_properties(my_img)
    
    #my_img = image_read(image_base_path, "DSC_000002905.jpg")
    '''equ_rgb = show_rgb_equalized(my_img, False)
    equ_bgr = rgb_to_bgr(equ_rgb)
    windowname = 'Original'
    cv2.namedWindow(windowname)
    hsv = cv2.cvtColor(equ_bgr, cv2.COLOR_BGR2HSV)
    cv2.imshow(windowname, hsv)
    cv2.imwrite(image_base_path + "/hsv.jpg", hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    i = 6
    #get_ship(image_path, my_images[i], image_base_path, limiar_=80)
    show_grayscale_equalized(my_img)

    print("Hello world!")

