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

    testeando(image_base_path, limiar=65)


    #display_multiple_images([my_img], ["Imagem Original:"], 1, 1)
    #img_r, img_g, img_b = single_band_representation(my_img, False)
    #show_histogram_rgb(my_img)
    #eq_r, eq_g, eq_b = histograms_equalization(img_r, img_g, img_b, show=False, compare_hist=False)

    #equ_rgb = show_rgb_equalized(my_img,False)
    #equ_bgr = rgb_to_bgr(equ_rgb)
    #show_hsv_equalized(my_img)

    #limiarizacao(path=image_path, img_cor="sudoku.jpg")
    #limiarizacao2(path=image_path, img_cor=my_images[3])
    # initial data
    BGR_low = [43, 6, 0]
    BGR_high = [135, 248, 115]
    #my_img = image_read(image_base_path, "DSC_000002905.jpg")
    #equ_rgb = show_rgb_equalized(my_img, False)
    #plt.imsave(image_base_path + "/equ_rgb.jpg", equ_rgb)
    '''detectC = DetectColor(image_path + "/" + my_images[2], BGR_low, BGR_high)
    detectC.trackingColor_and_palette()'''
    
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

    # --- AQUI SEGMENTAMOS EL CIELO --------
    '''my_img = image_read(image_base_path, "DSC_000002905.jpg")
    #equ_rgb = show_rgb_equalized(my_img, False)
    neg_cor = negativo_grises(my_img)
    plt.imsave(image_base_path + "/negative.jpg", neg_cor)
    result = otsu(image_base_path, "negative.jpg")
    result = use_erode_dilate(result)
    cv2.imwrite(image_base_path + "/otsu.jpg", result)'''
    # --------------- aqui sumamos el cielo con otra imagen
    '''img = cv2.imread(image_base_path + "/otsu.jpg")
    neg_img = negativo_grises(img)
    my_img = cv2.imread(image_path + "/" + "DSC_000000105.jpg")#2

    #------- Calculamos la diferencia absoluta de las dos imagenes
    diff_total = cv2.absdiff(my_img, neg_img)
    #hsv = cv2.cvtColor(diff_total, cv2.COLOR_BGR2HSV)
    cv2.imshow("aaaa", diff_total)
    cv2.imwrite(image_base_path + "/diff_total.jpg", diff_total)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    # ----------- escala de grises
    '''my_img = cv2.imread(image_base_path + "/" + "diff_total.jpg", 0)
    result = otsu(image_base_path, "diff_total.jpg")
    cv2.imshow("aaaa", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    # ------- convertir el cielo azul
    '''img = cv2.imread(image_base_path + "/otsu.jpg")
    my_img = cv2.imread(image_path + "/" + my_images[2])  # 2
    i = 0
    while i < img.shape[0]:
        j = 0
        while j < img.shape[1]:
            if img[i,j][0] == 0:
                my_img[i, j] = [255, 0, 0]
            j += 1
        i += 1
    hsv = cv2.cvtColor(my_img, cv2.COLOR_BGR2HSV)
    cv2.imshow("aaaa", hsv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    # --- canny
    #segment_sky(image_base_path, "DSC_000002905.jpg")
    '''merge_sky(image_base_path, "DSC_000009279.jpg", image_base_path)

    #main_canny(image_path , "/" + my_images[3], 46)
    result = get_canny(image_base_path, "/img_temp_diff.jpg", 65)#34

    result = use_erode_dilate(result) #dilate, luego erode, se modifico eso
    cv2.imwrite(image_base_path + "/result.jpg", result)'''

    '''merge_sky(image_path, my_images[0], image_base_path, "/segment_sea.jpg")
    #convertir el cielo y la montana a color del oceano
    main_canny(image_base_path, "/img_temp_diff.jpg", 46)'''


    '''bgr = get_media_color(image_base_path, "DSC_000009279.jpg")
    change_sky_mountain(image_base_path, "/diff_total.jpg", bgr)
    main_canny(image_base_path, "/mediacolor.jpg", 46)#60'''

    #----------suma y se queda con la media
    # x =0, y=530
    '''ratio = 3
    for x in range(0, my_img.shape[0]):
        for y in range(530, my_img.shape[1]):'''

    '''my_img = cv2.imread(image_base_path + "/" + "diff_total.jpg")
    my_img = rgb_to_bgr(my_img)
    a = cv2.cvtColor(my_img, cv2.COLOR_BGR2HSV)
    a[:, :, 2] += 80
    a = cv2.cvtColor(a, cv2.COLOR_HSV2BGR)
    #a = quantizacao(my_img, n=2)

    #plt.imshow(a, cmap='gray')
    #plt.show()

    cv2.imshow("aaaa", a)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''

    print("Hello world!")

