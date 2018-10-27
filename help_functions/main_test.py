from HelpFunctions import *

if __name__ == "__main__":
    image_path = "../data/scale_images"
    my_images = list_files_from_directory(image_path)
    my_img = image_read(image_path, my_images[0])
    show_image_properties(my_img)
    #display_multiple_images([my_img], ["Imagem Original:"], 1, 1)
    img_r, img_g, img_b = single_band_representation(my_img, show=False)
    #show_histogram_rgb(my_img)
    histograms_equalization(img_r, img_g, img_b, compare_hist=True)

    print("Hello world!")

