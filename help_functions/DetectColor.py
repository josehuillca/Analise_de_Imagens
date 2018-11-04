import sys
# para que funcione opencv en python3
sys.path.append('/usr/local/lib/python3.6/site-packages')
# import the necessary packages
import numpy as np
import cv2
import  matplotlib.pyplot as plt
from collections import deque
import argparse
import imutils


class DetectColor:
    blue, green, red, s = (0, 0, 0, 0)  # 's' is a value of switch
    switch = '0:low \n1:high'
    B, G, R = ('B', 'G', 'R')

    B_low, G_low, R_low = (0, 0, 0)
    B_high, G_high, R_high = (0, 0, 0)

    change_to_low = False
    change_to_high = False

    imgpath_current = ""

    def __init__(self,path_img, BGR_low=(0, 0, 0), BGR_high=(0, 0, 0)):
        self.B_low, self.G_low, self.R_low = BGR_low
        self.B_high, self.G_high, self.R_high = BGR_high
        self.imgpath_current = path_img

    # empty funtion to palette bar
    def empty_func(self, x):
        pass

    # Create Palette Bar
    def create_trackbar_bgr(self, windowname):
        cv2.createTrackbar(self.B, windowname, 0, 255, self.empty_func)
        cv2.createTrackbar(self.G, windowname, 0, 255, self.empty_func)
        cv2.createTrackbar(self.R, windowname, 0, 255, self.empty_func)

        # Create switch to change low and high color
        cv2.createTrackbar(self.switch, windowname, 0, 1, self.empty_func)

    def get_trackbar_bgr(self, windowname):
        self.blue = cv2.getTrackbarPos(self.B, windowname)
        self.green = cv2.getTrackbarPos(self.G, windowname)
        self.red = cv2.getTrackbarPos(self.R, windowname)
        self.s = cv2.getTrackbarPos(self.switch, windowname)


    def set_trackbar_bgr(self, windowname, B, G, R):
        cv2.setTrackbarPos(self.B, windowname, B)
        cv2.setTrackbarPos(self.G, windowname, G)
        cv2.setTrackbarPos(self.R, windowname, R)

    # a image_mask le pasamos erosion y dilatacion
    # Erode and dilate the image to further amplify features.
    def use_erode_dilate(self, image):
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        erode = cv2.erode(image, kernel_erode)
        dilate = cv2.dilate(erode, kernel_dilate)

        erode = cv2.erode(dilate, kernel_erode)
        dilate = cv2.dilate(erode, kernel_dilate)

        return dilate

        '''kernel = np.ones((5, 5), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)'''

    # draw circle how point
    def draw_point(self, imagen, x, y, r, color=(0, 0, 255)):
        cv2.circle(imagen, (x, y), r, color, -1)

    def read_file(self, filename):
        # Using the newer with construct to close the file automatically.
        with open(filename) as f:
            data = f.readlines()
        list = []
        for n, line in enumerate(data, 1):
            txt = line.rstrip()
            txt = txt.split(' ')
            list.append(txt)
        return list

    # La funcion convierte los valores decimales x_dec(de rango de 0 a 1) a
    # valores reales, x_real
    def convert_to_real(self, x_real, x_dec):
        return (x_dec*x_real)/1

    # img_coord[0] = name of image
    # img_coord[1], img_coord[2] = coordenada x, y, respectivamente
    def trackingColor_and_palette(self, use_erode_dilate=True):
        windowname = self.imgpath_current  #'Original'
        windowmask = 'imageMask'
        windowpalette = 'OpenCV BGR Color Palette'
        img1 = np.zeros((150, 400, 3), np.uint8)
        cv2.namedWindow(windowname)
        cv2.namedWindow(windowmask)
        cv2.namedWindow(windowpalette)

        self.create_trackbar_bgr(windowpalette)

        frame = cv2.imread(self.imgpath_current)
        # cv2.imshow(windowname, frame)
        if frame is None:
            return

        # initialize the list of tracked points, the frame counter,
        # and the coordinate deltas
        pts = deque()  # (maxlen=args["buffer"]) , default 32
        counter = 0
        (dX, dY) = (0, 0)
        direction = ""

        # frame = imutils.resize(frame, width=600)  # not necessary to do resize
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)

        while True:
            # here: if is webcam, we read the video capture

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Blue Color
            low = np.array([self.B_low, self.G_low, self.R_low])
            high = np.array([self.B_high, self.G_high, self.R_high])

            image_mask = cv2.inRange(hsv, low, high)

            if use_erode_dilate:
                image_mask = self.use_erode_dilate(image_mask)
                # use_erode_dilate = False

            # cv2.imshow(windowmask, image_mask)

            # find contours in the mask and initialize the current
            # (x, y) center of the ball
            ###cnts = cv2.findContours(image_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            # im2, cnts, hierarchy = cv2.findContours(image_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            ####cv2.drawContours(frame, cnts, -1, (0, 255, 0), 3)
            center = None
            # only proceed if at least one contour was found

            # show the frame to our screen and increment the frame counter
            cv2.imshow(windowname, hsv)

            cv2.imshow(windowmask, image_mask)
            # self.draw_point(frame, 0, 0, 34)
            cv2.imshow(windowpalette, img1)
            if cv2.waitKey(1) == 27:  # 27 es la tecla esc
                break

            self.get_trackbar_bgr(windowpalette)

            # Change to low and to high colo
            if self.s == 0:
                if not self.change_to_low:  # change_to_low == False
                    self.set_trackbar_bgr(windowpalette, self.B_low, self.G_low, self.R_low)
                    print("low color, BGR:(", self.B_low, self.G_low, self.R_low)
                    self.change_to_low = True
                    self.change_to_high = False
                self.B_low, self.G_low, self.R_low = (self.blue, self.green, self.red)
            else:
                if not self.change_to_high:
                    self.set_trackbar_bgr(windowpalette, self.B_high, self.G_high, self.R_high)
                    print("high color, BGR:(", self.B_high, self.G_high, self.R_high)
                    self.change_to_high = True
                    self.change_to_low = False
                    self.B_high, self.G_high, self.R_high = (self.blue, self.green, self.red)

            img1[:] = [self.blue, self.green, self.red]
            text1 = "BGR:(" + str(self.blue) + ", " + str(self.green) + ", " + str(self.red) + ")"
            cv2.putText(img1, text1, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255 - self.blue, 255 - self.green, 255 - self.red))


        cv2.destroyAllWindows()
        #print(radius)

