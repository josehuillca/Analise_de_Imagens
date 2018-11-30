import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/usr/local/lib/python3.6/site-packages')
import cv2

def feature_extraction(path, image1, image2):
    img1 = cv2.imread(path + image1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path + image2, cv2.IMREAD_GRAYSCALE)
    #img1 = cv2.Canny(img1, 30, 210, 3)
    #img2 = cv2.Canny(img2, 30, 210, 3)

    '''sift = cv2.xfeatures2d.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(img, None)

    img = cv2.drawKeypoints(img, keypoints, None)'''

    #ORB detector
    orb = cv2.ORB_create()
    #orb = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # Brute Force Matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x:x.distance)

    matching_result = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None)

    # BFMatcher with default params
    '''bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])
    # cv2.drawMatchesKnn expects list of lists as matches.
    matching_result = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)'''


    #img1 = cv2.drawKeypoints(img1, kp1, None)
    #img2 = cv2.drawKeypoints(img2, kp2, None)

    cv2.imshow("Image1", img1)
    cv2.imshow("Image2", img2)
    cv2.imshow("Matching result", matching_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
