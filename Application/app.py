import cv2
import numpy as np
import math
from tkinter import filedialog
from tkinter import *
import matplotlib.pyplot as plt
import sys


def findInnerBound(img):
    copy = img.copy()

    # Inverzna
    inv = cv2.bitwise_not(img)

    thresh = cv2.cvtColor(inv, cv2.COLOR_BGR2GRAY)

    # Thresholdiranje so prag 220
    ret, thresh1 = cv2.threshold(thresh, 190, 255, cv2.THRESH_BINARY)

    kernel_dilation = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(thresh1, kernel_dilation, iterations=1)

    # kelner filterov e diskutabilen
    kernel_erosion = np.ones((5, 5), np.uint8)
    # Erozija, se namaluva shumot okolu slikata, za da ostane jasna crnkata od okoto
    erosion = cv2.erode(dilation, kernel_erosion, iterations=1)

    # Baranje na konturi
    cnts, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Crtanje na konturi
    canvas_img = copy.copy()
    cv2.drawContours(canvas_img, cnts, -1, (0, 255, 0), 1)

    if len(cnts) != 0:
        c = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(copy, center, radius, (255, 0, 0), 2)
        circle = center, radius

    return copy, circle


def findUpperBound(img, gray, iris_center, iris_radius):
    gray = cv2.blur(gray, (3, 3))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 5, 5,
                               param1=105, param2=60, minRadius=90, maxRadius=120)
    min_dist = 500
    for circle in circles[0, :]:

        circle_center = int(circle[0]), int(circle[1])
        circle_radius = int(circle[2])
        distance = math.sqrt(((circle_center[0] - iris_center[0]) ** 2) + ((circle_center[1] - iris_center[1]) ** 2))
        if distance < min_dist:
            min_dist = distance
            needed_circle = int(circle[0]), int(circle[1]), circle_radius

    return needed_circle


def flatt(image, radiuspupil, radius_iris, center):
    iris_radius = radius_iris - radiuspupil

    nsamples = 360
    samples = np.linspace(0, 2.0 * np.pi, nsamples)[:-1]
    polar = np.zeros((iris_radius, nsamples))

    for r in range(iris_radius):
        for theta in samples:
            x = int((r + radiuspupil) * np.cos(theta) + center[0])
            y = int((r + radiuspupil) * np.sin(theta) + center[1])
            z = int(theta * nsamples / 2.0 / np.pi)
            if (y < image.shape[0]):
                polar[r][z] = image[y][x]

    return polar


def process_image(img):
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    kernel = np.ones((13, 13), np.uint8)
    copy = img.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    # PUPIL
    pupil_border, center = findInnerBound(img)
    pupil_center = center[0][0], center[0][1]
    pupil_radius = center[1]

    # Draw inner circle
    # cv2.circle(copy, pupil_center, pupil_radius, (255, 0, 0), 2)


    # IRIS
    iris_circle = findUpperBound(img, gray, pupil_center, pupil_radius)
    iris_center = iris_circle[0], iris_circle[1]
    iris_radius = iris_circle[2]

    # Draw outter circle
    # cv2.circle(copy, iris_center, iris_radius, (255, 0, 0), 2)


    # NORMALIZATION
    q = int(pupil_center[0])
    s = int(pupil_center[1])
    r1 = int(round(pupil_radius))
    r = int(round(iris_radius))
    center = (q, s)

    # cv2.circle(copy, (pupil_center[0], pupil_center[1]), pupil_radius, (0, 255, 0), 2)
    # cv2.circle(copy, (iris_center[0], iris_center[1]), iris_radius, (255, 0, 0), 2)

    new = flatt(gray, r1, r, center)
    return new

def useBruteForceWithRatioTest(img1, img2, kp1, kp2, des1, des2, type):

    if type == True:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
    else:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    totalDistance = 0
    for g in good:
        totalDistance += g.distance

    # cv2.drawMatchesKnn expects list of lists as matches.
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,[good],None,flags=2)

    return img3, matches, good, totalDistance

def useFLANN(img1, img2, kp1, kp2, des1, des2, type):
    # Fast Library for Approximate Nearest Neighbors
    MIN_MATCH_COUNT = 1
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 6

    if type == True:
        # Detect with ORB
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 10, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 3) #2
    else:
        # Detect with SIFT
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 10)

    # It specifies the number of times the trees in the index should be recursively traversed. Higher values gives better precision, but also takes more time
    search_params = dict(checks = 100)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    print(len(matches))

    
    
    if len(matches) > 0:
        good = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good.append([m])


        totalDistance = 0
        for g in good:
            totalDistance += g[0].distance

        
    else:
        print("No matches were found")
        totalDistance = 0
        good = []


    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

    return img3, matches, good, totalDistance



def useORB(img1, img2, matcher_type):

    # Initiate ORB detector
    orb = cv2.ORB_create()

    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    ORB = True

    if matcher_type == 1:
        return useBruteForceWithRatioTest(img1, img2, kp1, kp2, des1, des2, ORB)
    elif matcher_type == 2:
        return useFLANN(img1, img2, kp1, kp2, des1, des2, ORB)

def useBRISK(img1, img2, matcher_type):

    brisk = cv2.BRISK_create()

    # find the keypoints and descriptors with BRISK
    kp1, des1 = brisk.detectAndCompute(img1,None)
    kp2, des2 = brisk.detectAndCompute(img2,None)

    ORB = True

    if matcher_type == 1:
        return useBruteForceWithRatioTest(img1, img2, kp1, kp2, des1, des2, ORB)
    elif matcher_type == 2:
        return useFLANN(img1, img2, kp1, kp2, des1, des2, ORB)



def useSIFT(img1, img2, matcher_type):
    sift = cv2.xfeatures2d.SIFT_create()

    (kp1, des1) = sift.detectAndCompute(img1,None)
    (kp2, des2) = sift.detectAndCompute(img2, None)

    ORB = False
    if matcher_type == 1:
        return useBruteForceWithRatioTest(img1, img2, kp1, kp2, des1, des2, ORB)
    elif matcher_type == 2:
        return useFLANN(img1, img2, kp1, kp2, des1, des2, ORB)

def useFAST(img1, img2, matcher_type):

    fast = cv2.FastFeatureDetector_create()
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=32)

    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)
    copy = img1.copy()

    (kp1, des1) = brief.compute(img1, kp1)
    (kp2, des2) = brief.compute(img2, kp2)

    ORB = True
    if matcher_type == 1:
        return useBruteForceWithRatioTest(img1, img2, kp1, kp2, des1, des2, ORB)
    elif matcher_type == 2:
        return useFLANN(img1, img2, kp1, kp2, des1, des2, ORB)



if __name__ == "__main__":

    img1 = cv2.imread(filedialog.askopenfilename())
    img2 = cv2.imread(filedialog.askopenfilename())
    cv2.imshow('i1', img1)
    cv2.imshow('i2', img2)
    # cv2.waitKey(0)

    temp_norm1 = process_image(img1)
    temp_norm2 = process_image(img2)

    cv2.imwrite('i1.jpg', temp_norm1)
    cv2.imwrite('i2.jpg', temp_norm2)

    normalized_img1 = cv2.imread("i1.jpg")
    normalized_img2 = cv2.imread("i2.jpg")

    if (normalized_img1.shape[0] < normalized_img2.shape[0] or normalized_img1.shape[1] < normalized_img2.shape[1]):
        width, height = int(normalized_img1.shape[1]), int(normalized_img1.shape[0])
    else:
        width, height = int(normalized_img2.shape[1]), int(normalized_img2.shape[0])

    dim = (width, height)
    normalized_img2 = cv2.resize(normalized_img2, dim, interpolation=cv2.INTER_AREA)
    normalized_img1 = cv2.resize(normalized_img1, dim, interpolation=cv2.INTER_AREA)

    # cv2.imshow('normalized_1', normalized_img1)
    # cv2.imshow('normalized_2', normalized_img2)
    # cv2.waitKey(0)

    cv2.imwrite('i1_n.jpg', normalized_img1)
    cv2.imwrite('i2_n.jpg', normalized_img2)

    #FAST with BFMatcher with Aspect Ratio Test
    # imageFAST, matchesFAST, goodFAST, distanceFAST = useFAST(normalized_img2, normalized_img1, 1)

    # print("*****************************************************")
    # print("**** FAST with BFMatcher with Aspect Ratio Test *****")
    # print("All matches: {}".format(len(matchesFAST)))
    # print("Good matches: {}".format(len(goodFAST)))
    # print("Distance: {}".format(distanceFAST))

    # #FAST with FLANN
    # imageFAST1, matchesFAST1, goodFAST1, distanceFAST1 = useFAST(normalized_img2, normalized_img1, 2)

    # print("*****************************************************")
    # print("****************** FAST with FLANN ******************")
    # print("All matches: {}".format(len(matchesFAST1)))
    # print("Good matches: {}".format(len(goodFAST1)))
    # print("Distance: {}".format(distanceFAST1))


    # #ORB with BFMatcher with Aspect Ratio Test
    # imageORB, matchesORB, goodORB, distanceORB = useORB(normalized_img2, normalized_img1, 1)

    # print("*****************************************************")
    # print("***** ORB with BFMatcher with Aspect Ratio Test *****")
    # print("All matches: {}".format(len(matchesORB)))
    # print("Good matches: {}".format(len(goodORB)))
    # print("Distance: {}".format(distanceORB))

    # #ORB with FLANN
    # imageORB1, matchesORB1, goodORB1, distanceORB1 = useORB(normalized_img2, normalized_img1, 2)

    # print("*****************************************************")
    # print("****************** ORB with FLANN *******************")
    # print("All matches: {}".format(len(matchesORB1)))
    # print("Good matches: {}".format(len(goodORB1)))
    # print("Distance: {}".format(distanceORB1))

    # # BRISK with BFMatcher with Aspect Ratio Test
    # imageBRISK, matchesBRISK, goodBRISK, distanceBRISK = useBRISK(normalized_img2, normalized_img1, 1)

    # print("*****************************************************")
    # print("***** BRISK with BFMatcher with Aspect Ratio Test *****")
    # print("All matches: {}".format(len(matchesBRISK)))
    # print("Good matches: {}".format(len(goodBRISK)))
    # print("Distance: {}".format(distanceBRISK))

    # # BRISK with FLANN
    # imageBRISK1, matchesBRISK1, goodBRISK1, distanceBRISK1 = useBRISK(normalized_img2, normalized_img1, 2)

    # print("*****************************************************")
    # print("****************** BRISK with FLANN *****************")
    # print("All matches: {}".format(len(matchesBRISK1)))
    # print("Good matches: {}".format(len(goodBRISK1)))
    # print("Distance: {}".format(distanceBRISK1))

    # # SIFT with BFMatcher with Aspect Ratio Test
    # imageSIFT, matchesSIFT, goodSIFT, distanceSIFT = useSIFT(normalized_img2, normalized_img1, 1)

    # print("*****************************************************")
    # print("***** SIFT with BFMatcher with Aspect Ratio Test *****")
    # print("All matches: {}".format(len(matchesSIFT)))
    # print("Good matches: {}".format(len(goodSIFT)))
    # print("Distance: {}".format(distanceSIFT))

    # SIFT with FLANN
    imageSIFT1, matchesSIFT1, goodSIFT1, distanceSIFT1 = useSIFT(normalized_img2, normalized_img1, 2)

    print("*****************************************************")
    print("****************** SIFT with FLANN ******************")
    print("All matches: {}".format(len(matchesSIFT1)))
    print("Good matches: {}".format(len(goodSIFT1)))
    print("Distance: {}".format(distanceSIFT1))

    print("=================")
    if (len(goodSIFT1) >= 20):
        print("The irises come from the same identity")
    else:
        print("The irises come from different identity")

    # cv2.imshow('fast_bf',imageFAST)
    # cv2.imwrite("img_FAST_BF.jpg", imageFAST)

    # cv2.imshow('fast_flann',imageFAST1)
    # cv2.imwrite("img_FAST_FLANN.jpg", imageFAST1)

    # cv2.imshow('orb_bf',imageORB)
    # cv2.imwrite("img_ORB_BF.jpg", imageORB)

    # cv2.imshow('orb_flann',imageORB1)
    # cv2.imwrite("img_ORB_FLANN.jpg", imageORB1)

    # cv2.imshow('brisk_bf',imageBRISK)
    # cv2.imwrite("img_BRISK_BF.jpg", imageBRISK)

    # cv2.imshow('brisk_flann',imageBRISK1)
    # cv2.imwrite("img_BRISK_FLANN.jpg", imageBRISK1)

    # cv2.imshow('sift_bf',imageSIFT)
    # cv2.imwrite("img_SIFT_BF.jpg", imageSIFT)

    cv2.imshow('sift_flann',imageSIFT1)
    cv2.imwrite("img_SIFT_FLANN.jpg", imageSIFT1)
    cv2.waitKey(0)

    sys.exit()
