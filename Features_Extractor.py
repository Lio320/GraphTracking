import numpy as np
import cv2
from Utils.Predictions_data import get_images, get_labels, get_bboxes, transform_bboxes
from Utils.plot_with_bboxes import plot_yolo
from operator import itemgetter


####### GET IMAGES AND LABELS PATHS INSIDE THE FOLDER ########
images_paths = get_images('./video_demo_frames/')
labels_paths = get_labels('./final/labels/')


# Detect the keypoints using SURF Detector
minHessian = 400
detector = cv2.xfeatures2d.SURF_create(minHessian)
# detector = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

img1 = cv2.imread(images_paths[0])
_, bboxes = get_bboxes(labels_paths[0])
plot_yolo(img1, bboxes)
bboxes = transform_bboxes(bboxes)
kp1, des1 = detector.detectAndCompute(img1, None)
print(type(des1), len(des1), des1)
points = []
descriptors = []
for keypoint, descriptor in zip(kp1, des1):
    x = keypoint.pt[0]
    y = keypoint.pt[1]
    size = keypoint.size
    for j, bbox in enumerate(sorted(bboxes, key=itemgetter(0))):
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            points.append(keypoint)
            descriptors.append(descriptor)
            break

kp1 = tuple(points)
des1 = np.array(descriptors)
print(type(des1), len(des1), des1)

img2 = cv2.imread(images_paths[1])
_, bboxes = get_bboxes(labels_paths[1])
plot_yolo(img2, bboxes)
bboxes = transform_bboxes(bboxes)
kp2, des2 = detector.detectAndCompute(img2, None)

points = []
descriptors = []
for keypoint, descriptor in zip(kp2, des2):
    x = keypoint.pt[0]
    y = keypoint.pt[1]
    size = keypoint.size
    for j, bbox in enumerate(sorted(bboxes, key=itemgetter(0))):
        if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
            points.append(keypoint)
            descriptors.append(descriptor)
            break
kp2 = tuple(points)
des2 = np.array(descriptors)

matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
match_img = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
cv2.imshow('Matches', match_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# img2 = cv2.imread(images[1])
# kernel = np.ones((5,5),np.float32)/25
# blur = cv2.GaussianBlur(img1,(5,5),0)

# images_dict = {}
# minHessian = 400
# ######## SURF #########
# detector = cv2.xfeatures2d.SURF_create(minHessian)
# bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
#
# for i, image_path in enumerate(images_paths):
#     image = cv2.imread(image_path)
#     _, bboxes = get_bboxes(labels_paths[i])
#     plot_yolo(image, bboxes)
#     bboxes = transform_bboxes(bboxes)
#
#     # Convert the image to gray.
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     # Compute SURF features on gray image
#     kp1, des1 = detector.detectAndCompute(gray, None)
#     kp1 = list(kp1)
#     print(len(kp1))
#     points = []
#     for keypoint in kp1[:]:
#         x = keypoint.pt[0]
#         y = keypoint.pt[1]
#         size = keypoint.size
#         print(x, y)
#         for j, bbox in enumerate(sorted(bboxes, key=itemgetter(0))):
#             print(bbox)
#             if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
#                 points.append(keypoint)
#                 break
#
#     print(len(points))
#     kp1 = tuple(points)
#     image1 = cv2.drawKeypoints(image, kp1, image)
#     cv2.imshow('Matches', image1)
#     cv2.waitKey(0)
#     # for keypoint in kp1:
#     #     print(f'x {round(keypoint.pt[0])} y {round(keypoint.pt[1])}')
#
#     images_dict[i] = [image1, kp1, des1]
#     if i-1 in images_dict:
#         image = cv2.imread(images_paths[i-1])
#         kp = images_dict[i-1][1]
#         des = images_dict[i-1][2]
#         matches = bf.match(des, des1)
#         matches = sorted(matches, key=lambda x: x.distance)
#         match_img = cv2.drawMatches(image, kp, image1, kp1, matches, None)
#         # cv2.imshow('Matches', match_img)
#
#     if cv2.waitKey(0) & 0xff == 27:
#         cv2.destroyAllWindows()
#     if i == 1:
#         break

####### GET CONTOURS #########
# # First obtain the threshold using the greyscale image
# ret, thresh = cv2.threshold(gray, 100, 250, cv2.THRESH_BINARY)
# # visualize the binary image
# cv2.imshow('Binary image', thresh)
# contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# image = cv2.drawContours(image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
# lineType=cv2.LINE_AA)
# cv2.imshow('Contours', image)

####### FAST DETECTOR #########
# fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
# kp = fast.detect(image, None)
# image = cv2.drawKeypoints(image, kp, None, color=(255, 0, 0))
# cv2.imshow("output", image)

####### HOUGH TRANSFORM #########
# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 41, param1=70, param2=30, minRadius=1, maxRadius=30)
# # convert the (x, y) coordinates and radius of the circles to integers
# circles = np.round(circles[0, :]).astype("int")
# for (x, y, r) in circles:
#     # draw the circle in the output image, then draw a rectangle
#     # corresponding to the center of the circle
#     cv2.circle(image, (x, y), r, (0, 255, 0), 4)
#     cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
#     # show the output image
# cv2.imshow("output", image)

######## HOG #########
# _, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
#                    cells_per_block=(1, 1), visualize=True, multichannel=True)
# cv2.imshow('HoG', hog_image)

######## BLOB DETECTOR #########
# Setup SimpleBlobDetector parameters.
# params = cv2.SimpleBlobDetector_Params()
# Change thresholds
# params.minThreshold = 10
# params.maxThreshold = 200

# Filter by Area.
# params.filterByArea = True
# params.minArea = 1500

# Filter by Circularity
# params.filterByCircularity = True
# params.minCircularity = 0.1

# Filter by Convexity
# params.filterByConvexity = True
# params.minConvexity = 0.87

# Filter by Inertia
# params.filterByInertia = True
# params.minInertiaRatio = 0.01

# detector = cv2.SimpleBlobDetector_create()
# keypoints = detector.detect(gray)
# im_with_keypoints = cv2.drawKeypoints(gray, keypoints, np.array([]), (0, 0, 255),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv2.imshow('BLOB', im_with_keypoints)

######## CORNER HARRIS #########
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, 2, 3, 0.04)
# dst = cv2.dilate(dst, None)
# image[dst>0.01*dst.max()]=[0,0,255]
# cv2.imshow('Harris', image)

######## ORB DETECTOR #########
# orb = cv2.ORB_create()
# kp, des = orb.detectAndCompute(image, None)
# image = cv2.drawKeypoints(image, kp, image)
# cv2.imshow('ORB', image)

######## CANNY EDGES #########
# edges = cv2.Canny(image, 100, 200)
# cv2.imshow('edges', edges)
