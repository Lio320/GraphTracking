import numpy as np
import cv2
from operator import itemgetter
from math import sqrt


def detect_features(image, bboxes=[], detector='surf', plot=False):
    if detector == 'surf':
        # Detect the keypoints using SURF Detector
        minHessian = 100
        detector = cv2.xfeatures2d.SURF_create(minHessian)
    if detector == 'sift':
        detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(image, None)
    if bboxes:
        points = []
        new_descriptors = []
        ######## TAKE ONLY POINTS IN THE BBOXES  ########
        for keypoint, descriptor in zip(keypoints, descriptors):
            x = keypoint.pt[0]
            y = keypoint.pt[1]
            size = keypoint.size
            for j, bbox in enumerate(sorted(bboxes, key=itemgetter(0))):
                if bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]:
                    points.append(keypoint)
                    new_descriptors.append(descriptor)
                    break
        keypoints = tuple(points)
        descriptors = np.array(new_descriptors)
    return keypoints, descriptors


def features_matcher(kp1, kp2, des1, des2, good_ratio=0.7, matcher='brute_force'):
    if matcher == 'brute_force':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matched_points1 = []
    matched_points2 = []
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    num_matches = len(matches)

    # Filter to obtain only good matches
    good = []
    for m in matches:
        if m.distance < good_ratio:
            good.append(m)
            matched_points1.append(kp1[m.queryIdx])
            matched_points2.append(kp2[m.trainIdx])
    return good, matched_points1, matched_points2


def ransac(kp1, kp2, good, mp1, mp2, MIN_MATCH_COUNT=10, inlier_threshold=10.0):
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        return None, None, None
    inliers1 = []
    inliers2 = []
    good_matches = []
    for i, m in enumerate(mp1):
        col = np.ones((3, 1), dtype=np.float64)
        col[0:2, 0] = m.pt

        col = np.dot(M, col)
        col /= col[2, 0]
        dist = sqrt(pow(col[0, 0] - mp2[i].pt[0], 2) +
                    pow(col[1, 0] - mp2[i].pt[1], 2))

        if dist < inlier_threshold:
            good_matches.append(cv2.DMatch(len(inliers1), len(inliers2), 0))
            inliers1.append(mp1[i])
            inliers2.append(mp2[i])

    return good_matches, inliers1, inliers2

