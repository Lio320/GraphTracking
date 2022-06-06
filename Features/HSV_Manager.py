import cv2
import numpy as np


def nothing(x):
    pass


def hsv_tracker(image):
    """
    function that if called generates a GUI to change the values of Hue, Saturation and Values

    Args:
        image (image):  image on which the HSV values have to be manipulated through the GUI
    """
    cv2.namedWindow("Tracking")
    cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
    cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
    cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    while True:
        l_h = cv2.getTrackbarPos("LH", "Tracking")
        l_s = cv2.getTrackbarPos("LS", "Tracking")
        l_v = cv2.getTrackbarPos("LV", "Tracking")

        u_h = cv2.getTrackbarPos("UH", "Tracking")
        u_s = cv2.getTrackbarPos("US", "Tracking")
        u_v = cv2.getTrackbarPos("UV", "Tracking")

        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

        mask = cv2.inRange(hsv, l_b, u_b)

        res = cv2.bitwise_and(image, image, mask=mask)

        # cv2.imshow("frame", image)
        # cv2.imshow("mask", mask)
        cv2.imshow("res", res)

        key = cv2.waitKey(1)
        if key == 27:
            break


def hsv_extractor(image, l_h, l_s, l_v, u_h, u_s, u_v):
    """
    function to extract the hue, saturation and value from an image

    Args:
        image (image):      list containing the keypoints extracted in the previous frame
        l_h (float):        lower value of hue
        l_s (float):        lower value of saturation
        l_v (float):        lower bound of value
        u_h (float):        upper bound of hue
        u_s (float):        upper bound of saturation
        u_v (float)         upper bound of value
    Returns:
        res  (cv2 image):   image with the hsv values extracted
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, l_b, u_b)
    res = cv2.bitwise_and(image, image, mask=mask)
    return res


if __name__ == 'main':
    path = './Detection_frames/Santos_video/images/frame_0000.jpg'
    hsv_tracker(path)
