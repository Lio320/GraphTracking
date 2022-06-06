from collections import defaultdict
import cv2


def associate_points(bboxes, inliers1, inliers2):
    """
    Associate to the bounding boxes the features inliers matched from the previous
    to the current frame

    Args:
        bboxes (list):          list of all bounding boxes detected in the image
        inliers1 (list):        list of all inliers in the previous frame, associated 
                                to the ones in the current one
        inliers2 (list):        list of all inliers in the current frame, associated to
                                those in the previous frame, therefore inliers1 and
                                inliers2 have the same dimesions and the point in position
                                inliers1[i] is the one corresponding to inliers2[i]
    Returns:
        bboxes_2_points (dict): dictionary containing the lists of features points
                                associated to the k-th bounding box. Has the dimension 
                                of all the bounding boxes present in the frame
    """
    bboxes_2_points = defaultdict(list)
    for j, grape in enumerate(sorted(bboxes)):
        x_min = grape[0]
        y_min = grape[1]
        x_max = grape[2]
        y_max = grape[3]
        for k, point in enumerate(inliers1):
            x = round(float(point.pt[0])) - 1
            y = round(float(point.pt[1])) - 1
            if x_min <= x <= x_max and y_min <= y <= y_max:
                bboxes_2_points[j].append(inliers2[k])
    return bboxes_2_points


def generate_labels(img, bboxes_2_points, bboxes):
    """
    Function that propagates the bounding boxes from the previous frame to the current
    one according to the features matched between the two frames

    Args:
        img (image):            image on which the new boundng boxes have to be generated
        bboxes_2_points (dict): dictionary containing the lists of features points
                                associated to the k-th bounding box. Has the dimension 
                                of all the bounding boxes present in the frame
        bboxes (list):          list containing the bounding boxes present in the previous frame, that
                                have to be propagated in the current one
    Returns:
        new_bboxes (list):      list of the new bounding boxes generated in the current frame
    """
    new_bboxes = []
    for bbox in sorted(bboxes_2_points):
        x_list = []
        y_list = []
        # if len(bboxes_2_points[bbox]) > 3:
        for point in bboxes_2_points[bbox]:
            x = round(float(point.pt[0])) - 1
            y = round(float(point.pt[1])) - 1
            x_list.append(x)
            y_list.append(y)
        x_cen = sum(x_list) / len(x_list)
        y_cen = sum(y_list) / len(y_list)
        curr_bbox = bboxes[bbox]
        width = curr_bbox[2] - curr_bbox[0]
        length = curr_bbox[3] - curr_bbox[1]
        x_min_prev = curr_bbox[0]
        y_min_prev = curr_bbox[1]
        x_max_prev = curr_bbox[2]
        y_max_prev = curr_bbox[3]
        x_min = int(round(x_cen - (width / 2)))
        y_min = int(round(y_cen - (length / 2)))
        x_max = int(round(x_cen + (width / 2)))
        y_max = int(round(y_cen + (length / 2)))
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=4)
        cv2.rectangle(img, (x_min_prev, y_min_prev), (x_max_prev, y_max_prev), color=(0, 255, 0), thickness=4)
        new_bboxes.append([x_min, y_min, x_max, y_max])
    return new_bboxes
