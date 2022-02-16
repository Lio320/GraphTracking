from collections import defaultdict
import cv2


def associate_points(bboxes1, inliers1, inliers2):
    bboxes_2_points = defaultdict(list)
    for j, grape in enumerate(sorted(bboxes1)):
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


def generate_labels(img2, bboxes_2_points, bboxes1):
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
        curr_bbox = bboxes1[bbox]
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
        cv2.rectangle(img2, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=4)
        cv2.rectangle(img2, (x_min_prev, y_min_prev), (x_max_prev, y_max_prev), color=(0, 255, 0), thickness=4)
        new_bboxes.append([x_min, y_min, x_max, y_max])
    return new_bboxes
