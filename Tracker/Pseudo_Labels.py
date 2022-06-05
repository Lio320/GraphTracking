from collections import defaultdict
import cv2
import sys
import os
from Utils.Data_Management import pascal2yolo


def most_frequent(lst):
    counter = 0
    num = lst[0]
    for n in lst:
        curr_frequency = lst.count(n)
        if curr_frequency > counter:
            counter = curr_frequency
            num = n
    return num, counter


def tracker(points, bboxes, curr_num_nodes, image, plot_points=False, idx='', path=''):
    bboxes_2_points = defaultdict(list)
    curr_frame_points_to_obj = []
    ######## MANAGE PATHS FOR THE RESULTS ########
    img_name = 'frame' + str(idx).zfill(4) + '.jpg'
    img_path = path + 'images/'
    label_name = 'frame' + str(idx).zfill(4) + '.txt'
    label_path = path + 'labels/'

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    ######## ASSOCIATION ########
    # Associate for the current frame, the ID of the point and the node (object) to whom is linked
    for j, grape in enumerate(sorted(bboxes)):
        x_min = grape[0]
        y_min = grape[1]
        x_max = grape[2]
        y_max = grape[3]

        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=4)
        for point in points:
            x = round(float(point[0]))-1
            y = round(float(point[1]))-1

            point_id = point[2]
            if x_min <= x <= x_max and y_min <= y <= y_max:
                curr_frame_points_to_obj.append((point_id, j))
                bboxes_2_points[j].append(point_id)
                if plot_points:
                    cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)

    ####### SAVE LABELS ########
    yolo_bboxes = pascal2yolo(image, bboxes)
    with open(label_path + label_name, 'w') as f:
        for bbox in yolo_bboxes:
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[2] < 0:
                bbox[2] = 0
            if bbox[3] < 0:
                bbox[3] = 0
            f.write('0 ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')

    ####### SAVE IMAGE ########
    cv2.imwrite(img_path + img_name, image)

    ####### DECOMMENT TO SHOW IMAGE WITH POINTS ########
    # cv2.imshow("plot", image)
    # k = cv2.waitKey(33) & 0xFF
    # if k == 27:
    #     sys.exit()

    ######## UPDATE CURRENT AND PREVIOUS POINTS ########
    curr_frame_points_to_obj.insert(0, curr_num_nodes)
    prev_frame_points_to_obj = curr_frame_points_to_obj

    return prev_frame_points_to_obj, bboxes, bboxes_2_points


def generate_pseudo_labels(points, bboxes, bboxes_2_points, image, idx, path=''):
    new_bboxes_2_points = defaultdict(list)
    bboxes = sorted(bboxes)

    ######## MANAGE PATHS FOR THE RESULTS ########
    img_name = 'frame' + str(idx).zfill(4) + '.jpg'
    img_path = path + 'images/'
    label_name = 'frame' + str(idx).zfill(4) + '.txt'
    label_path = path + 'labels/'

    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    ####### UPDATE 3D POINTS IN BOUNDING BOX ########
    for point in points:
        x = round(float(point[0])) - 1
        y = round(float(point[1])) - 1
        # cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        point_id = point[2]
        for bbox in bboxes_2_points:
            for bbox_point in bboxes_2_points[bbox]:
                if bbox_point == point_id:
                    new_bboxes_2_points[bbox].append(point)
    new_bboxes = []
    for bbox in sorted(new_bboxes_2_points):
        x_list = []
        y_list = []
        # if len(new_bboxes_2_points[bbox]) > 3:
        for point in new_bboxes_2_points[bbox]:
            x = round(float(point[0])) - 1
            y = round(float(point[1])) - 1
            cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            x_list.append(x)
            y_list.append(y)
            cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        x_cen = sum(x_list) / len(x_list)
        y_cen = sum(y_list) / len(y_list)
        # cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=4)
        curr_bbox = bboxes[bbox]
        width = curr_bbox[2] - curr_bbox[0]
        length = curr_bbox[3] - curr_bbox[1]
        x_min_prev = curr_bbox[0]
        y_min_prev = curr_bbox[1]
        x_max_prev = curr_bbox[2]
        y_max_prev = curr_bbox[3]
        x_min = int(round(x_cen - (width/2)))
        y_min = int(round(y_cen - (length/2)))
        x_max = int(round(x_cen + (width/2)))
        y_max = int(round(y_cen + (length/2)))
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=4)
        cv2.rectangle(image, (x_min_prev, y_min_prev), (x_max_prev, y_max_prev), color=(0, 255, 0), thickness=4)
        new_bboxes.append([x_min, y_min, x_max, y_max])

    ####### SAVE LABELS ########
    yolo_bboxes = pascal2yolo(image, new_bboxes)
    with open(label_path + label_name, 'w') as f:
        for bbox in yolo_bboxes:
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[2] < 0:
                bbox[2] = 0
            if bbox[3] < 0:
                bbox[3] = 0
            f.write('0 ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')
    ####### SAVE IMAGE ########
    cv2.imwrite(img_path + img_name, image)

    ####### DECOMMENT TO SHOW IMAGE WITH POINTS ########
    # cv2.imshow("plot", image)
    # k = cv2.waitKey(33) & 0xFF
    # if k == 27:
    #     sys.exit()
