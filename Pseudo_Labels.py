from collections import defaultdict
import cv2
import sys


def most_frequent(lst):
    counter = 0
    num = lst[0]
    for n in lst:
        curr_frequency = lst.count(n)
        if curr_frequency > counter:
            counter = curr_frequency
            num = n

    return num, counter


def tracker2(G, points, bboxes, prev_frame_points_to_obj, curr_num_nodes, image_path, plot_points=False, name=''):
    weights = defaultdict(list)
    bboxes_2_points = defaultdict(list)
    curr_frame_points_to_obj = []
    ######## ASSOCIATION ########
    # Associate for the current frame, the ID of the point and the node (object) to whom is linked
    if plot_points:
        image = cv2.imread(image_path)
    for j, grape in enumerate(sorted(bboxes)):
        x_min = grape[0]
        y_min = grape[1]
        x_max = grape[2]
        y_max = grape[3]
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=4)
        for point in points:
            x = round(float(point[0]))-1
            y = round(float(point[1]))-1
            cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
            point_id = point[2]
            if grape[0] <= x <= grape[2] and grape[1] <= y <= grape[3]:
                curr_frame_points_to_obj.append((point_id, j))
                bboxes_2_points[j].append(point_id)
                if plot_points:
                    cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
    if plot_points:
        cv2.imshow("plot", image)
        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            sys.exit()
    name = 'frame-' + name + '.jpg'
    path = './Track_frames3/' + name
    cv2.imwrite(path, image)
    ######## UPDATE WEIGHTS, HOW MANY TIME A POINT IS CONNECTED TO THE NEXT ########
    for curr_elem in curr_frame_points_to_obj:
        for value in prev_frame_points_to_obj[1:]:
            if curr_elem[0] == value[0]:
                weights[curr_num_nodes+curr_elem[1]].append(prev_frame_points_to_obj[0]+value[1])

    curr_frame_points_to_obj.insert(0, curr_num_nodes)
    prev_frame_points_to_obj = curr_frame_points_to_obj

    ######## CONNECT EDGES THROUGH WEIGHTS ########
    # Weights contain how many instances connect two nodes
    for key in weights:
        if weights[key]:
            highest, weight = most_frequent(weights[key])
        else:
            continue
        if weight < 2:
            G.add_edge(highest, key, weight=weight, color='blue')
            continue
        if weight < 5:
            G.add_edge(highest, key, weight=weight, color='deepskyblue')
            continue
        if weight < 7:
            G.add_edge(highest, key, weight=weight, color='lime')
            continue
        if weight < 10:
            G.add_edge(highest, key, weight=weight, color='yellow')
            continue
        if weight < 12:
            G.add_edge(highest, key, weight=weight, color='orangered')
            continue
        if weight < 15:
            G.add_edge(highest, key, weight=weight, color='red')
            continue
        else:
            G.add_edge(highest, key, weight=weight, color='fuchsia')
    return G, prev_frame_points_to_obj, bboxes, bboxes_2_points


def generate_pseudo_labels(points, bboxes, bboxes_2_points, image, name, max_min=True, prev=False):
    new_bboxes_2_points = defaultdict(list)
    bboxes = sorted(bboxes)
    ####### UPDATE 3D POINTS IN BOUNDING BOX ########
    for point in points:
        x = round(float(point[0])) - 1
        y = round(float(point[1])) - 1
        cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
        point_id = point[2]
        for bbox in bboxes_2_points:
            for bbox_point in bboxes_2_points[bbox]:
                if bbox_point == point_id:
                    new_bboxes_2_points[bbox].append(point)
    if max_min:
        for bbox in sorted(new_bboxes_2_points):
            x_min = 1920
            y_min = 1080
            x_max = 0
            y_max = 0
            for point in new_bboxes_2_points[bbox]:
                x = round(float(point[0])) - 1
                y = round(float(point[1])) - 1
                cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
                if x < x_min:
                    x_min = x
                if y < y_min:
                    y_min = y
                if x > x_max:
                    x_max = x
                if y > y_max:
                    y_max = y
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=4)
    elif prev:
        for bbox in sorted(new_bboxes_2_points):
            x_list = []
            y_list = []
            for point in new_bboxes_2_points[bbox]:
                x = round(float(point[0])) - 1
                y = round(float(point[1])) - 1
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
            x_min = int(round(x_cen + (width/2)))
            y_min = int(round(y_cen + (length/2)))
            x_max = int(round(x_cen - (width/2)))
            y_max = int(round(y_cen - (length/2)))
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 0, 255), thickness=4)
            cv2.rectangle(image, (x_min_prev, y_min_prev), (x_max_prev, y_max_prev), color=(0, 255, 0), thickness=4)
    ####### DECOMMENT TO SHOW IMAGE WITH POINTS ########
    cv2.imshow("plot", image)
    k = cv2.waitKey(33) & 0xFF
    if k == 27:
        sys.exit()
    name = 'frame-' + name + '.jpg'
    path = './Track_frames3/' + name
    cv2.imwrite(path, image)
