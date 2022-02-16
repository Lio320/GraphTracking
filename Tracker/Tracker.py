from collections import defaultdict
import Utils.Predictions_data as predData
import cv2
import sys
import Utils.plot_with_bboxes as plt_bboxes
from operator import itemgetter


def most_frequent(lst):
    counter = 0
    num = lst[0]
    for n in lst:
        curr_frequency = lst.count(n)
        if curr_frequency > counter:
            counter = curr_frequency
            num = n

    return num, counter


def tracker(G, points, bboxes, prev_frame_points_to_obj, curr_num_nodes, image_path, plot_points=False):
    weights = defaultdict(list)
    bboxes_2_points = defaultdict(list)
    curr_frame_points_to_obj = []
    ######## ASSOCIATION ########
    # Associate for the current frame, the ID of the point and the node (object) to whom is linked
    if plot_points:
        image = cv2.imread(image_path)
    for j, grape in enumerate(bboxes):
        for point in points:
            x = round(float(point[0]))-1
            y = round(float(point[1]))-1
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
    return G, prev_frame_points_to_obj


def tracker_memory(G, points, bboxes, prev_frame_points_to_obj, curr_num_nodes, image_path, plot_points=False):
    weights = defaultdict(list)
    bboxes_2_points = defaultdict(list)
    curr_frame_points_to_obj = []
    to_remove = []
    ######## ASSOCIATION ########
    # Associate for the current frame, the ID of the point and the node (object) to whom is linked
    if plot_points:
        image = cv2.imread(image_path)
    for j, grape in enumerate(bboxes):
        for point in points:
            x = round(float(point[0]))-1
            y = round(float(point[1]))-1
            point_id = point[2]
            if grape[0] <= x <= grape[2] and grape[1] <= y <= grape[3]:
                curr_frame_points_to_obj.append((point_id, curr_num_nodes+j))
                bboxes_2_points[j].append(point_id)
                if plot_points:
                    cv2.circle(image, (x, y), radius=4, color=(0, 0, 255), thickness=-1)
    if plot_points:
        cv2.imshow("plot", image)
        k = cv2.waitKey(33) & 0xFF
        if k == 27:
            sys.exit()
    ######## UPDATE WEIGHTS, HOW MANY TIME A POINT IS CONNECTED TO THE NEXT ########
    for curr_elem in curr_frame_points_to_obj:
        for value in prev_frame_points_to_obj:
            if curr_elem[0] == value[0]:
                weights[curr_elem[1]].append(value[1])
                # print('are they equal?', curr_elem, value)
                if value in prev_frame_points_to_obj:
                    prev_frame_points_to_obj.remove(value)
                    to_remove.append(value[0])
    for element in prev_frame_points_to_obj:
        # if element[0] in to_remove:
        prev_frame_points_to_obj.remove(element)
    ######## UPDATE POINTS SEEN PREVIOUSLY, UPDATE IF SEEN, KEEP IF NOT SEEN ########
    # curr_frame_points_to_obj.insert(0, curr_num_nodes)
    prev_frame_points_to_obj.extend(curr_frame_points_to_obj)
    # if len(prev_frame_points_to_obj) > 10:
    #     prev_frame_points_to_obj.pop(-1)
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
    return G, prev_frame_points_to_obj


def draw_tracked_paths(G, image, bboxes, curr_node, paths_dict, plot=False):
    bboxes = sorted(bboxes, key=itemgetter(0))
    plotted_bboxes = []
    colors = []
    ids = []
    for j, bbox in enumerate(bboxes):
        out_edges = list(G.out_edges(curr_node + j, data=True))
        # in_edges = G.in_edges(curr_node + j, data=True)
        if out_edges:
            plotted_bboxes.append(bbox)
            colors.append(out_edges[0][2]['color'])
        for key in paths_dict:
            if curr_node + j in paths_dict[key]:
                ids.append(key)
    plt_bboxes.plot_yolo(image, plotted_bboxes, colors, ids, plot)
    return j, ids, plotted_bboxes, image
