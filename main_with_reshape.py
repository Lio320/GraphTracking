import os
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import Utils.Predictions_data as predData
import Utils.Graph as Graph
import Utils.SfM_Data as sfmData
import time
import scipy as sp
from collections import defaultdict
import Utils.Mask_handler as maskHandler
from PIL import Image
import cv2


def most_frequent(lst):
    counter = 0
    num = lst[0]

    for n in lst:
        curr_frequency = lst.count(n)
        if curr_frequency > counter:
            counter = curr_frequency
            num = n

    return num, counter


# Paths to the files containing the SfM reconstruction
camera_path = './Colmap/New_colmaps/SfM_full_res_exhaustive/ModelText/cameras.txt'
images_path = './Colmap/New_colmaps/SfM_full_res_exhaustive/ModelText/images.txt'
points_path = './Colmap/New_colmaps/SfM_full_res_exhaustive/ModelText/points3D.txt'

####### ASSOCIATE POINTS ID WITH IMAGES ######## (Unuseful but cool, frames IDs are wrong)
points_cam_association = sfmData.get_points_3d(points_path)
first_point = []
for key in points_cam_association:
    for value in points_cam_association[key]:
        if value == '1':
            first_point.append(key)
print(len(first_point))

####### ASSOCIATE EACH IMAGE WITH THE POINTS SEEN ########
frame_points_association = sfmData.get_frame_points(images_path)
count = 0
for value in frame_points_association[1]:
    count += 1
print(count)

####### GENERATE GRAPH ########
nodes = []
for i in range(1, 2*len(frame_points_association), 2):
    idx = str(i).zfill(5)
    bbox_path = './predictions/bboxes/frame-' + idx + '.txt'
    ######## GET BBOXES ########
    num_nodes = predData.get_bboxes(bbox_path)
    nodes.append(num_nodes)

G, pos = Graph.generate_graph(nodes, plot=False)
nodes = []

# Create weights as dictionary, one key for each instance in current frame
weights = defaultdict(list)
prev_weights = defaultdict(list)
prev_frame_points_to_obj = []
curr_frame_points_to_obj = []
prev_num_nodes = 0
curr_num_nodes = 0
banned = []
max_x = []
max_y = []
print(len(frame_points_association))
for i in range(1, 2*len(frame_points_association), 2):
    weights = defaultdict(list)
    curr_frame_points_to_obj = []
    # Take the index of the current data to analyze
    idx = str(i).zfill(5)
    print('Processing the frame number', idx)
    # Manage paths
    mask_path = './predictions/masks/frame-' + idx + '.npz'
    bbox_path = './predictions/bboxes/frame-' + idx + '.txt'
    image_path = './predictions/images/frame-' + idx + '.jpg'

    ######## GET MASKS ########
    masks = predData.get_masks(mask_path)
    points = frame_points_association[i]
    total_mask = np.zeros((1080, 1920))
    ######## ASSOCIATION ########
    # Associate for the current frame, the ID of the point and the node (object) to whom is linked
    for j, grape in enumerate(masks[0][0][:]):
        # j is the index of the object in the frame
        mask = masks[:, :, j]
        mask = maskHandler.reshape_mask(mask)
        total_mask += mask

        for point in points:
            x = round(float(point[0]))-1
            y = round(float(point[1]))-1
            point_id = point[2]
            if mask[y][x]:
                # print('The 3D point of ID', point_id, 'of coordinates ' + str(x) +
                #       ' and ' + str(y) + ' belongs to the object ', j)
                curr_frame_points_to_obj.append((point_id, j))
                # plt.scatter(x, y)
                # print('Current frame points to object', curr_frame_points_to_obj)
    # print('Prev frame', prev_frame_points_to_obj, '\ncurr frame', curr_frame_points_to_obj)
    # plt.imshow(total_mask, cmap='Greys')
    # plt.show()

    ######## GET NUM NODES ########
    num_nodes = predData.get_bboxes(bbox_path)
    nodes.append(num_nodes)
    # print(curr_num_nodes)
    # curr_num_nodes = curr_num_nodes + num_nodes

    ######## UPDATE WEIGHTS, HOW MANY TIME A POINT IS CONNECTED TO THE NEXT ########
    print(len(curr_frame_points_to_obj))
    for curr_elem in curr_frame_points_to_obj:
        for prev_elem in prev_frame_points_to_obj:
            for value in prev_elem[1:]:
                if curr_elem[0] == value[0]:
                    # print('The equal values are:', curr_elem[0], value[0])
                    # print('current element', curr_elem[1])
                    weights[curr_num_nodes+curr_elem[1]].append(prev_elem[0]+value[1])

    # print(curr_num_nodes)
    curr_frame_points_to_obj.insert(0, curr_num_nodes)
    # prev_num_nodes = curr_num_nodes

    curr_num_nodes = curr_num_nodes + num_nodes
    prev_frame_points_to_obj.append(curr_frame_points_to_obj)
    if len(prev_frame_points_to_obj) > 10:
        prev_frame_points_to_obj.pop(0)
    # print('weights', weights)
    # print(curr_num_nodes)
    # print('Previous frame points to object', prev_frame_points_to_obj[-1])

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

    # if i == 9:
    #     break

edges = G.edges()
colors = [G[u][v]['color'] for u, v in edges]
# weights = [G[u][v]['weight'] for u, v in edges]

# nx.draw(G, pos, node_color='red', with_labels=False, node_size=3, verticalalignment='baseline')

# limit_x = max(max_x)
# limit_y = max(max_y)
# print('X boundary', limit_x)
# print('Y boundary', limit_y)
# counter = 0
# for value in max_x:
#     if value == limit_x:
#         counter = counter+1
# print(counter)

G = Graph.filter_graph(G)

# nx.draw(G, pos, node_size=3, edge_color=colors, width=weights, arrows=True, with_labels = True)
nx.draw(G, pos, node_size=3, edge_color=colors, arrows=True, with_labels=True)
plt.show()


# Program to find most frequent
# element in a list
