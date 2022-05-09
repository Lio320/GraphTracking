import numpy as np
import networkx as nx
from matplotlib import cm
import Utils.Predictions_data as predData
import Utils.Graph as Graph
import Utils.SfM_Data as sfmData
from collections import defaultdict
import cv2
from Tracker.Tracker import tracker, draw_tracked_paths
from Utils.Predictions_data import get_images, yolo2mot, get_labels, fill_labels
import matplotlib.pyplot as plt
import os
from Utils.config import config


####### CONFIGURE ALL THE PATHS WITH THE NECESSARY INFORMATION ########
camera_path, images_path, points_path, image_path, label_path, save_images, save_txt = config()

####### ASSOCIATE POINTS ID WITH IMAGES ######## (Unuseful but cool, frames IDs are wrong)
points_cam_association = sfmData.get_points_3d(points_path)

####### ASSOCIATE EACH IMAGE WITH THE POINTS SEEN ########
frame_points_association, ids_list = sfmData.get_frame_points(images_path)

####### GET IMAGES AND LABELS PATHS INSIDE THE FOLDER ########
images_paths = get_images(image_path)
labels_paths = get_labels(label_path)

####### PATHS TO SAVE THE RESULTS ########
if not os.path.exists(save_images):
    os.makedirs(save_images)

####### GENERATE DUMMY LABELS IF NOT PRESENT ########
# fill_labels(ids_list, labels_paths, label_path, 'left')

####### TAKE NEW LABELS IF PRESENT ########
labels_paths = get_labels(label_path)

####### GENERATE GRAPH ########
nodes = []
for label in labels_paths:
    ######## GET BBOXES ########
    num_nodes, bboxes = predData.get_bboxes(label)
    nodes.append(num_nodes)

nodes_mantain = nodes
G, pos, layers = Graph.generate_graph(nodes, plot=False)

prev_frame_points_to_obj = []
curr_num_nodes = 0

for i, frame_id in enumerate(ids_list):
    # Create weights as dictionary, one key for each instance in current frame
    weights = defaultdict(list)
    # Take the index of the current data to analyze
    print('Processing the frame number', frame_id)
    # Manage paths
    bbox_path = labels_paths[i]
    image_path = images_paths[i]
    image = cv2.imread(image_path)
    num_nodes, bboxes = predData.get_bboxes(bbox_path)
    bboxes = predData.yolo2pascal(image, bboxes)
    points = frame_points_association[frame_id]

    ####### RUN TRACKER ########
    G, prev_frame_points_to_obj = tracker(G, points, bboxes, prev_frame_points_to_obj, curr_num_nodes, image_path, False)
    curr_num_nodes = curr_num_nodes + num_nodes

edges = G.edges()
colors = [G[u][v]['color'] for u, v in edges]

nx.draw(G, pos, node_size=3, edge_color=colors, arrows=True, with_labels=True)

######## FILTER EDGES BY WEIGHTS ########
G = Graph.filter_graph(G)

######## EXTRACT PATHS LONGER THAN 5 ########
# At this point there is only one edge that goes out each node
nodes = G.nodes()
banned = []
paths = []
for node in nodes:
    path = []
    if node not in banned:
        Graph.explore_edge(G, node, path, banned)
    if len(path) > 4:
        paths.append(path)

nx.draw(G, pos, node_size=3, edge_color=colors, arrows=True, with_labels=True)
plt.show()

# New graph
G2, pos2, layers2 = Graph.generate_graph(nodes_mantain, plot=False)
nodes = G2.nodes()
colors = cm.jet(np.linspace(0, 1, 10))
paths_dict = {}

for i, path in enumerate(paths):
    col = i % 10
    prev = 0
    for node in path:
        if prev != 0:
            color = colors[col]
            G2.add_edge(prev, node, weight=5, color=color)
        prev = node
    paths_dict[i] = path[:-1]


edges = G2.edges()
colors = [G2[u][v]['color'] for u, v in edges]

nx.draw(G2, pos2, node_size=3, edge_color=colors, arrows=True, with_labels=True)
curr_node = 0
last_id = 0

open('tracker_bboxes.txt', 'w').close()
######## DRAW AND SAVE TRACKED PATHS WITH IMAGES ########
for i, frame_id in enumerate(ids_list):
    # Take the index of the current data to analyze
    print('Processing (again) the frame number', frame_id)
    # Manage paths
    bbox_path = labels_paths[i]
    image_path = images_paths[i]
    name = 'frame-' + str(frame_id) + '.jpg'
    num_nodes, bboxes = predData.get_bboxes(bbox_path)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if not bboxes:
        continue
    j, ids, new_bboxes, image = draw_tracked_paths(G2, image, bboxes, curr_node, paths_dict, plot=True)
    save_path = save_images + name
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, image)
    print(f'found {j} grapes in the frame {frame_id}')
    new_bboxes = yolo2mot(image, new_bboxes)
    for k, bbox in enumerate(sorted(new_bboxes)):
        with open('tracker_bboxes.txt', 'a') as f:
            f.write(str(i+1) + ', ' + str(ids[k]) + ', ' + str(bbox)[1:-1] + ', -1, -1, -1, -1' + '\n')

    curr_node += num_nodes
