import numpy as np
import networkx as nx
from matplotlib import cm
import matplotlib.pyplot as plt
import Utils.Data_Management as predData
import Utils.Graph as Graph
import Utils.SfM_Data as sfmData
from collections import defaultdict
import cv2
from Tracker.Tracker import draw_tracked_paths, tracker_memory
from Utils.Data_Management import get_images, yolo2mot, get_labels
import os
import yaml


####### CONFIGURE ALL THE PATHS WITH THE NECESSARY INFORMATION ########
with open("./Config/config_track.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

camera_path = cfg['camera_path']
images_path = cfg['images_path']
points_path = cfg['points_path']
image_path = cfg['image_path']
label_path = cfg['label_path']
save_images = cfg['save_images']
save_txt = cfg['save_txt']

####### ASSOCIATE POINTS ID WITH IMAGES ########
points_cam_association = sfmData.get_points_3d(points_path)

####### ASSOCIATE EACH IMAGE WITH THE POINTS SEEN ########
frame_points_association, ids_list = sfmData.get_frame_points(images_path)
images_paths = get_images(image_path)
labels_paths = get_labels(label_path)

####### PATHS TO SAVE THE RESULTS ########
if not os.path.exists(save_images):
    os.makedirs(save_images)

####### GENERATE DUMMY LABELS IF NOT PRESENT ########
# fill_labels(ids_list, labels_paths, label_path, 'frame_')

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
    G, prev_frame_points_to_obj = tracker_memory(G, points, bboxes, prev_frame_points_to_obj, curr_num_nodes, image_path, False)
    curr_num_nodes = curr_num_nodes + num_nodes

edges = G.edges()
colors = [G[u][v]['color'] for u, v in edges]

######## FILTER EDGES BY WEIGHTS ########
G = Graph.filter_graph(G)

######## EXTRACT PATHS LONGER THAN 5 FRAMES ########
# At this point there is only one edge that goes out each node
nodes = G.nodes()
print(type(nodes))
banned = []
paths = []
for node in nodes:
    if node not in banned:
        path, banned = Graph.explore_edge(G, node, banned)
    if len(path) > 5:
        paths.append(path)

######## GENERATE NEW GRAPH WITH PATHS ONLY LONGER THAN 5 FRAMES ########
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
plt.show()

######## DRAW AND SAVE TRACKED PATHS WITH IMAGES ########
curr_node = 0
last_id = 0
open(save_txt, 'w').close()
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
    j, ids, new_bboxes, image = draw_tracked_paths(G2, image, bboxes, curr_node, paths_dict, plot=False)
    save_path = save_images + name
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_path, image)
    print(f'found {j} grapes in the frame {frame_id}')
    new_bboxes = yolo2mot(image, new_bboxes)
    for k, bbox in enumerate(sorted(new_bboxes)):
        with open(save_txt, 'a') as f:
            f.write(str(i+1) + ', ' + str(ids[k]) + ', ' + str(bbox)[1:-1] + ', -1, -1, -1, -1' + '\n')
    curr_node += num_nodes
