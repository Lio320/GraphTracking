import numpy as np
import networkx as nx
from matplotlib import cm
import Utils.Predictions_data as predData
import Utils.Graph as Graph
import Utils.SfM_Data as sfmData
from collections import defaultdict
import cv2
from Tracker import tracker, draw_tracked_paths
from Utils.Predictions_data import get_images
import matplotlib.pyplot as plt


# Paths to the files containing the SfM reconstruction
camera_path = './Colmap/New_colmaps/SfM_full_res_exhaustive/ModelText/cameras.txt'
images_path = './Colmap/New_colmaps/SfM_full_res_exhaustive/ModelText/images.txt'
points_path = './Colmap/New_colmaps/SfM_full_res_exhaustive/ModelText/points3D.txt'

####### ASSOCIATE POINTS ID WITH IMAGES ######## (Unuseful but cool, frames IDs are wrong)
points_cam_association = sfmData.get_points_3d(points_path)

####### ASSOCIATE EACH IMAGE WITH THE POINTS SEEN ########
frame_points_association, ids_list = sfmData.get_frame_points(images_path)

####### GET IMAGES AND LABELS PATHS INSIDE THE FOLDER ########
images = get_images('./video_demo_frames/')

####### GENERATE GRAPH ########
nodes = []
for i in range(1, 2*len(frame_points_association), 2):
    idx = str(i).zfill(5)
    bbox_path = './final/labels/frame-' + idx + '.txt'
    ######## GET BBOXES ########
    num_nodes, bboxes = predData.get_bboxes(bbox_path)
    nodes.append(num_nodes)

nodes_mantain = nodes
G, pos, layers = Graph.generate_graph(nodes, plot=False)

prev_frame_points_to_obj = []
curr_num_nodes = 0

for i in range(1, 2*len(frame_points_association), 2):
    # Create weights as dictionary, one key for each instance in current frame
    weights = defaultdict(list)
    # Take the index of the current data to analyze
    idx = str(i).zfill(5)
    print('Processing the frame number', idx)
    # Manage paths
    bbox_path = './final/labels/frame-' + idx + '.txt'
    image_path = './video_demo_frames/frame-' + idx + '.jpg'
    num_nodes, bboxes = predData.get_bboxes(bbox_path)
    bboxes = predData.transform_bboxes(bboxes)
    points = frame_points_association[i]
    ####### RUN TRACKER ########
    G, prev_frame_points_to_obj = tracker(G, points, bboxes, prev_frame_points_to_obj, curr_num_nodes, image_path, False)
    curr_num_nodes = curr_num_nodes + num_nodes
    # if i == 41:
    #     break

edges = G.edges()
colors = [G[u][v]['color'] for u, v in edges]
# weights = [G[u][v]['weight'] for u, v in edges]

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

# nx.draw(G, pos, node_size=3, edge_color=colors, width=weights, arrows=True, with_labels = True)
# nx.draw(G, pos, node_size=3, edge_color=colors, arrows=True, with_labels=True)
# plt.show()

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
ids = []

######## DRAW AND SAVE TRACKED PATHS WITH IMAGES ########
for i in range(1, 2*len(frame_points_association), 2):
    # Take the index of the current data to analyze
    idx = str(i).zfill(5)
    print('Processing (again) the frame number', idx)
    # Manage paths
    bbox_path = './final/labels/frame-' + idx + '.txt'
    image_path = './video_demo_frames/frame-' + idx + '.jpg'
    name = 'frame-' + idx + '.jpg'
    num_nodes, bboxes = predData.get_bboxes(bbox_path)
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    j = draw_tracked_paths(G2, image, bboxes, curr_node, paths_dict, ids, name, plot=True)
    print(f'found {j} grapes in the frame {idx}')
    curr_node += num_nodes
