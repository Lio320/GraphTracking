import Utils.Predictions_data as predData
import Utils.Graph as Graph
import Utils.SfM_Data as sfmData
from collections import defaultdict
import cv2
from Tracker.Pseudo_Labels import generate_pseudo_labels, tracker
from Utils.Predictions_data import get_images, get_labels

####### PATHS TO FILES THAT CONTAIN SFM RECONSTRUCTION ########
camera_path = 'Colmap/Video_ionut/ModelText/cameras.txt'
images_path = './Colmap/Video_ionut/ModelText/images.txt'
points_path = './Colmap/Video_ionut/ModelText/points3D.txt'

####### GET IMAGES AND LABELS PATHS INSIDE THE FOLDER ########
image_path = './Detection_frames/Video_ionut/Images/'
label_path = './Detection_frames/Video_ionut/labels/'

images_paths = get_images(image_path)
labels_paths = get_labels(label_path)

####### ASSOCIATE POINTS ID WITH IMAGES ######## (Unuseful but cool, frames IDs are wrong)
points_cam_association = sfmData.get_points_3d(points_path)

####### ASSOCIATE EACH IMAGE WITH THE POINTS SEEN ########
frame_points_association, ids_list = sfmData.get_frame_points(images_path)

skips = [2, 5, 8, 11, 14]
####### DEFINE PATH WHERE TO SAVE RESULTS AND SKIP ########
for skip in skips:
    save_path = './Pseudolabels/Video_ionut/SfmLabels/SfmLabels_skip' + str(skip) + '/'

    prev_frame_points_to_obj = []
    curr_num_nodes = 0
    for i, frame_id in enumerate(ids_list):
        ####### MANAGE PATHS ########
        bbox_path = labels_paths[i]
        image_path = images_paths[i]
        image = cv2.imread(image_path)
        num_nodes, bboxes = predData.get_bboxes(bbox_path)
        bboxes = predData.yolo2pascal(image, bboxes)
        points = frame_points_association[frame_id]
        ####### RUN TRACKER ONE TIME EVERY TWO FRAMES ########
        if i % skip:
            print('in_ labels', i)
            generate_pseudo_labels(points, prev_bboxes, bboxes_2_points, image, i+1, save_path)

        else:
            print('in track', i)
            prev_frame_points_to_obj, prev_bboxes, bboxes_2_points = tracker(points, bboxes,
                                                                             curr_num_nodes,
                                                                             image, True,
                                                                             i+1, save_path)

        curr_num_nodes = curr_num_nodes + num_nodes

