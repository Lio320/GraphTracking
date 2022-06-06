import Utils.Data_Management as predData
import Utils.SfM_Data as sfmData
import cv2
from Tracker.Pseudo_Labels import generate_pseudo_labels, tracker
from Utils.Data_Management import get_images, get_labels
import yaml

####### PATHS TO FILES THAT CONTAIN SFM RECONSTRUCTION ########
with open("./Config/config_sfm_labels.yaml", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

camera_path = cfg['camera_path']
images_path = cfg['images_path']
points_path = cfg['points_path']
image_path = cfg['image_path']
label_path = cfg['label_path']
results_path = cfg['results_path']
skips = cfg['skips']

images_paths = get_images(image_path)
labels_paths = get_labels(label_path)

####### ASSOCIATE POINTS ID WITH IMAGES ######## (Unuseful but cool, frames IDs are wrong)
points_cam_association = sfmData.get_points_3d(points_path)

####### ASSOCIATE EACH IMAGE WITH THE POINTS SEEN ########
frame_points_association, ids_list = sfmData.get_frame_points(images_path)

####### DEFINE PATH WHERE TO SAVE RESULTS AND SKIP ########
for skip in skips:
    save_path = results_path + str(skip) + '/'
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

        ####### RUN TRACKER ONE TIME EVERY SKIP FRAMES ########
        if i % skip:
            print('Generatin pseudo labels at frame', i)
            generate_pseudo_labels(points, prev_bboxes, bboxes_2_points, image, i+1, save_path)

        else:
            print('Picking detector prediciton at frame', i)
            prev_frame_points_to_obj, prev_bboxes, bboxes_2_points = tracker(points, bboxes,
                                                                             curr_num_nodes,
                                                                             image, True,
                                                                             i+1, save_path)

        curr_num_nodes = curr_num_nodes + num_nodes

