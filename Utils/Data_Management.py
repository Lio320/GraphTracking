import os
import numpy as np
from operator import itemgetter
import cv2

def get_images(path):
    """
    Get images inside a given path

    Args:
        path (str):     path to the file containing the images

    Returns:
        images (list):  list containing the paths of all the images in the folder
    """
    images = []
    for image in sorted(os.listdir(path)):
        images.append(path + image)
    return images


def get_labels(path):
    """
    Get labels (detection annotations) inside a given path

    Args:
        path (str):     path to the file containing the labels

    Returns:
        labels (list):  list containing the paths of all the labels in the folder
    """
    labels = []
    for label in sorted(os.listdir(path)):
        labels.append(path + label)
    return labels


def fill_labels(ids_list, labels_paths, label_path, name='frame'):
    """
    Function that fills holes in the annotations, meaning that if an annotation is missing for a given frame
    (no instances detected) it automatically generates an empty txt file with the name corresponding to the image
    it is associated with

    Args:
        ids_list (list): path to the file containing the labels
        labels_paths
        label_path
        name
    """
    for i, frame_id in enumerate(ids_list):
        print(i)
        print(len(labels_paths))
        label_id = [int(i) for i in labels_paths[i-1] if i.isdigit()]
        label_id = int(''.join([str(x) for x in label_id[-4:]]))
        if label_id != int(frame_id):
            print(label_id, frame_id)
            idx = label_path + name + str(frame_id).zfill(4) + '.txt'
            open(idx, 'a').close()


def get_masks(path):
    """
    Function to retrieve the masks inside a path

    Args:
        path (str):             path to the file (npz/npy) containing the segmentation masks
    Returns:
        mask[item] (array):     array containing the segmentation masks in the path
    """
    masks = np.load(path)
    lst = masks.files
    for item in lst:
        return masks[item]


def get_bboxes(path):
    """
    Function to get the bounding boxes containe in a file

    Args:
        path (str):         path to the file containing the bounding boxes annotations
    Returns:
        num_nodes (int):    number of instances (bounding boxes), corresponding to the number of nodes in the graph to create
        bboxes (list):      list containing the bounding boxes (annotations) in the path file
    """
    with open(path, 'r') as f:
        bboxes = []
        lines = f.readlines()
        num_nodes = 0
        for j, line in enumerate(lines):
            new_line = []
            line = line[1:].strip()
            line = line.split(' ')
            num_nodes += 1
            for element in line:
                new_line.append(float(element))
            bboxes.append(new_line)
    f.close()
    return num_nodes, bboxes


####### FROM YOLO TO PASCAL VOC ########
# [x_min, y_min, x_max, y_max] Not normalized
def yolo2pascal(image, bboxes):
    """
    Convert YOLO annotation to Pascal

    Args:
        image (image):      image on which the bounding boxes have to be 
        bboxes (list):      list containing the bounding boxes in the image (in YOLO format)
    Returns:
        new_bboxes (list):  list containing the new bounding boxes (in Pascal format)
    """
    width = len(image[0])
    height = len(image)
    new_bboxes = []
    for bbox in bboxes:
        # Convert bboxes from tuple to list
        bbox = list(bbox)
        # Denormalization process
        x_cen = bbox[0] * width
        y_cen = bbox[1] * height
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)
        # Obtain x_min and y_min
        x_min = int(x_cen - (w / 2))
        y_min = int(y_cen - (h / 2))
        x_max = int(x_cen + (w / 2))
        y_max = int(y_cen + (h / 2))
        new_bboxes.append([x_min, y_min, x_max, y_max])
        new_bboxes = sorted(new_bboxes, key=itemgetter(0))
    return new_bboxes


####### FROM PASCAL VOC TO YOLO ########
# [x_center, y_center, width, height] Normalized
def pascal2yolo(image, bboxes):
    """
    Convert Pascal annotations to YOLO

    Args:
        image (image):      image on which the bounding boxes have to be 
        bboxes (list):      list containing the bounding boxes in the image (in Pascal format)
    Returns:
        new_bboxes (list):  list containing the new bounding boxes (in YOLO format)
    """
    width = len(image[0])
    height = len(image)
    new_bboxes = []
    for bbox in bboxes:
        # Convert bboxes from tuple to list
        bbox = list(bbox)
        # Normalization process
        x_min = bbox[0] / width
        y_min = bbox[1] / height
        x_max = bbox[2] / width
        y_max = bbox[3] / height
        w = x_max - x_min
        h = y_max - y_min
        x_cen = x_min + (w / 2)
        y_cen = y_min + (h / 2)
        new_bboxes.append([x_cen, y_cen, w, h])
        new_bboxes = sorted(new_bboxes, key=itemgetter(0))
    return new_bboxes


####### FROM PASCAL VOC TO MOT CHALLENGE########
# [bb_left, bb_top, w, h] Not normalized
def yolo2mot(image, bboxes):
    """
    Convert YOLO annotations to MOT challenge format (for tracking evaluation)

    Args:
        image (image):      image on which the bounding boxes have to be 
        bboxes (list):      list containing the bounding boxes in the image (in YOLO format)
    Returns:
        new_bboxes (list):  list containing the new bounding boxes (in MOT format)

    For more informations about the MOT format go to the following link:
    https://github.com/JonathonLuiten/TrackEval
    """
    width = len(image[0])
    height = len(image)
    new_bboxes = []
    for i, bbox in enumerate(bboxes):
        # Convert bboxes from tuple to list
        bbox = list(bbox)
        # Denormalization process
        x_cen = bbox[0] * width
        y_cen = bbox[1] * height
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)

        # Obtain x_min and y_min
        bb_left = int(x_cen - (w / 2))
        bb_top = int(y_cen - (h / 2))
        new_bboxes.append([bb_left, bb_top, w, h])
        new_bboxes = sorted(new_bboxes, key=itemgetter(0))
    return new_bboxes


# Yolo format: [x_center, y_center, width, height] (normalized)
def plot_yolo(image, bboxes, colors=[], labels=[], plot=False):
    """
    Function to plot the images in YOLO format, personalization options are provided in the plot,
    to provide a better and more clear representation of the tracker images

    Args:
        image (cv2 image): image to plot
        bboxes (list): list of the bounding boxes to plot
        colors (list): list of the colors to assing to each bounding box
        labels (list): list of the labels (numbers) to assign to each bounding box
        plot (bool): choose to visulize or not the image
    """
    width = len(image[0])
    height = len(image)
    for i, bbox in enumerate(bboxes):
        # Convert bboxes from tuple to list
        bbox = list(bbox)

        # Denormalization process
        x_cen = bbox[0] * width
        y_cen = bbox[1] * height
        w = int(bbox[2] * width)
        h = int(bbox[3] * height)

        # Obtain x_min and y_min
        x_min = int(x_cen - (w/2))
        y_min = int(y_cen - (h/2))

        # Draw rectangle
        if colors and labels:
            color = colors[i]*256
            label = str(labels[i])
            cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h), color, 4)
            cv2.putText(image, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        else:
            cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h), [255, 0, 0], 4)
            # cv2.putText(image, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    # display image with opencv or any operation you like
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if plot:
        cv2.imshow("plot", image)
        cv2.waitKey(0)

