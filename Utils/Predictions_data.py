import os
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter


def get_images(path):
    """
    Get images path inside a folder
    Keyword arguments:
    path: path to the file containing the images
    """
    images = []
    for image in sorted(os.listdir(path)):
        images.append(path + image)
    return images


def get_labels(path):
    """
    Get images path inside a folder
    Keyword arguments:
    path: path to the file containing the images
    """
    labels = []
    for label in sorted(os.listdir(path)):
        labels.append(path + label)
    return labels


def fill_labels(ids_list, labels_paths, label_path, name='frame'):
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
    Get masks from a single file
    Keyword arguments:
    path: path to the file containing the mask
    """
    masks = np.load(path)
    lst = masks.files
    for item in lst:
        return masks[item]


def get_bboxes(path):
    """
    Get number of bboxes from a single file
    Keyword arguments:
    path: path to the file containing the bboxes
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
