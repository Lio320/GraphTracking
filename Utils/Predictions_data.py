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


def transform_bboxes(bboxes):
    width = 1920
    height = 1080
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
