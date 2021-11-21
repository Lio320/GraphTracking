import os
import numpy as np
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from operator import itemgetter


def get_points_3d(path):
    """
    Get 3D points seen in the SfM reconstruction
    Keyword arguments:
    path: path to the file containing the points
    """
    # Create dictionary that associate to each 3D point, the images (therefore the cameras) from where it is seen.
    p2c_association = {}
    # Points start from line index 3 (before there are only comments)
    with open(path) as f:
        lines = f.readlines()
        lines = lines[3:]
        for i, line in enumerate(lines):
            line = line.split(' ')
            ID = line[0]
            X = line[1]
            Y = line[2]
            Z = line[3]
            # print('X:', X, '\nY:', Y, '\nZ:', Z)
            track = line[8:]
            for j in range(0, int(len(track)), 2):
                ImageID = track[j]
                Point2D_IDX = track[j+1]
                # If key already exists,
                if ID in p2c_association:
                    p2c_association[ID].append(ImageID)
                else:
                    p2c_association[ID] = [ImageID]
    return p2c_association


def get_frame_points(path):
    """
    Get for each frame the 2D point projection
    Keyword arguments:
    path: path to the file containing the points
    returns:
    - c2p_association -> dictionary with association of frames and points
    - ids_list -> ordered list with the ids of the images
    """
    ids_list = []
    c2p_association = {}
    with open(path) as f:
        lines = f.readlines()
        lines = lines[4:]
        for i, line in enumerate(lines):
            temp_points = []
            line = line.split(' ')
            # Even number line
            if i % 2 == 0:
                image_ID = line[0]
                camera_ID = line[8]
                frame_id = int(line[9].strip()[6:11])
            # Odd number line
            else:
                # Remove '\n' from last element
                line[-1] = line[-1].strip()
                for j in range(0, int(len(line)), 3):
                    X = line[j]
                    Y = line[j+1]
                    Point3D_ID = line[j+2]
                    if Point3D_ID != '-1':
                        temp_points.append((X, Y, Point3D_ID))
                c2p_association[frame_id] = temp_points
                ids_list.append(frame_id)
    return c2p_association, sorted(ids_list)
