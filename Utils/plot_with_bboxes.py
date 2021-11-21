import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import sys


# Yolo format: [x_center, y_center, width, height] (normalized)
def plot_yolo(image, bboxes, colors=[], labels=[], name='', plot=False):
    # fig, ax = plt.subplots()
    # DPI = fig.get_dpi()
    # fig.set_size_inches(1920.0 / float(DPI), 1080.0 / float(DPI))
    # ax.imshow(image)
    # width = len(image)
    # height = len(image[0])
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
            cv2.rectangle(image, (x_min, y_min), (x_min + w, y_min + h), 4)
            # cv2.putText(image, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4)
    # display image with opencv or any operation you like
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if plot:
        cv2.imshow("plot", image)
        cv2.waitKey(0)
        # k = cv2.waitKey(33) & 0xFF
        # if k == 27:
        #     sys.exit()
    if name:
        path = './Track_frames2/' + name
        cv2.imwrite(path, image)
