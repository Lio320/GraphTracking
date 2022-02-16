from __future__ import print_function
import cv2
from Utils.Predictions_data import get_images, get_labels, get_bboxes, yolo2pascal, pascal2yolo
from Utils.plot_with_bboxes import plot_yolo
from Features.HSV_Manager import hsv_tracker, hsv_extractor
from Features.Features_Manager import detect_features, features_matcher, ransac
from Features.Features_labels import generate_labels, associate_points
import os

####### GET IMAGES AND LABELS PATHS INSIDE THE FOLDER ########
# images_paths = get_images('./Detection_frames/Santos_video/Images/')
# labels_paths = get_labels('./Detection_frames/Santos_video/labels/')

images_paths = get_images('./Detection_frames/Test_frames/Images/')
labels_paths = get_labels('./Detection_frames/Test_frames/labels/')

skips = [2, 5, 8, 11, 14]
####### DEFINE PATH WHERE TO SAVE RESULTS AND SKIP ########
for skip in skips:
    save_path = './Pseudolabels/Test_frames/FeaturesLabels/FeaturesLabels_skip' + str(skip) + '/'
    img_path = save_path + 'images/'
    label_path = save_path + 'labels/'
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)

    for i in range(len(images_paths)):
        print('Processing frame', i, 'out of', len(images_paths))
        ######## MANAGE PATHS FOR THE RESULTS ########
        img_name = 'frame' + str(i).zfill(4) + '.jpg'
        label_name = 'frame' + str(i).zfill(4) + '.txt'

        if not i % skip:
            img1 = cv2.imread(images_paths[i])
            # hsv_tracker(img1)
            # img1 = hsv_extractor(img1, 15, 88, 46, 35, 255, 255)
            _, bboxes1 = get_bboxes(labels_paths[i])
            # plot_yolo(img1, bboxes1)
            bboxes1 = yolo2pascal(img1, bboxes1)
            kp1, des1 = detect_features(img1, bboxes1)
            cv2.drawKeypoints(img1, kp1, img1)

            ####### SAVE LABELS ########
            yolo_bboxes = pascal2yolo(img1, bboxes1)
            with open(label_path + label_name, 'w') as f:
                for bbox in yolo_bboxes:
                    if bbox[0] < 0:
                        bbox[0] = 0
                    if bbox[1] < 0:
                        bbox[1] = 0
                    if bbox[2] < 0:
                        bbox[2] = 0
                    if bbox[3] < 0:
                        bbox[3] = 0
                    f.write('0 ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')

            ####### SAVE IMAGE ########
            cv2.imwrite(img_path + img_name, img1)
            # cv2.imshow('image', img1)
            # if cv2.waitKey(0) & 0xff == 27:
            #     cv2.destroyAllWindows()

        else:
            ######## SECOND IMAGE ########
            img2 = cv2.imread(images_paths[i])
            # img2 = hsv_extractor(img2, 15, 88, 46, 35, 255, 255)
            kp2, des2 = detect_features(img2)
            if i == 401:
                img3 = cv2.drawKeypoints(img2, kp2, None)
                img3 = cv2.resize(img3, (1280, 720))
                cv2.imshow('image', img3)
                if cv2.waitKey(0) & 0xff == 27:
                    cv2.destroyAllWindows()

            good, matched_points1, matched_points2 = features_matcher(kp1, kp2, des1, des2)

            good_matches, inliers1, inliers2 = ransac(kp1, kp2, good, matched_points1, matched_points2)

            if good_matches is None:
                continue
            bboxes_2_points = associate_points(bboxes1, inliers1, inliers2)
            new_bboxes = generate_labels(img2, bboxes_2_points, bboxes1)
            yolo_bboxes = pascal2yolo(img2, new_bboxes)
            ####### SAVE LABELS ########
            with open(label_path + label_name, 'w') as f:
                for bbox in yolo_bboxes:
                    if bbox[0] < 0:
                        bbox[0] = 0
                    if bbox[1] < 0:
                        bbox[1] = 0
                    if bbox[2] < 0:
                        bbox[2] = 0
                    if bbox[3] < 0:
                        bbox[3] = 0
                    f.write('0 ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + '\n')
            ####### SAVE IMAGE ########
            cv2.imwrite(img_path + img_name, img2)



