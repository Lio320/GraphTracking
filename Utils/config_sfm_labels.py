
def config():
    ####### PATHS TO SfM RECONSTRUCTION MODEL ########
    camera_path = './Colmap/Video_validation/ModelText/cameras.txt'
    images_path = './Colmap/Video_validation/ModelText/images.txt'
    points_path = './Colmap/Video_validation/ModelText/points3D.txt'
    ####### GET IMAGES AND LABELS PATHS INSIDE THE FOLDER ########
    image_path = './Detection_frames/Video_validation/Images/'
    label_path = './Detection_frames/Video_validation/aug/labels/'
    ####### PATHS TO SAVE THE RESULTS (IMAGES AND TRACKING) ########
    results_path = './Pseudolabels/Video_ionut/SfmLabels/SfmLabels_skip'
    skips = [2, 5, 8, 11, 14]
    return camera_path, images_path, points_path, image_path, label_path, results_path, skips
