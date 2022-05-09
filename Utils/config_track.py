
def config():
    ####### PATHS TO SfM RECONSTRUCTION MODEL ########
    camera_path = './Colmap/Video_validation/ModelText/cameras.txt'
    images_path = './Colmap/Video_validation/ModelText/images.txt'
    points_path = './Colmap/Video_validation/ModelText/points3D.txt'
    ####### GET IMAGES AND LABELS PATHS INSIDE THE FOLDER ########
    image_path = './Detection_frames/Video_validation/Images/'
    label_path = './Detection_frames/Video_validation/aug/labels/'
    ####### PATHS TO SAVE THE RESULTS (IMAGES AND TRACKING) ########
    save_images = './Video_Results/Video_validation/Tracked_Images_memory/'
    save_txt = './Uva_tracker.txt'
    return camera_path, images_path, points_path, image_path, label_path, save_images, save_txt