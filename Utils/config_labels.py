
def config():
    ####### PATHS IMAGES AND YOLO LABELS ########
    image_path = './Detection_frames/Video_ionut/Images/'
    label_path = './Detection_frames/Video_ionut/labels/'
    ####### DEFINE SKIPS ########
    skips = [2, 5, 8, 11, 14]
    ####### PATH WHERE TO SAVE RESULTS ########
    results_path = './Pseudolabels/Video_ionut/FeaturesLabels/FeaturesLabels_skip'
    return image_path, label_path, skips, results_path
