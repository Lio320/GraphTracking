# GraphTracking

Master thesis of Leonardo Saraceni at Sapienza University of Rome in collaboration with Canopies Project.

* Tracking algorithm that exploits detection (YOLO) and 3D information 
* Pseudo-labels generation for semi-supervised learning pipeline

The paper can be obtained to the following link:
* Link to the paper

Scripts:
* Bboxes_Tracker: Tracking algorithm for grape instances, requires a COLMAP model of the scene and the predictions obtained by a detector (YOLO in the case of this project). Need to modify the configuration file in Config/config_track.yaml with the paths required by your project.
```
python3 Bboxes_Tracker.py
```
* Generate_Features_Labels: Algorithm to generate the pseudo labels to finetune the detector on the target set, using the features extracted using SURF. It requires the prediction of the detector to finetune (YOLO in the case of this project), and to have installed OpenCV by compiling him to support also the non-free features.
Is necessary to modify the configuration file in Config/config_labels.yaml with the paths required by your project.
```
python3 Generate_Features_Labels.py
```
* Generate_SfM_Labels: Algorithm to generate the pseudo labels to finetune the detector on the target set. It requires the prediction of the detector to finetune (YOLO in the case of this project) and a 3-D model (txt model) of the environment generated by COLMAP, a SfM framework that given a sequence of frames returns the 3D reconstruction of the environment.
Is necessary to modify the configuration file in Config/config_sfm_labels.yaml with the paths required by your project.
```
python3 Generate_SfM_Labels.py
```




