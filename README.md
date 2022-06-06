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
* Generate_Features_Labels: Algorithm to generate the pseudo labels to finetune the detector on the target set. It requires the prediction of the detector to finetune (YOLO in the case of this project). Need to modify the configuration file in Config/config_labels.yaml with the paths required by your project.
```
python3 Generate_Features_Labels.py
```




