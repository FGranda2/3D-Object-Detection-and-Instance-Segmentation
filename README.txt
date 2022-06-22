AER 1515 - Perception For Robotics
Assignment # 3 - 3D Object Detection and Instance Segmentation

NAME: FRANCISCO GRANDA
STUDENT NUMBER: 1006655941

INCLUDED PYTHON FILES:
'part1_estimate_depth.py' - Script used to generate the estimated depth images.
'part2_yolo.py'           - Script used to generate image with detected bounding boxes.
'part3_segmentation.py'   - Script used to generate the segmentation mask of each image.
'kitti_dataHandler.py'    - Provided script with various functions.

REQUIRED DEPENDENCIES:
As provided in the starter codes, the included files make use of OpenCV, NumPY, and Matplotlib.
Adittionally, the script 'part3_segmentation.py' requires the installation of the Skimage Python package.

REQUIRED FOLDERS IN DIRECTORY FOR INPUT AND OUTPUT:
In addition to the provided folders in the data, the following folders are required inside 'data/test/' base directory:
"est_depths"       - Store estimated depth maps.
"output_box"       - Store information of bounding boxes.
"output_images"    - Store images with overlayed bounding boxes.
"est_segmentation" - Store segmentation masks.

