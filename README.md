# About
Python package for semantic segmentation
Models implemented:
-DeepLabV3+
-ResUnet
-ResUnet++
-Unet
-DoubleUnet
-TriUnet

# Requirements
Before launching **any** script, set up virtual environment.

To install the package using pip use commnad:
pip install -e PACKAGE_PATH


# How to use
To train the model use DeepEndo/Scripts/Train.py --config PATH_TO_CONFIG
Config template can be found in DeepEndo/configs/sample_config.yaml

input data must be .csv file with two columns:
-abs_path_images containing absolute paths to images
-abs_path_masks containing absolute paths to segmentation masks
