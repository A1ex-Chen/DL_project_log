# Generates y_labels (ground truth class labels)

# This script must be run AFTER the raw images have been processed and saved as .bin

# To save as .bin, use a script like:
# https://github.com/SiliconLabs/platform_ml_models/blob/master/eembc/Person_detection/buildPersonDetectionDatabase.py
# but save as .bin instead of .jpg

import os

GROUND_TRUTH_FILENAME='y_labels.csv'
PERSON_DIR = 'vw_coco2014_96_bin/person'
NON_PERSON_DIR = 'vw_coco2014_96_bin/non_person'


generate_file_contents()