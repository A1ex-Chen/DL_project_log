# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc-sa/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Download Flickr-Faces-HQ (FFHQ) dataset to current working directory."""

import os
import sys
import requests
import html
import hashlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import argparse
import itertools
import shutil
from collections import OrderedDict, defaultdict

PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True # avoid "Decompressed Data Too Large" error

#----------------------------------------------------------------------------

json_spec = dict(file_url='https://drive.google.com/uc?id=16N0RV4fHI6joBuKbQAoG34V_cQk7vxSA', file_path='ffhq-dataset-v2.json', file_size=267793842, file_md5='425ae20f06a4da1d4dc0f46d40ba5fd6')

tfrecords_specs = [
    dict(file_url='https://drive.google.com/uc?id=1LnhoytWihRRJ7CfhLQ76F8YxwxRDlZN3', file_path='tfrecords/ffhq/ffhq-r02.tfrecords', file_size=6860000,      file_md5='63e062160f1ef9079d4f51206a95ba39'),
    dict(file_url='https://drive.google.com/uc?id=1LWeKZGZ_x2rNlTenqsaTk8s7Cpadzjbh', file_path='tfrecords/ffhq/ffhq-r03.tfrecords', file_size=17290000,     file_md5='54fb32a11ebaf1b86807cc0446dd4ec5'),
    dict(file_url='https://drive.google.com/uc?id=1Lr7Tiufr1Za85HQ18yg3XnJXstiI2BAC', file_path='tfrecords/ffhq/ffhq-r04.tfrecords', file_size=57610000,     file_md5='7164cc5531f6828bf9c578bdc3320e49'),
    dict(file_url='https://drive.google.com/uc?id=1LnyiayZ-XJFtatxGFgYePcs9bdxuIJO_', file_path='tfrecords/ffhq/ffhq-r05.tfrecords', file_size=218890000,    file_md5='050cc7e5fd07a1508eaa2558dafbd9ed'),
    dict(file_url='https://drive.google.com/uc?id=1Lt6UP201zHnpH8zLNcKyCIkbC-aMb5V_', file_path='tfrecords/ffhq/ffhq-r06.tfrecords', file_size=864010000,    file_md5='90bedc9cc07007cd66615b2b1255aab8'),
    dict(file_url='https://drive.google.com/uc?id=1LwOP25fJ4xN56YpNCKJZM-3mSMauTxeb', file_path='tfrecords/ffhq/ffhq-r07.tfrecords', file_size=3444980000,   file_md5='bff839e0dda771732495541b1aff7047'),
    dict(file_url='https://drive.google.com/uc?id=1LxxgVBHWgyN8jzf8bQssgVOrTLE8Gv2v', file_path='tfrecords/ffhq/ffhq-r08.tfrecords', file_size=13766900000,  file_md5='74de4f07dc7bfb07c0ad4471fdac5e67'),
    dict(file_url='https://drive.google.com/uc?id=1M-ulhD5h-J7sqSy5Y1njUY_80LPcrv3V', file_path='tfrecords/ffhq/ffhq-r09.tfrecords', file_size=55054580000,  file_md5='05355aa457a4bd72709f74a81841b46d'),
    dict(file_url='https://drive.google.com/uc?id=1M11BIdIpFCiapUqV658biPlaXsTRvYfM', file_path='tfrecords/ffhq/ffhq-r10.tfrecords', file_size=220205650000, file_md5='bf43cab9609ab2a27892fb6c2415c11b'),
]

license_specs = {
    'json':      dict(file_url='https://drive.google.com/uc?id=1SHafCugkpMZzYhbgOz0zCuYiy-hb9lYX', file_path='LICENSE.txt',                    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'images':    dict(file_url='https://drive.google.com/uc?id=1sP2qz8TzLkzG2gjwAa4chtdB31THska4', file_path='images1024x1024/LICENSE.txt',    file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'thumbs':    dict(file_url='https://drive.google.com/uc?id=1iaL1S381LS10VVtqu-b2WfF9TiY75Kmj', file_path='thumbnails128x128/LICENSE.txt',  file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'wilds':     dict(file_url='https://drive.google.com/uc?id=1rsfFOEQvkd6_Z547qhpq5LhDl2McJEzw', file_path='in-the-wild-images/LICENSE.txt', file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
    'tfrecords': dict(file_url='https://drive.google.com/uc?id=1SYUmqKdLoTYq-kqsnPsniLScMhspvl5v', file_path='tfrecords/ffhq/LICENSE.txt',     file_size=1610, file_md5='724f3831aaecd61a84fe98500079abc2'),
}

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------


#----------------------------------------------------------------------------


#----------------------------------------------------------------------------



#----------------------------------------------------------------------------


#----------------------------------------------------------------------------


#----------------------------------------------------------------------------


#----------------------------------------------------------------------------


#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_cmdline(sys.argv)

#----------------------------------------------------------------------------