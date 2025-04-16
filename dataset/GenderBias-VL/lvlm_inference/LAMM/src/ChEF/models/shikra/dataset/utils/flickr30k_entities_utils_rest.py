import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

from tqdm import tqdm












PHRASE_ST_PLACEHOLDER = '<ph_st>'
PHRASE_ED_PLACEHOLDER = '<ph_ed>'




if __name__ == '__main__':
    filenames = [
        r'D:\home\dataset\flickr30kentities\train.txt',
        r'D:\home\dataset\flickr30kentities\val.txt',
        r'D:\home\dataset\flickr30kentities\test.txt',
    ]
    for filename in filenames:
        annotation_dir = r'D:\home\dataset\flickr30kentities'
        indexes = [line.strip() for line in open(filename, 'r', encoding='utf8')]
        flatten_annotation(annotation_dir, indexes)