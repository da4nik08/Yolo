import os
import numpy as np
import pandas as pd


def load_bbx(bbx_path):
    with open(bbx_path, mode='r') as file:
        lines = file.readlines()
        
    annotations = {}
    i = 0
    while i < len(lines):
        file_name = lines[i].strip()
        i += 1
        num_boxes = int(lines[i].strip())
        i += 1
        boxes = []
        for _ in range(num_boxes):
            box_info = lines[i].strip().split()
            box = {
                'x': int(box_info[0]),
                'y': int(box_info[1]),
                'w': int(box_info[2]),
                'h': int(box_info[3]),
            }
            boxes.append(box)
            i += 1
        annotations[file_name] = boxes

    return annotations