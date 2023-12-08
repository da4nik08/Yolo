import os
import pandas as pd


def load_label_from_txt_to_df(path_txt):
    with open(path_txt, 'r') as file:
        lines = file.readlines()

    # Process and convert the data
    yolo_labels = []
    for line in lines:
        values = line.strip().split('\t')
        class_id, x, y, w, h = map(float, values)

        # Creating YOLO-formatted label
        yolo_labels.append([class_id, x, y, w, h])

    # Creating a DataFrame from the YOLO-formatted labels with appropriate column names
    return pd.DataFrame(yolo_labels, columns=[0, 1, 2, 3, 4])


def load_label_from_txt_to_str(path_txt):
    with open(path_txt, 'r') as file:
        lines = file.readlines()
    
    # Process and convert the data
    yolo_labels = []
    for line in lines:
        values = line.strip().split('\t')
        class_id, x, y, w, h = map(float, values)
    
        # Creating YOLO-formatted label
        yolo_label = f"{int(class_id)} {x} {y} {w} {h}"
        yolo_labels.append(yolo_label)

    return yolo_labels