import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_boxes(img, df):
    if len(img.shape) == 3:
        h, w, _ = img.shape
    else:
        h, w = img.shape
    fig, ax = plt.subplots()
    for index, row in df.iterrows():
        patch = Rectangle(
            ((row[1] - row[3] / 2.0) * w, (row[2] - row[4] / 2.0) * h),
            row[3] * w,
            row[4] * h,
            edgecolor = 'red',
            fill=False,
        )
        ax.add_patch(patch)
    
    plt.imshow(img)