import numpy as np


def transform_label(label, class_ind):
    output_list = list()
    for l in label:
        temp_list = [1.0]
        C, x, y, w, h = map(float, l.split(" "))
        #print(f"Class-{C}, x-{x}, y-{y}, w-{w}, h-{h}")
        temp_list = temp_list + [x, y, w, h] + ([0.0] * len(class_ind))
        temp_list[4 + class_ind.index(C) + 1] = 1.0
        output_list.append(temp_list)
    return output_list 


def Yolo_format_label(labels, n, anchor):
    yolo_labels = np.zeros((n, n, len(labels[0]) * anchor))

    # Assign the labels to the array based on the structure provided
    anchor_index = np.zeros((n, n))
    count = 0
    
    for i in range(len(labels)):
        label = labels[i]
        p, x, y, w, h, *class_probs = label
    
        # Calculate grid cell indices for x and y values
        cell_x = int(x * n)
        cell_y = int(y * n)

        if cell_x == n: cell_x = n-1
        if cell_y == n: cell_y = n-1
        
        # Fill in the values in the yolo_labels array
        if anchor_index[cell_x, cell_y] + len(label) > yolo_labels.shape[2]:
            #print(F"You need more anchor on cell {cell_x}, {cell_y}")
            #print(f"anchor_index[cell_x, cell_y] = {anchor_index[cell_x, cell_y]}")
            count += 1
        else:
            yolo_labels[cell_x, cell_y, int(anchor_index[cell_x, cell_y]):
                                        int(anchor_index[cell_x, cell_y]) + len(label)] = label

        # Calculate the anchor index based on object detectability in one cell
        anchor_index[cell_x, cell_y] += len(label)
    
    return yolo_labels, count