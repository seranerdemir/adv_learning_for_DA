import numpy as np
import torch

def BalanceCompute(labels):
    label_clone = labels.clone()
    target_label_balanced = label_clone.type(torch.FloatTensor)

    for batch_label, label_map in enumerate(labels):
        unique_classes = label_map.unique(sorted=True)
        class_element_count = torch.stack([(label_map == c).sum() for c in unique_classes])
        total_located_pixel = label_map.shape[0] * label_map.shape[1]
        for class_, pixel_count in zip(unique_classes, class_element_count):
            target_label_balanced[batch_label][target_label_balanced[batch_label] == float(class_.item())] = 1 - (
                    pixel_count.item() / total_located_pixel)
    return target_label_balanced
