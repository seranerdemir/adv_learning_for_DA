# The dataset loader
import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image



class Gta5Dataset(Dataset):
    def __init__(self, root_dir, transform=None, image_list=None):
        # images are png
        self.root_dir = root_dir
        self.image_list = image_list
        self.image_dir = os.path.join(self.root_dir, "ProjectDataset/GTADataset/images")
        self.label_dir = os.path.join(self.root_dir, "ProjectDataset/GTADataset/labels")

        if image_list == None:
            self.image_list = os.listdir(self.image_dir)
            self.images = [os.path.join(self.image_dir, name) for name in self.image_list]
            self.labels = [os.path.join(self.label_dir, name) for name in self.image_list]
        else:
            self.images = self.image_list
            self.labels = [path.replace("images", "labels") for path in self.images]

        self.transform = transform

        self.scale = (1024, 512)
        self.classes = [[0, 255], [1, 255], [2, 255], [3, 255], [4, 255], [5, 255], [6, 255], [7, 0], [8, 1], [9, 255], [10, 255],
                        [11, 2], [12, 3], [13, 4], [14, 255], [15, 255], [16, 255], [17, 5], [18, 255], [19, 6], [20, 7], [21, 8],
                        [22, 9], [23, 10], [24, 11], [25, 12],  [26, 13], [27, 14],  [28, 15], [29, 255], [30, 255],  [31, 16],
                        [32, 17], [33, 18],  [-1, 255]]

    def __len__(self):
        return len(self.images)

    def remap_label(self, label):
        dont_care = 255
        label_copy = np.ones(label.shape, dtype=np.float32) * dont_care
        for i, j in self.classes:
            label_copy[label == i] = j

        return label_copy

    def __getitem__(self, index):
        label_path = None
        img_path = None

        img_path = self.images[index]
        label_path = self.labels[index]

        image = Image.open(img_path)
        image = image.convert("RGB")
        label = Image.open(label_path)

        #resize image, label
        image = image.resize(self.scale)
        label = label.resize(self.scale)
        #convert array
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        #remap label
        label = self.remap_label(label)

        size = image.shape
        image = image[:, :, ::-1]

        # Convert HWC to CHW
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), str(self.images[index])


