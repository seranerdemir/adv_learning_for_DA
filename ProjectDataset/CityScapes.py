import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image


class CityScapesDataset(Dataset):
    def __init__(self, root_dir, target=True, train = False, transform=None, num_shot = 1):
        # images are png
        self.root_dir = root_dir
        self.num_shot = num_shot
        self.target = target
        self.train = train
        if self.target==True and self.train==False:
            self.images = [i_id.strip() for i_id in open('Utils/Shots/Images%sShot.txt' % self.num_shot)]
            self.labels = [i_id.strip() for i_id in open('Utils/Shots/Labels%sShot.txt' % self.num_shot)]
            self.image_dir = self.images
            self.label_dir = self.labels
        elif self.target==False and self.train==False:
            self.image_dir = os.path.join(self.root_dir, "ProjectDataset/CityScapesDataset/val/images")
            self.label_dir = os.path.join(self.root_dir, "ProjectDataset/CityScapesDataset/val/labels")
            self.images = os.listdir(self.image_dir)
            self.labels = os.listdir(self.label_dir)
        elif self.target==False and self.train==True:
            self.image_dir = os.path.join(self.root_dir, "ProjectDataset/CityScapesDataset/train/images")
            self.label_dir = os.path.join(self.root_dir, "ProjectDataset/CityScapesDataset/train/labels")
            self.images = os.listdir(self.image_dir)
            self.labels = os.listdir(self.label_dir)

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
        img_path = ""
        label_path = ""
        if self.target:
            img_path = self.image_dir[index]
            label_path = self.label_dir[index]
        else:
            img_path = os.path.join(self.image_dir, self.images[index])
            split_path = img_path.split(sep="_")
            split_img_root_path = split_path[0].split(sep="/")
            label_path = self.label_dir + '/' + split_img_root_path[-1] + '_' +split_path[1] + '_' + split_path[2] + '_' + 'gtFine_labelIds.png'

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

