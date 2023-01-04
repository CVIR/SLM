import torch
from torchvision import transforms, datasets
import params
from PIL import Image, ImageOps
import torch.utils.data as util_data
import datasets.data_list
from datasets.data_list import ImageList
import numpy as np
import numbers

class ResizeImage():
    def __init__(self, size):
      if isinstance(size, int):
        self.size = (int(size), int(size))
      else:
        self.size = size
    def __call__(self, img):
      th, tw = self.size
      return img.resize((th, tw))

class PlaceCrop(object):
    """Crops the given PIL.Image at the particular index.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (w, h), a square crop (size, size) is
            made.
    """

    def __init__(self, size, start_x, start_y):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.start_x = start_x
        self.start_y = start_y

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        th, tw = self.size
        return img.crop((self.start_x, self.start_y, self.start_x + tw, self.start_y + th))



def get_office_home_PDA(source_path, target_path, batch_size=128, shuffle=True, num_workers=4):

    data_transform_train = transforms.Compose([    
        ResizeImage(256),
        transforms.RandomResizedCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Imagenet values
    ])

    resize_size=256
    crop_size=224
    start_center = (resize_size - crop_size - 1) / 2

    data_transform_test = transforms.Compose([
        ResizeImage(resize_size),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])  # Imagenet values
    ])

    dsets_source = ImageList(open(source_path).readlines(), transform=data_transform_train)
    dsets_target = ImageList(open(target_path).readlines(), transform=data_transform_train)
    dsets_target_test = ImageList(open(target_path).readlines(), transform=data_transform_test)

    class_sample_count = [0.0] * 65

    for iLen in range(len(dsets_source)):
        class_sample_count[dsets_source[iLen][1]] += 1.0

    # print(class_sample_count)
    weights_per_class = float(sum(class_sample_count)) / torch.Tensor(class_sample_count)
    weights_per_class = weights_per_class.double()

    class_sample_weights = [0.0] * len(dsets_source)
    for iLen in range(len(dsets_source)):
    	class_sample_weights[iLen] = weights_per_class[dsets_source[iLen][1]]
    sampler = torch.utils.data.sampler.WeightedRandomSampler(class_sample_weights, len(dsets_source)) # Uniform sampler across classes.


    source_train_loader = util_data.DataLoader(dsets_source, batch_size=batch_size, shuffle=False, num_workers=4, sampler=sampler)
    target_train_loader = util_data.DataLoader(dsets_target, batch_size=batch_size, shuffle=True, num_workers=4)
    target_test_loader = util_data.DataLoader(dsets_target_test, batch_size=batch_size, shuffle=True, num_workers=4)

    print('Found {} Mini-batches for Office-Home (Source-Train) -- '.format(
        str(len(source_train_loader))))
    print('Found {} Mini-batches for Office-Home (Target-Train) -- '.format(
        str(len(target_train_loader))))
    print('Found {} Mini-batches for Office-Home (Target-Test) -- '.format(
        str(len(target_test_loader))))

    # inputs, classes = next(iter(source_train_loader))
    # print('Inputs shape: ' + str(inputs.shape))
    # print('Classes shape: ' + str(classes.shape) + ' --> ' + str(classes))

    return source_train_loader, target_train_loader, target_test_loader