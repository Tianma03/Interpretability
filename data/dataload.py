"""
github:https://github.com/yewsiang/ConceptBottleneck.git modify
General utils for training, evaluation and data loading
"""
import torch
import pickle
import numpy as np
import torchvision.transforms as transforms

from pathlib import Path
from PIL import Image
from torch.utils.data import BatchSampler
from torch.utils.data import Dataset, DataLoader


class CUBDataset(Dataset):
    """
    Returns a compatible Torch Dataset object customized for the CUB dataset
    """

    def __init__(self, pkl_file_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform=None):
        """
        Arguments:
        pkl_file_paths: list of full path to all the pkl data
        use_attr: whether to load the attributes (e.g. False for simple finetune)
        no_img: whether to load the images (e.g. False for A -> Y model)
        uncertain_label: if True, use 'uncertain_attribute_label' field (i.e. label weighted by uncertainty score, e.g. 1 & 3(probably) -> 0.75)
        image_dir: root directory containing the CUB image class folders
        n_class_attr: number of classes to predict for each attribute. If 3, then make a separate class for not visible
        transform: whether to apply any special transformation. Default = None, i.e. use standard ImageNet preprocessing
        """
        pkl_file_paths = [str(path) for path in pkl_file_paths]
        self.data = []
        self.is_train = any(["train" in path for path in pkl_file_paths])
        if not self.is_train:
            assert any([("test" in path) or ("val" in path) for path in pkl_file_paths])
        for file_path in pkl_file_paths:
            with open(file_path, 'rb') as file:
                self.data.extend(pickle.load(file))
        self.transform = transform
        self.use_attr = use_attr
        self.no_img = no_img
        self.uncertain_label = uncertain_label
        self.image_dir = Path(image_dir).expanduser()
        if image_dir == 'images' and not self.image_dir.is_dir():
            self.image_dir = Path('CUB_200_2011/images')
        self.n_class_attr = n_class_attr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        stored_path = Path(img_data['img_path'])

        # Rebase the authors' absolute paths onto the local image directory.
        if 'images' in stored_path.parts:
            images_index = stored_path.parts.index('images')
            relative_path = Path(*stored_path.parts[images_index + 1:])
        elif stored_path.is_absolute():
            relative_path = Path(*stored_path.parts[-2:])
        else:
            relative_path = stored_path

        img_path = self.image_dir / relative_path

        with Image.open(img_path) as image:
            img = image.convert('RGB')

        class_label = img_data['class_label']
        if self.transform:
            img = self.transform(img)

        if self.use_attr:
            if self.uncertain_label:
                attr_label = img_data['uncertain_attribute_label']
            else:
                attr_label = img_data['attribute_label']
            if self.no_img:
                if self.n_class_attr == 3:
                    one_hot_attr_label = np.zeros((len(attr_label), self.n_class_attr))
                    one_hot_attr_label[np.arange(len(attr_label)), attr_label] = 1
                    return one_hot_attr_label, class_label
                else:
                    return attr_label, class_label
            else:
                return img, class_label, attr_label
        else:
            return img, class_label


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """

    def __init__(self, dataset, indices=None):
        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices)

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):  # Note: for single attribute dataset
        return dataset.data[idx]['attribute_label'][0]

    def __iter__(self):
        idx = (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))
        return idx

    def __len__(self):
        return self.num_samples

def load_data(
    pkl_paths,
    use_attr,
    no_img,
    batch_size,
    uncertain_label=False,
    n_class_attr=2,
    image_dir='images',
    resampling=False,
    resol=299,
    normalization='upstream',
    num_workers=0,
    pin_memory=False,
    worker_init_fn=None,
):
    """
    Load CUB with upstream augmentations and configurable input preprocessing.
    Defaults preserve the original loader; ResNet training passes resol=224 and
    normalization='imagenet'. Resampling still uses only the first attribute.
    """
    if normalization == 'upstream':
        mean, std = [0.5, 0.5, 0.5], [2, 2, 2]
    elif normalization == 'imagenet':
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    else:
        raise ValueError("normalization must be 'upstream' or 'imagenet'")

    is_training = any(['train.pkl' in str(f) for f in pkl_paths])
    if is_training:
        transform = transforms.Compose([
            transforms.ColorJitter(brightness=32/255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(resol),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])
    else:
        transform = transforms.Compose([
            # Keep the upstream direct center crop without a preceding resize.
            transforms.CenterCrop(resol),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            ])

    dataset = CUBDataset(pkl_paths, use_attr, no_img, uncertain_label, image_dir, n_class_attr, transform)
    if is_training:
        drop_last = True
        shuffle = True
    else:
        drop_last = False
        shuffle = False
    loader_options = {
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'worker_init_fn': worker_init_fn,
    }
    if resampling:
        sampler = BatchSampler(ImbalancedDatasetSampler(dataset), batch_size=batch_size, drop_last=drop_last)
        loader = DataLoader(dataset, batch_sampler=sampler, **loader_options)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **loader_options)
    return loader

def find_class_imbalance(pkl_file, multiple_attr=False, attr_idx=-1):
    """
    Calculate class imbalance ratio for binary attribute labels stored in pkl_file
    If attr_idx >= 0, then only return ratio for the corresponding attribute id
    If multiple_attr is True, then return imbalance ratio separately for each attribute. Else, calculate the overall imbalance across all attributes
    """
    imbalance_ratio = []
    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    n = len(data)
    n_attr = len(data[0]['attribute_label'])
    if attr_idx >= 0:
        n_attr = 1
    if multiple_attr:
        n_ones = [0] * n_attr
        total = [n] * n_attr
    else:
        n_ones = [0]
        total = [n * n_attr]
    for d in data:
        labels = d['attribute_label']
        if multiple_attr:
            for i in range(n_attr):
                n_ones[i] += labels[i]
        else:
            if attr_idx >= 0:
                n_ones[0] += labels[attr_idx]
            else:
                n_ones[0] += sum(labels)
    for j in range(len(n_ones)):
        if n_ones[j] == 0:
            raise ValueError("Cannot compute imbalance weights for concepts with no positive labels")
        imbalance_ratio.append(float(total[j]/n_ones[j] - 1))
    if not multiple_attr: #e.g. [9.0] --> [9.0] * 312
        imbalance_ratio *= n_attr
    return imbalance_ratio
