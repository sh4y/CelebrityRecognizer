import os
import matplotlib
matplotlib.use('Qt5Agg')
from skimage import io


import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.Dataset as Dataset

import torchvision
import torchvision.transforms as transforms


#seed = 42
#np.random.seed(seed)
#torch.manual_seed(seed)

actors = ('scarlett_johansson', 'robert_downey', 'chris_evans')

kernel_size = 5
stride = 2
padding = 2


features, targets = [], []


def gather_data(actor):
    folder = "./Actor_Database/" + actor
    for file in os.listdir(folder):
        if ".png" in file or ".jpg" in file:
            print('file_name: ' + file)
            # image = io.imread(folder + "/"+ file)
            # io.imshow(image)
            # io.show()

            features.append(io.imread(folder + "/" + file))

            targets.append(actor)


for actor in actors:
    gather_data(actor)

print("Gather " + str(len(features)) + " amount of features.")


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample


class MyCustomDataset(Dataset):
    def __init__(self, data_features, data_targets, transforms=None):
        # stuff
        self.features = data_features
        self.targets = data_targets
        self.transforms = transforms  # Use rescale transformation maybe?

    def __getitem__(self, index):
        # stuff
        image = self.features[index]
        label = self.targets[index]
        #
        # if self.transforms is not None:
        #     data = self.transforms(data)
        # If the transform variable is not empty
        # then it applies the operations in the transforms with the order that it is created.

        return np.ndarray(image, label)

    def __len__(self):
        return len(self.features)  # of how many data(images?) you have


custom_dataset = MyCustomDataset(features, targets)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


n_hidden = 10
rnn = RNN(3, n_hidden, 3)




