import os
import matplotlib
matplotlib.use('Qt5Agg')
from skimage import io


import numpy as np
import torch
import torch.nn as nn
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


#torch.nn.Conv2d(1, 1, kernel_size, stride, padding)

import torch.nn as nn



input_size = (3, 1)
hidden_size = ()
output_size = ()


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


