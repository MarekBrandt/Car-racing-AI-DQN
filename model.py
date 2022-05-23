import torch.nn as nn
import torch
import os


class DQN(nn.Module):

    def __init__(self, width, height, outputs, action_skip):
        super(DQN, self).__init__()

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convW = conv2d_size_out(width, 8)
        convW = conv2d_size_out(convW, 5)
        convW = conv2d_size_out(convW, 3, 1)

        convH = conv2d_size_out(height, 8)
        convH = conv2d_size_out(convH, 5)
        convH = conv2d_size_out(convH, 3, 1)
        linear_input_size = convW * convH * 64

        self.selectTrackFeatures = nn.Sequential(
            nn.Conv2d(action_skip, 32, kernel_size=8, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(linear_input_size, 100),
            nn.Linear(100, outputs)
        )

    def forward(self, state):
        x = self.selectTrackFeatures(state)
        x = x.view(x.size(0), -1)
        # print("actual size of input ",x.size())
        return self.fc1(x)

    def save(self, file_name='model.pth'):
        model_folder_path = 'model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        path = os.path.join('model', file_name)
        self.load_state_dict(torch.load(path))
