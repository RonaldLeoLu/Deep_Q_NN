# Brain DQN
import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable

class DeepQNN(nn.Module):
    '''
    The architecture of deep Q neural network

    While training, we use minibatch, so the input has its own batch size.
    While playing, each time, there just need one sample so no batch_size
    or batch_size = 1.

    Input dim    : [batch_size, channels, height, width]
    Kernel dim   : [filter_height, filter_width, in_channels, out_channels]
    Conv out dim : [batch_size, channels, height, width]
    Flatten      : [batch_size, length]

    Input Layer : [32, 4, 80, 80]
    Conv Layer1 :
        - kernel size : [8, 8, 4, 32]
        - padding : 2
        - stride : 4
        - output : [32, 32, 20, 20]
    MaxPooling Layer1 :
        - kernel size : [2, 2]
        - output : [32, 32, 10, 10]
    Conv Layer2 :
        - kernel size : [4, 4, 32, 64]
        - padding : 1
        - stride : 2
        - output : [32, 64, 5, 5]
    Conv Layer3 :
        - kernel size : [3, 3, 64, 64]
        - padding : 1
        - stride : 1
        - output : [32, 64, 5, 5]
    Flatten : [32, 1600]  # torch.flatten(x, start_dim=1)
    Linear Layer1 :
        - activation : ReLu
        - output : [32, 512]
    Linear Layer2 : 
        - output : [32, 2]
    '''

    def __init__(self):
        super(DeepQNN, self).__init__()
        # build basic layers
        self.Conv1 = nn.Conv2d(4, 32, 8, stride=4, padding=2)
        self.MaxPooling = nn.MaxPool2d(2)
        self.Conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.Conv3 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.Linear1 = nn.Linear(1600, 512)
        self.Linear2 = nn.Linear(512, 2)
        # init weights of layers
        self.set_init_weights()

    def forward(self, frame):
        if not isinstance(frame, Variable):
            if isinstance(frame, np.ndarray):
                frame = Variable(torch.Tensor(frame))
            else:
                raise ValueError(
                    'Only accept \'torch.variable\' or \'numpy.ndarray\'')

        if len(frame.shape) == 3:
            frame = frame.reshape((1,)+frame.shape)

        h0 = nn.functional.relu(self.Conv1(frame))
        h0 = self.MaxPooling(h0)
        h1 = nn.functional.relu(self.Conv2(h0))
        h2 = nn.functional.relu(self.Conv3(h1))

        fc = torch.flatten(h2, start_dim=1)
        fc1 = nn.functional.relu(self.Linear1(fc))
        fc2 = self.Linear2(fc1)

        return fc2

    def set_init_weights(self):
        # init conv layers
        nn.init.kaiming_normal_(self.Conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.Conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.Conv3.weight, mode='fan_out', nonlinearity='relu')
        # init full connection layers
        nn.init.normal_(self.Linear1.weight)
        nn.init.normal_(self.Linear2.weight)


    def get_action(self, epsilon, frame):
        if random.random() <= epsilon:
            return self.get_random_action()
        return self.get_optim_action(frame)[0]

    def get_random_action(self):
        action = [0, 0]
        idx = 0 if random.random() < 0.8 else 1
        action[idx] = 1
        return action


    def get_optim_action(self, frame):
        q = self.forward(frame).data.numpy()
        #print('q shape',q.shape)
        indices = np.argmax(q, axis=1)
        actions = np.zeros_like(q)
        for i in range(q.shape[0]):
            actions[i][indices[i]] = 1
        return actions


if __name__ == '__main__':
    a = np.random.randn(4, 80, 80)
    b = True
    li = [(a, b)] * 32
    fp = random.sample(li, 5)
    aka = [data[0] for data in fp]
    wp = Variable(torch.Tensor(np.array(aka)))
    model = DeepQNN()
    q = model.forward(wp)
    print(q)

