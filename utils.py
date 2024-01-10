from matplotlib import pyplot as plt
import torch.nn as nn


def plot_curve(data, data_name):
    plt.plot(range(len(data)), data, color='blue')
    plt.legend([str(data_name)], loc='upper right')
    plt.xlabel('step')
    plt.ylabel(str(data_name))
    plt.show()


def init_nor(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight, gain=1)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight, gain=1)
    # elif type(m) == nn.ReLU:
    #     nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, mean=0, std=0.01)
    # elif type(m) == nn.Sigmoid:
    #     nn.init.xavier_uniform_(m.weight, gain=1)
