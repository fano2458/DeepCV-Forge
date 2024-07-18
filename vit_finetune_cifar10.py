import torch
import torchvision




if __name__ == "__main__":

    dataset_train = torchvision.datasets.CIFAR10(root='/data', train=True, download=True)
    dataset_eval = torchvision.datasets.CIFAR10(root='/data', train=False, download=True)
