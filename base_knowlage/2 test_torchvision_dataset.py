import torchvision

train_set = torchvision.datasets.CIFAR10(root="./Dataset/CIFAR10_dataset", train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./Dataset/CIFAR10_dataset", train=False, download=True)