from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models.CifarResnet as resnet
import models.CifarVGG as vgg

def get_training_dataloader(datasetName,mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    if datasetName == 'cifar10':
        training_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    elif datasetName == 'cifar100':
        training_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    else:
        raise Exception('Unsupported dataset: choose from cifar10 or cifar100')

    training_loader = DataLoader(
        training_dataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,persistent_workers=True)

    return training_loader

def get_test_dataloader(datasetName,mean, std, batch_size=16, num_workers=2, shuffle=True):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    if datasetName == 'cifar10':
        testDataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    elif datasetName == 'cifar100':
        testDataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    else:
        raise Exception('Unsupported dataset: choose from cifar10 or cifar100')

    testLoader = DataLoader(testDataset, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size,persistent_workers=True)

    return testLoader

def load_config(config_path='config.json'):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_model(modelName,numClasses=10):
    if modelName == 'resnet32':
        model = resnet.resnet32(num_classes=numClasses)
    elif modelName == 'resnet56':
        model = resnet.resnet56(num_classes=numClasses)
    elif modelName == 'resnet110':
        model = resnet.resnet110(num_classes=numClasses)
    elif modelName == 'vgg16':
        model = vgg.vgg16_bn(num_classes=numClasses)
    else:
        raise NotImplementedError('Model name {} not supported'.format(modelName))
    return model

class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

