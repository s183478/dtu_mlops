import torch
from torchvision import datasets, transforms
import os


def mnist():
    """Return train and test dataloaders for MNIST."""
    file_path = "../../../data/corruptmnist/"
    
    train_images = torch.Tensor()
    train_target = torch.Tensor()
    test_images = torch.Tensor()
    test_target = torch.Tensor()
    
    for file in os.listdir(file_path):
        if file.startswith("train_images"):
            train_images = torch.cat((train_images, torch.load(file_path + file)), dim=0)
        elif file.startswith("train_target"):
            train_target = torch.cat((train_target, torch.load(file_path + file)), dim=0)
        elif file.startswith("test_images"):
            test_images = torch.cat((test_images, torch.load(file_path + file)), dim=0)
        elif file.startswith("test_target"):
            test_target = torch.cat((test_target, torch.load(file_path + file)), dim=0)
            
    trainset = torch.utils.data.TensorDataset(train_images, train_target)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torch.utils.data.TensorDataset(test_images, test_target)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

    return trainloader, testloader
def test_cmnist():
    file_path = "../../../data/corruptmnist/"
    
    train_images = torch.Tensor()
    train_target = torch.Tensor()
    test_images = torch.Tensor()
    test_target = torch.Tensor()
    
    # Define a transform to normalize the data
    norm = transforms.Normalize(mean=(0,0,0), std=(1,1,1))
    
    
    for file in os.listdir(file_path):
        if file.startswith("train_images"):
            print("Found train images")
            train_images = torch.cat((train_images, torch.load(file_path + file)), dim=0)
        elif file.startswith("train_target"):
            print("Found train target")
            train_target = torch.cat((train_target, torch.load(file_path + file)), dim=0)
        elif file.startswith("test_images"):
            print("Found test images")
            test_images = torch.cat((test_images, torch.load(file_path + file)), dim=0)
        elif file.startswith("test_target"):
            print("Found test target")
            test_target = torch.cat((test_target, torch.load(file_path + file)), dim=0)
            
    print(train_images.shape)
    trainset = torch.utils.data.TensorDataset(norm(train_images), train_target)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torch.utils.data.TensorDataset(norm(test_images), test_target)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    test_tensor = torch.load(file_path + "test_target.pt")
    return test_tensor, trainloader, testloader
