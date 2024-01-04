import torch
from torchvision import transforms
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if __name__ == "__main__":
    """Processes corrupted MNIST data."""
    # file_path = "\\data\\raw\\"
    # target_path = "\\data\\processed\\"

    file_path = "C:\\Users\\tobia\\Documenter\\Master\\02476_MLOps\\dtu_mlops\\s2_organisation_and_version_control\\CookiecutterExercise\\data\\raw\\"
    target_path = "C:\\Users\\tobia\\Documenter\\Master\\02476_MLOps\\dtu_mlops\\s2_organisation_and_version_control\\CookiecutterExercise\\data\\processed\\"

    train_images = torch.Tensor()
    train_target = torch.Tensor()
    test_images = torch.Tensor()
    test_target = torch.Tensor()

    norm = transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1))

    for file in os.listdir(file_path):
        if file.startswith("train_images"):
            train_images = torch.cat((train_images, torch.load(file_path + file)), dim=0)
        elif file.startswith("train_target"):
            train_target = torch.cat((train_target, torch.load(file_path + file)), dim=0)
        elif file.startswith("test_images"):
            test_images = torch.cat((test_images, torch.load(file_path + file)), dim=0)
        elif file.startswith("test_target"):
            test_target = torch.cat((test_target, torch.load(file_path + file)), dim=0)

    # Normalize tensors
    train_images = (train_images - torch.mean(train_images)) / torch.std(train_images)
    test_images = (test_images - torch.mean(test_images)) / torch.std(test_images)

    torch.save(train_images, target_path + "train_images.pt")
    torch.save(train_target, target_path + "train_target.pt")
    torch.save(test_images, target_path + "test_images.pt")
    torch.save(test_target, target_path + "test_target.pt")
