import torch
from models.model import Classifier
from torch import nn, optim


def prepare_dataloaders(batch_size=64):
    file_path = "C:\\Users\\tobia\\Documenter\\Master\\02476_MLOps\\dtu_mlops\\s2_organisation_and_version_control\\CookiecutterExercise\\data\\processed\\"

    train_images = torch.load(file_path + "train_images.pt")
    train_target = torch.load(file_path + "train_target.pt")
    test_images = torch.load(file_path + "test_images.pt")
    test_target = torch.load(file_path + "test_target.pt")

    trainset = torch.utils.data.TensorDataset(train_images, train_target)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size, shuffle=True)
    testset = torch.utils.data.TensorDataset(test_images, test_target)
    testloader = torch.utils.data.DataLoader(testset, batch_size, shuffle=True)

    return trainloader, testloader


if __name__ == "__main__":
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    train_set, _ = prepare_dataloaders()

    epochs = 30
    train_losses, test_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for images, labels in train_set:
            model.train
            optimizer.zero_grad()

            log_ps = model(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        if e % 5 == 0 & e != 0:
            torch.save(model.state_dict(), f"checkpoint_epoch{e}.pth")
