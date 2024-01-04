import click
import torch
from model import Classifier, train
from torch import nn, optim
from data import mnist
import torch.nn.functional as F


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = Classifier()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_set, _ = mnist()
    
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
            torch.save(model.state_dict(), f'checkpoint_epoch{e}.pth')

    
    


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = torch.load(model_checkpoint)
    _, test_set = mnist()


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
