import torch
import torch.optim as optim
from model.resnet import MyResNet
from src.data_loader import get_transformed_data
import torch.nn.functional as F
import wandb

dtype = torch.float32
rng_seed = 90
torch.manual_seed(rng_seed)


def check_accuracy(loader, model, analysis=False):
    # function for test accuracy on validation and test set

    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            x = x.to(device=device, dtype=dtype)  # move to device
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
            if t == 0 and analysis:
                stack_labels = y
                stack_predicts = preds
            elif analysis:
                stack_labels = torch.cat([stack_labels, y], 0)
                stack_predicts = torch.cat([stack_predicts, preds], 0)
        acc = float(num_correct) / num_samples
        print(
            "Got %d / %d correct of val set (%.2f)"
            % (num_correct, num_samples, 100 * acc)
        )
        if analysis:
            print("check acc", type(stack_predicts), type(stack_labels))
            # confusion(stack_predicts, stack_labels)
            # incorrect_preds(preds, y, x)
        return float(acc)
    
def get_avg_validation_loss():
    model.eval()
    total_val_loss = 0
    total_val_batches = 0
    with torch.no_grad():
        for x_val, y_val in loader_val:
            x_val = x_val.to(device=device, dtype=dtype)
            y_val = y_val.to(device=device, dtype=torch.long)
            val_scores = model(x_val)
            val_loss = F.cross_entropy(val_scores, y_val)

            total_val_loss += val_loss.item()
            total_val_batches += 1

    avg_val_loss = total_val_loss / total_val_batches
    return avg_val_loss


def train_part(model, optimizer, device, loader_train, epochs=1):
    """
    Train a model on NaturalImageNet using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        total_loss = 0
        total_batches = 0
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            loss.backward()

            # Update the parameters of the model using the gradients
            optimizer.step()

            if t % 10 == 0:
                print("Epoch: %d, Iteration %d, loss = %.4f" % (e, t, loss.item()))
            total_loss += loss.item()
            total_batches += 1

        avg_train_loss = total_loss / total_batches
        avg_validation_loss = get_avg_validation_loss()
        print(f"Epoch: {e}, Avg Train Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_validation_loss:.4f}")
        acc = check_accuracy(loader_val, model)
        wandb.log({"epoch": e, "accuracy": acc, "train_loss": avg_train_loss, "validation_loss": avg_validation_loss})


def get_device(USE_GPU=True):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(device)
    return device


if __name__ == "__main__":
    # Initialise Wandb 
    wandb.init(project="DL Coursework 1", name="Double-Depth-Model", config={
        "epochs": 30,
        "batch_size": 128,
        "lr": 1e-4,
    })
    config = wandb.config
    
    # define and train the network
    model = MyResNet()
    optimizer = optim.Adamax(model.parameters(), lr=config.lr, weight_decay=1e-7)
    device = get_device()
    loader_train, loader_val, loader_test = get_transformed_data()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))

    train_part(model, optimizer, device, loader_train, epochs=config.epochs)

    # report test set accuracy
    check_accuracy(loader_val, model, analysis=True)

    # save the model
    torch.save(model.state_dict(), "model.pt")
    wandb.save("model.pt")
    wandb.finish()
