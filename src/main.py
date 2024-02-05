import torch
import torch.optim as optim
from model.resnet import MyResNet
from src.data_loader import get_transformed_data
import torch.nn.functional as F
import wandb
import ast
import os

dtype = torch.float32
rng_seed = 90
torch.manual_seed(rng_seed)


SWEEP_CONFIG = {
    "method": "random",
    "run_cap": 500,
    "name": "Early Stopping + Less Epoch + Adam",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "epochs": {"values": [10]},
        "batch": {"min": 16, "max": 256, "q": 8, "distribution": "q_log_uniform_values"},
        "lr": {"min": 1e-4, "max": 5e-2, "distribution": "log_uniform_values"},
        "momentum": {"min": 1e-2, "max": 1e0, "distribution": "log_uniform_values"},
        "decay": {"min": 1e-6, "max": 1e-2, "distribution": "log_uniform_values"},
        'optimizer': {
            'values': ['adam', 'adamw'] #'adamax', 'sgd'
        },
        'blocks': {
            'values': ['residual', 'bottleneck'] # 'wide', ,
        },
        'layers': {
            'values': ['(2, 2, 2, 2)',
                       '(3, 4, 6, 3)',
                       '(2, 2, 2, 2, 2, 2)']
        },
    },
}

def check_accuracy(loader, model, device, analysis=False):
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
    
def get_avg_validation_loss(model, loader_val, device):
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


def train_part(model, optimizer, device, loader_train, loader_val, epochs=1):
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
        avg_validation_loss = get_avg_validation_loss(model, loader_val, device)
        print(f"Epoch: {e}, Avg Train Loss: {avg_train_loss:.4f}, Avg Validation Loss: {avg_validation_loss:.4f}")
        acc = check_accuracy(loader_val, model, device)
        wandb.log({"accuracy": acc, "train_loss": avg_train_loss, "validation_loss": avg_validation_loss})
        if e == 2 and acc < 0.25:
            print(f"Stopping early at epoch {e} due to low accuracy.")
            return


def get_device(USE_GPU=True):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(device)
    return device

def build_optimiser(model, optimizer, learning_rate, decay=1e-7, momentum=0.9):
    if optimizer == "adamax":
        optimizer = optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=decay)
    elif optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    return optimizer


def train(config = None):
    with wandb.init(config=config):
        config = wandb.config
        
        # define and train the network
        layers = ast.literal_eval(config.layers)
        print("Layers Config",layers)
        model = MyResNet(config.blocks, layers)
        device = get_device()
        loader_train, loader_val, loader_test = get_transformed_data()

        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of parameters is: {}".format(params))
        optimizer = build_optimiser(model, config.optimizer, config.lr, config.decay, config.momentum)
        train_part(model, optimizer, device, loader_train, loader_val, epochs=config.epochs)

        # report test set accuracy
        check_accuracy(loader_test, model, device, analysis=True)

        # Save the model
        # Define the directory where you want to save your models
        model_save_dir = '/vol/bitbucket/mc620/DeepLearningCW1/models/'
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_path = os.path.join(model_save_dir, f"model_{wandb.run.id}.pt")
        torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    os.environ['WANDB_DIR'] = '/vol/bitbucket/mc620/DeepLearningCW1/' 
    # sweep_id = wandb.sweep(sweep=SWEEP_CONFIG, project="DL Coursework 1")
    sweep_id = "apph5f22"
    print("Sweep_id",sweep_id)
    wandb.agent(sweep_id, train, project="DL Coursework 1", count=300)
    
    
