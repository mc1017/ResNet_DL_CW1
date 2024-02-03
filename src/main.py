import torch
import torch.optim as optim
from model.resnet import MyResNet

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


def train_part(model, optimizer, device, epochs=1):
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

            if t % print_every == 0:
                print("Epoch: %d, Iteration %d, loss = %.4f" % (e, t, loss.item()))
        check_accuracy(loader_val, model)


def get_device(USE_GPU=True):
    if USE_GPU and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print(device)


if __name__ == "__main__":
    # define and train the network
    model = MyResNet()
    optimizer = optim.Adamax(model.parameters(), lr=0.0001, weight_decay=1e-7)
    device = get_device()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters is: {}".format(params))

    train_part(model, optimizer, device, epochs=10)

    # report test set accuracy
    check_accuracy(loader_val, model, analysis=True)

    # save the model
    torch.save(model.state_dict(), "model.pt")
