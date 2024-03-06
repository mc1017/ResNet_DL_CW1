import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# DATA_PATH = "/vol/bitbucket/mc620/DeepLearningCW1/dataset/"
DATA_PATH = "/home/marco/Documents/Deep-Learning-CW1/dataset/"


def get_transformed_data():
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean.tolist(), std.tolist()),
        ]
    )
    train_path = DATA_PATH + "NaturalImageNetTrain"
    test_path = DATA_PATH + "NaturalImageNetTest"

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    # Create train val split
    n = len(train_dataset)
    n_val = int(n / 10)

    train_set, val_set = torch.utils.data.random_split(
        train_dataset, [n - n_val, n_val]
    )

    print(
        "Dataset Length: \n Train: {}, Validation: {}, Test: {}".format(
            len(train_set), len(val_set), len(test_dataset)
        )
    )
    batch_size = 128

    loader_train = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    loader_val = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
    loader_test = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    return loader_train, loader_val, loader_test
