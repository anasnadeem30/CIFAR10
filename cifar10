#NADEEM, Anas


# Imports
import torch, torchvision
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

# Please modify this line according to your device. We trained this model on Google Collab where CUDA was available.
hardware = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_type = torch.float16  # Using float16 to speed up training and reduce memory usage

# Training configuration
NUM_EPOCHS = 24
SIZE_BATCH = 512
VELOCITY = 0.9
DECAY_RATE = 5e-4 * SIZE_BATCH
schedule_points = [0, NUM_EPOCHS / 5, NUM_EPOCHS]  # Epoch milestones for adjusting learning rate
schedule_values = [0.1 / SIZE_BATCH, 0.6 / SIZE_BATCH, 0]
IMG_WIDTH, IMG_HEIGHT = 32, 32
MASK_SIZE = 8
SHIFT_SIZE = 4

# Convolutional neural network architecture
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        layers = [3, 64, 128, 128, 128, 256, 512, 512, 512]
        modules = []
        for j in range(len(layers) - 1):
            channels_in, channels_out = layers[j], layers[j + 1]
            modules.append(nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=3, padding=1, stride=1, bias=False))
            if channels_out == channels_in * 2:
                modules.append(nn.MaxPool2d(kernel_size=2))
            modules.append(nn.BatchNorm2d(channels_out))
            modules.append(nn.CELU(alpha=0.075))
        self.network = nn.Sequential(*modules, nn.MaxPool2d(kernel_size=4), nn.Flatten(), nn.Linear(layers[-1], 10, bias=False))

    def forward(self, input, target):
        input = self.network(input) / 8
        return F.cross_entropy(input, target, reduction='none', label_smoothing=0.2), (input.argmax(dim=1) == target) * 100

# Load and preprocess CIFAR-10 dataset
def fetchCIFAR10(hardware):
    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
    test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    collections = [torch.tensor(data, device=hardware) for data in (train_set.data, train_set.targets, test_set.data, test_set.targets)]
    std_dev, avg = torch.std_mean(collections[0].float(), dim=(0, 1, 2), unbiased=True, keepdim=True)
    for i in [0, 2]:
        collections[i] = ((collections[i] - avg) / std_dev).to(data_type).permute(0, 3, 1, 2)
    return collections

# Shuffle and augment data during training
def partitionData(data_X, data_y, is_training):
    if is_training:
        permutation = torch.randperm(len(data_X), device=hardware)
        data_X, data_y = data_X[permutation], data_y[permutation]
        Crop = ([ (y0, x0) for x0 in range(SHIFT_SIZE + 1) for y0 in range(SHIFT_SIZE + 1)],
                lambda img, y0, x0: nn.ReflectionPad2d(SHIFT_SIZE)(img.float())[..., y0:y0 + IMG_HEIGHT, x0:x0 + IMG_WIDTH].to(data_type))
        Flip = ([ (True,), (False,)],
                lambda img, is_flip: torch.flip(img, [-1]) if is_flip else img)
        for options, transform in (Crop, Flip):
            option_indices = torch.randint(len(options), (len(data_X),), device=hardware)
            for i in range(len(options)):
                data_X[option_indices == i] = transform(data_X[option_indices == i], *options[i])
    return ((data_X[i:i + SIZE_BATCH], data_y[i:i + SIZE_BATCH]) for i in range(0, len(data_X) - is_training * (len(data_X) % SIZE_BATCH), SIZE_BATCH))

data_X_train, data_y_train, data_X_test, data_y_test = fetchCIFAR10(hardware)

# Initialize the network and optimizer
network_model = ConvNet().to(data_type).to(hardware)
optimizer = torch.optim.SGD(network_model.parameters(), lr=0, momentum=VELOCITY, weight_decay=DECAY_RATE, nesterov=True)

# Training loop
time_training = step_index = 0
final_test_accuracy = 0.0
for epoch_index in range(NUM_EPOCHS):
    time_start = time.perf_counter()
    list_losses, list_accuracies = [], []
    network_model.train()
    for X, y in partitionData(data_X_train, data_y_train, True):
        step_index += 1
        optimizer.param_groups[0]['lr'] = np.interp([step_index / (len(data_X_train) // SIZE_BATCH)], schedule_points, schedule_values)[0]
        loss_value, accuracy_value = network_model(X, y)
        network_model.zero_grad()
        loss_value.sum().backward()
        optimizer.step()
        list_losses.append(loss_value.detach())
        list_accuracies.append(accuracy_value.detach())
    time_training += time.perf_counter() - time_start

    network_model.eval()
    with torch.no_grad():
        list_test_accuracies = [network_model(X, y)[1].detach() for X, y in partitionData(data_X_test, data_y_test, False)]
        epoch_test_accuracy = torch.mean(torch.cat(list_test_accuracies).float())
        final_test_accuracy = epoch_test_accuracy

    compute_average = lambda items: torch.mean(torch.cat(items).float())
    print(f'Epoch {epoch_index + 1:02d}: Training Loss: {compute_average(list_losses):.3f}, Training Accuracy: {compute_average(list_accuracies):.2f}, Test Accuracy: {epoch_test_accuracy:.2f}, Training Time: {time_training:.2f}')

print(f'Training completed. Final model accuracy: {final_test_accuracy:.2f}%')
