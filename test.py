import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from train import NeuralNetwork

training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor()
)

batch_size = 128

train_dataloader = DataLoader(training_data, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)

model = NeuralNetwork()
model.load_state_dict(torch.load("data/fashion_mnist.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

figure = plt.figure(figsize=(6, 6))
cols, rows, i = 3, 3, 1

model.eval()
for data in range(20):
    x, y = test_data[data][0], test_data[data][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
        if predicted != actual:
            figure.add_subplot(3, 3, i)
            i += 1
            plt.title(f'{predicted}:{actual}')
            plt.axis('off')
            plt.imshow(x.squeeze(), cmap='gray')
plt.show()