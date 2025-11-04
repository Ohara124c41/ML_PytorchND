# Create Your Own Image Classifier - CIFAR-10 Image Classifier and Build-vs-Buy Decision

## Overview
This project trains a PyTorch image classifier on CIFAR-10 and evaluates accuracy to inform a build-vs-buy decision. The baseline convolutional model with basic augmentation achieves 77.8% top-1 test accuracy in 10 epochs, surpassing Detectocorp's 70% claim. State of the art models exceed 96% but are significantly larger and more expensive to train.

## Dataset
CIFAR-10 contains 60,000 color images of size 32x32 across 10 classes with 50,000 training and 10,000 test samples. The project uses torchvision.datasets.CIFAR10 with download=True. Normalization uses CIFAR-10 mean and standard deviation.

## Environment
Python 3.9 or later with PyTorch and torchvision. GPU is optional but recommended for training speed.
Install:
```
pip install torch torchvision matplotlib
```

## Data and Transforms
Training uses RandomHorizontalFlip, RandomCrop(32, padding=4), ToTensor, and Normalize. Testing uses ToTensor and Normalize only. A viewer loader with only ToTensor is used to visualize samples.

## Model
The model is a compact CNN with three convolutional blocks and max pooling, followed by two fully connected layers. The forward pass returns softmax probabilities over 10 classes. Loss is NLLLoss applied to log probabilities, and optimization uses Adam with lr=1e-3.

## Training
Training runs for 10 epochs with batch size 128 for train and 256 for test. Average training loss is recorded each epoch and plotted, and test accuracy is computed after each epoch.
After training, save weights to:
```
cifar10_cnn.pth
```

## Results
Final test accuracy is 77.8 percent, meeting the rubric threshold of 45 percent and exceeding Detectocorp's 70 percent. The loss curve decreases and the accuracy stabilizes between 76 and 78 percent by epoch 10.

## Recommendation
Build in-house. The minimal CNN, standard augmentation, and routine training already outperform the external 70 percent claim at low cost. If needed, transfer learning with a small pretrained backbone can improve accuracy further without incurring state of the art training budgets.

## Testing and Reuse
After training, evaluate with the test DataLoader and compute top-1 accuracy using argmax over probabilities. To reuse the model, instantiate the class and load weights with torch.load, then set model.eval().

## Rubric Compliance
Transforms include ToTensor and at least one augmentation, and DataLoaders are defined for train and test. Dataset size and tensor shapes are printed, sample images are displayed, and a Model class with softmax is implemented. A classification loss and optimizer are specified, average epoch loss is plotted, test accuracy is computed with the test DataLoader, a recommendation is provided, and weights are saved with torch.save.
"
