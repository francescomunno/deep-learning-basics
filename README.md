# Deep Learning Basics

Deep learning exercises with PyTorch, starting from MNIST and CIFAR-10.

## MNIST: fully connected network

`mnist_train.py` trains a small fully connected network to classify handwritten digits.

- **Model:** 784 → 128 (ReLU) → 10, 101,770 parameters
- **Data:** 50,000 training / 10,000 validation (split from the official training set) / 10,000 test
- **Training:** Adam, learning rate 0.001, batch size 64, cross-entropy loss
- **Early stopping:** stops after 3 epochs without improvement in validation loss and keeps the best model

| Set | Loss | Accuracy |
|---|---|---|
| Validation (best epoch: 11) | 0.0919 | 97.24% |
| Test | 0.0798 | 97.75% |

Single run, seed 42. Changing the seed moved validation accuracy after one epoch by about 0.3 points.

Without early stopping, training accuracy keeps rising to 99.9% while validation loss goes back up after epoch 11: the network starts to overfit.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python check_setup.py
python mnist_train.py
```
