import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim


def to_num(x):
    x = x.to(dtype=torch.int32)
    x = x.flip(dims=[-1])
    powers_of_two = 2 ** torch.arange(x.size(-1))
    return torch.sum(x * powers_of_two).item()


def to_list(n):
    binary_str = bin(n)[2:]
    binary_str = binary_str.zfill(32)
    binary_list = [float(bit) for bit in binary_str]

    return binary_list


def real_f(x):
    num = to_num(x)
    if num % 7 == 0 and num % 9 == 0:
        return 0.0
    elif num % 7 == 0 or num % 9 == 0:
        return 1.0
    else:
        return 0.0


num_sequences = 100000
X_list = [to_list(i) for i in range(num_sequences)]
X = torch.tensor(X_list)
Y = torch.tensor([real_f(x) for x in X]).view(-1, 1)
split_train = int(0.6 * num_sequences)
train_x = X[:split_train]
train_y = Y[:split_train]
split_test = int(0.2 * num_sequences)
test_x = X[split_train : split_train + split_test]
test_y = Y[split_train : split_train + split_test]

val_x = X[split_test + split_train :]
val_y = Y[split_test + split_train :]

for i in range(len(X)):
    num = to_num(X[i])
    if i != to_num(X[i]):
        print("Error at index", i)
        input()


class BinNet(nn.Module):
    def __init__(self, D_in=32, H=32, D_out=1):
        super(BinNet, self).__init__()
        self.linear1 = nn.Linear(D_in, H)
        self.linear2 = nn.Linear(H, H)
        self.linear3 = nn.Linear(H, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))

        return F.sigmoid(self.linear3(x))


def train(model, loss_fn, train_x, train_y, val_x, val_y):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    for i in range(75):
        output = model(train_x)
        loss = loss_fn(output, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val_output = model(val_x)
        loss_val = loss_fn(val_output, val_y)
        print(f"Iter {i+1}: Train Loss = {loss:.4f}, Val Loss = {loss_val:.4f}")


def test(model, loss_fn, test_x, test_y):
    model.eval()
    with torch.no_grad():
        out_test = model(test_x)
        loss_test = loss_fn(out_test, test_y)
        pred = out_test > 0.5
        acc = (pred == test_y).float().mean()

        print(f"acc: {acc}, Loss test = {loss_test:.4f}")


model = BinNet()
loss_fn = nn.SmoothL1Loss()

train(model, loss_fn, train_x, train_y, val_x, val_y)
test(model, loss_fn, test_x, test_y)
