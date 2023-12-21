# imports
import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt


# Linear regression
# f = w * x

# Define DATA
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

n_samples, n_features = X.shape
print(f'#samples: {n_samples}, #features: {n_features}')

test_sample = 342
X_test = torch.tensor([test_sample], dtype=torch.float32)

# 1) Design Model, the model has to implement the forward pass
input_size = n_features
output_size = n_features

# we can call this model with samples X
model = nn.Linear(input_size, output_size)

print(f'Prediction before training: f(5) = {model(X_test).item():.3f}')

# 2) Define loss and optimizer
learning_rate = 0.01
n_iters = 2000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
losses = []
for epoch in range(n_iters):
    # predict = forward pass with our model
    y_predicted = model(X)

    # loss
    l = loss(Y, y_predicted)

    # calculate gradients = backward pass
    l.backward()

    # update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()
    
    losses.append(l.item())
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, loss: {l.item()}")

print("-------------------------")
[w, b] = model.parameters() # unpack parameters
print(f"Parameters, Intercept: {b.item()} | Slope: {w.item()}")
print(f'Prediction after training: f({test_sample}) = {model(X_test).item():.3f}')

plt.plot(losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Loss vs Epochs")
plt.show()
