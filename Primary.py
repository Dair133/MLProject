

# Below is a very simple example of adjusting the weights and biases in a loop depending on the loss(difference between desired answer and calculated answer)

# We are basically saying below to the model "When input is x output must be y, so adjust your weights and biases so this is the case"
# first iteration we will not get y as the anwer when we input x, but model will attempt to fix itself by adjusting its weights and biases
# second iteration it will be sligtly close
# third iteration close
# and so on 
# eventually when we input x we will get y
# Eventually the weights and biases will be adjusted menaing that when we input x we will get y.

# This is basicallty like a very very watered down version of what happens in a real model, normally we give the model say a picture of a number between 1 and 10,
# translated to a matrix of numbers along with the answer(we tell the model what number the picture is of)
# The model runs the number  (represented by a matrix of numbers) through itself, compares how wrong it is and then adjusts itself accordingly
# eventually it canr ecognise drawings of any number between 1 and 10

import torch
import torch.optim as optim

# Initialize parameters
x = torch.ones(5)  # Input tensor
y = torch.zeros(3)  # Expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)

# Define optimizer
optimizer = optim.SGD([w, b], lr=0.01)  # Stochastic Gradient Descent

# Example training loop
num_epochs = 1000

for epoch in range(num_epochs):
    # Forward pass: compute predicted y
    z = torch.matmul(x, w) + b
    loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

    # Backward pass: compute gradient of the loss with respect to parameters
    loss.backward()

    # Print gradients
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
        print(f'Gradients of w: {w.grad}')
        print(f'Gradients of b: {b.grad}')

    # Update parameters
    optimizer.step()

    # Zero the gradients for the next iteration
    optimizer.zero_grad()

print("Training complete")
print(f"Final weights: {w}")
print(f"Final biases: {b}")
