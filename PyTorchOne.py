import os
import torch
from torch import optim, nn
import matplotlib.pyplot as plt
import numpy as np




# Function to visualize images
def show_images(images, num_images=5):
    images = images.view(images.size(0), 28, 28)  # Reshape images to 28x28
    images = images.detach().numpy()  # Convert to numpy array
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    plt.show()

# Define the generator and discriminator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()  # Assuming output images are 28x28 pixels
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)



# Instances of Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

if os.path.exists('generator_state.pth'):
    generator.load_state_dict(torch.load('generator_state.pth'))
if os.path.exists('discriminator_state.pth'):
    discriminator.load_state_dict(torch.load('discriminator_state.pth'))
# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCELoss()

# Training loop
num_epochs = 50
batch_size = 32

for epoch in range(num_epochs):
    for _ in range(10):  # Simulate 10 batches per epoch
        # Generate random images (28x28, flattened to 784)
        real_images = torch.randn(batch_size, 784)
        current_batch_size = real_images.size(0)

        # Train Discriminator with real images
        discriminator.zero_grad()
        real_labels = torch.ones(current_batch_size, 1)
        real_output = discriminator(real_images)
        loss_real = criterion(real_output, real_labels)

        # Generate fake images
        noise = torch.randn(current_batch_size, 100)
        fake_images = generator(noise)
        fake_labels = torch.zeros(current_batch_size, 1)
        fake_output = discriminator(fake_images.detach())
        loss_fake = criterion(fake_output, fake_labels)

        # Total discriminator loss
        loss_d = (loss_real + loss_fake) / 2
        loss_d.backward()
        optimizer_D.step()

        # Train Generator
        generator.zero_grad()
        trick_labels = torch.ones(current_batch_size, 1)
        output_trick = discriminator(fake_images)
        loss_g = criterion(output_trick, trick_labels)
        loss_g.backward()
        optimizer_G.step()

    # Show images every few epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss D: {loss_d.item()}, Loss G: {loss_g.item()}")
        with torch.no_grad():
            test_noise = torch.randn(5, 100)
            test_images = generator(test_noise)
            show_images(test_images)



torch.save(generator.state_dict(), 'generator_state.pth')
torch.save(discriminator.state_dict(), 'discriminator_state.pth')