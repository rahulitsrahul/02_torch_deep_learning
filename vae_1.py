import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the Encoder and Decoder classes
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(1, 64)
        self.fc_mean = nn.Linear(64, latent_dim)
        self.fc_log_var = nn.Linear(64, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        latent_mean = self.fc_mean(x)
        latent_log_var = self.fc_log_var(x)
        return latent_mean, latent_log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 64)
        self.fc_out = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc(x))
        reconstructed_x = self.fc_out(x)
        return reconstructed_x

# Define the VAE model
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)  # Calculate the standard deviation
        eps = torch.randn_like(std)     # Generate a sample from N(0,1)
        return mean + eps * std         # Reparameterization trick

    def forward(self, x):
        latent_mean, latent_log_var = self.encoder(x)
        z = self.reparameterize(latent_mean, latent_log_var)
        reconstructed_x = self.decoder(z)
        return reconstructed_x, latent_mean, latent_log_var

# Create synthetic data from a univariate Gaussian distribution
data_mean = 3
data_std = 1.5
data = torch.randn(1000) * data_std + data_mean

# Create the VAE model
latent_dim = 1
vae = VAE(latent_dim)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(vae.parameters(), lr=0.001)

# Training the VAE
latent_space_data = []

num_epochs = 1000
for epoch in range(num_epochs):
    vae.train()
    optimizer.zero_grad()
    reconstructed_data, latent_mean, latent_log_var = vae(data.unsqueeze(1).float())
    
    # Calculate Reconstruction Loss
    reconstruction_loss = criterion(reconstructed_data, data.unsqueeze(1).float())
    
    # Calculate KL Divergence Loss
    kl_div_loss = -0.5 * torch.sum(1 + latent_log_var - latent_mean.pow(2) - latent_log_var.exp())
    
    # Combine the two losses
    loss = reconstruction_loss + kl_div_loss
    
    # Backpropagation
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}")
        
    if epoch % 50 == 0:
        # Store latent space representations for visualization
        vae.eval()
        with torch.no_grad():
            latent_means, _ = vae.encoder(data.unsqueeze(1).float())
            latent_space_data.append(latent_means.numpy())

# Flatten and concatenate latent space representations
latent_space_data = np.concatenate(latent_space_data, axis=0)

# Generating samples from the learned latent space
vae.eval()
with torch.no_grad():
    z_samples = torch.randn(1000, latent_dim)
    generated_data = vae.decoder(z_samples).detach().numpy()

# Plotting the original data and generated samples
plt.figure(figsize=(8, 4))
plt.hist(data.numpy(), bins=30, alpha=0.5, label='Original Data', density=True)
plt.hist(generated_data, bins=30, alpha=0.5, label='Generated Samples', density=True)
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.title('Original Data vs Generated Samples')
plt.show()

# Plotting the learned Gaussian distribution in the latent space
plt.figure(figsize=(6, 4))
plt.hist(latent_space_data, bins=30, alpha=0.7, color='orange', label='Latent Space Distribution', density=True)
plt.xlabel('Latent Space')
plt.ylabel('Density')
plt.legend()
plt.title('Learned Gaussian Distribution in Latent Space')
plt.show()
