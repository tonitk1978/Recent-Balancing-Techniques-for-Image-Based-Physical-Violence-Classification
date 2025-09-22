import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Definir la clase DeepSMOTE
class DeepSMOTE:
    def __init__(self, dim_h=64, n_z=600, lr=0.0002, epochs=500, batch_size=2, device=None):  # Reducir n_z
        self.dim_h = dim_h
        self.n_z = n_z
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Inicializar encoder y decoder
        self.encoder = self.Encoder(dim_h, n_z).to(self.device)
        self.decoder = self.Decoder(dim_h, n_z).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=self.lr)
    
    # Definir la clase del encoder
    class Encoder(nn.Module):
        def __init__(self, dim_h, n_z):
            super(DeepSMOTE.Encoder, self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, dim_h, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim_h, dim_h * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dim_h * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim_h * 2, dim_h * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dim_h * 4),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim_h * 4, dim_h * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dim_h * 8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim_h * 8, dim_h * 16, 4, 2, 1, bias=False),
                nn.BatchNorm2d(dim_h * 16),
                nn.LeakyReLU(0.2, inplace=True)
            )
            self.fc = nn.Linear(dim_h * 16 * 7 * 7, n_z)

        def forward(self, x):
            x = self.conv(x)
            x = x.view(x.size(0), -1)  # Aplanar
            x = self.fc(x)
            return x
    
    # Definir la clase del decoder
    class Decoder(nn.Module):
        def __init__(self, dim_h, n_z):
            super(DeepSMOTE.Decoder, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(n_z, dim_h * 16 * 7 * 7),
                nn.ReLU()
            )
            self.deconv = nn.Sequential(
                nn.ConvTranspose2d(dim_h * 16, dim_h * 8, 4, stride=2, padding=1),  # Salida: 14x14
                nn.BatchNorm2d(dim_h * 8),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim_h * 8, dim_h * 4, 4, stride=2, padding=1),  # Salida: 28x28
                nn.BatchNorm2d(dim_h * 4),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim_h * 4, dim_h * 2, 4, stride=2, padding=1),  # Salida: 56x56
                nn.BatchNorm2d(dim_h * 2),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim_h * 2, dim_h, 4, stride=2, padding=1),  # Salida: 112x112
                nn.BatchNorm2d(dim_h),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim_h, 3, 4, stride=2, padding=1),  # Salida: 224x224
                nn.Tanh()
            )

        def forward(self, x):
            x = self.fc(x)
            x = x.view(-1, 1024, 7, 7)  # Asegurarse de que la cantidad de canales sea correcta (1024 canales)
            x = self.deconv(x)
            return x

    # Método para entrenar el autoencoder
    def train(self, train_loader):
        self.encoder.train()
        self.decoder.train()

        for epoch in range(self.epochs):
            total_loss = 0
            for images, _ in train_loader:
                images = images.to(self.device)
                self.optimizer.zero_grad()

                z_hat = self.encoder(images)
                x_hat = self.decoder(z_hat)
                loss = self.criterion(x_hat, images)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {total_loss/len(train_loader):.4f}')
            torch.cuda.empty_cache()

    def generate_synthetic_samples_in_batches(self, minority_class_data, n_to_sample, batch_size=1, n_neighbors=5):
        torch.cuda.empty_cache()  # Limpiar la caché antes de mover los datos
        
        encoded_minority_data = []
        for i in range(0, len(minority_class_data), batch_size):
            batch = minority_class_data[i:i + batch_size].to(self.device)
            with torch.no_grad():  # Evitar cálculos de gradientes
                encoded_batch = self.encoder(batch).detach().cpu().numpy()
            encoded_minority_data.append(encoded_batch)
            torch.cuda.empty_cache()  # Liberar la memoria después de cada lote

        encoded_minority_data = np.vstack(encoded_minority_data)

        # Aplicar SMOTE en el espacio latente
        nn_model = NearestNeighbors(n_neighbors=n_neighbors+1, n_jobs=-1)
        nn_model.fit(encoded_minority_data)

        synthetic_images = []
        for i in range(0, n_to_sample, batch_size):
            actual_batch_size = min(batch_size, n_to_sample - i)

            # Seleccionar índices base
            base_indices = np.random.choice(len(encoded_minority_data), actual_batch_size)
            base_points = encoded_minority_data[base_indices]

            # Encontrar los vecinos más cercanos a cada punto base
            neighbor_indices = nn_model.kneighbors(base_points, return_distance=False)[:, 1:n_neighbors+1]

            # Seleccionar uno de los vecinos cercanos para cada punto base
            random_neighbors = np.random.choice(n_neighbors, actual_batch_size)
            X_neighbor = encoded_minority_data[neighbor_indices[np.arange(actual_batch_size), random_neighbors]]

            # Interpolación lineal entre los puntos base y los vecinos seleccionados
            synthetic_latent_samples = base_points + np.multiply(np.random.rand(actual_batch_size, 1), (X_neighbor - base_points))

            # Decodificar las muestras sintéticas generadas en lotes
            with torch.no_grad():  # Asegurar que no se calculen gradientes
                synthetic_latent_samples = torch.Tensor(synthetic_latent_samples).to(self.device)
                synthetic_images_batch = self.decoder(synthetic_latent_samples).cpu().detach().numpy()
                synthetic_images.append(synthetic_images_batch)

            # Limpiar caché periódicamente para evitar errores de memoria
            if i % (10 * batch_size) == 0:
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        return np.vstack(synthetic_images)  # Concatenar todos los lotes generados



    

# Ejemplo de uso:
# violence_images_tensor = torch.stack(violence_images)
# samples_to_generate = 100
# deep_smote = DeepSMOTE()
# synthetic_images = deep_smote.generate_synthetic_samples_in_batches(violence_images_tensor, samples_to_generate, batch_size=2)
