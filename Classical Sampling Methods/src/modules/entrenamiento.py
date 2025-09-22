
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, OneSidedSelection, NearMiss, TomekLinks, EditedNearestNeighbours
from imblearn.combine import SMOTETomek, SMOTEENN



class TrainingModules:

        
    train_violent_dir = None
    train_non_violent_dir = None
    test_violent_dir = None
    test_non_violent_dir = None
    muestreos=None

    def __init__(self):
        self.train_violent_dir = "../resources/dataset/train/violencia/"
        self.train_non_violent_dir = "../resources/dataset/train/no_violencia/"
        self.test_violent_dir = "../resources/dataset/test/violencia/"
        self.test_non_violent_dir = "../resources/dataset/test/no_violencia/"
        self.muestreos = {
        "Original": "", 
        "RandomOverSampler": RandomOverSampler(),
        "RandomUnderSampler": RandomUnderSampler(),
        "SMOTE": SMOTE(),
        "ADASYN": ADASYN(),
        "OneSidedSelection": OneSidedSelection(),
        "NearMiss": NearMiss(),
        "TomekLinks": TomekLinks(),
        "SMOTETomek": SMOTETomek(),
        "SMOTEENN": SMOTEENN(),
        "ENN": EditedNearestNeighbours()
        }
    
    def load_images_and_labels(self,directory, label):
        """
        Función para cargar imágenes y etiquetas en arrays NumPy
        Recibe el path donde se encuentran las imagenes y 
        """
        images = []
        labels = []
        for filename in os.listdir(directory):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((224, 224))  # Redimensionar imágenes a 224x224
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label)
        return np.array(images), np.array(labels)
    
    def muestreo(self, X, y, metodo="Original"):
        """
        Método para muestrear el conjunto de datos, recibe:
        X: datos (imágenes)
        y: etiquetas (violencia/no violencia)
        metodo: cadena con el nombre del método de muestreo
        """
        if metodo != "Original":
            print(f"Balanceando Base de datos con: {metodo}")
            sampler = self.muestreos[metodo]
            
            # Aplanar imágenes para que el sampler funcione
            X_reshaped = X.reshape(X.shape[0], -1)  # Aplana las imágenes
            X_sampled, y_sampled = sampler.fit_resample(X_reshaped, y)
            
            # Volver a dar forma a las imágenes a su formato original
            X_sampled = X_sampled.reshape(-1, 224, 224, 3)  # Ajustar según la forma original de tus imágenes
            return X_sampled, y_sampled
        else:
            print(f"No se balancea la Base de datos: {metodo}")
            return X, y


    def getModel(self):
        ## muy bien
        # Cargar el modelo ResNet18 preentrenado
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Reemplazar la última capa completamente conectada
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)  # Ajustar para 2 clases (violencia/no violencia)

        # Congelar todas las capas convolucionales excepto el último bloque residual (layer4)
        for name, param in model.named_parameters():
            if "layer4" not in name and "fc" not in name:
                param.requires_grad = False
        return model
    
    def fit(self,epocas,model,device,train_loader,val_loader,optimizer,criterion,datasetSize):
        num_train_samples =  datasetSize#len(train_dataset)
        train_losses, train_accs = [], []
        val_losses, val_accs = [], []
        #tqdm(range(epocas), desc="Entrenando", unit="iter"):
        for epoch in tqdm(range(epocas), desc="Entrenando", unit="iter"):
                model.train()  
                running_loss = 0.0
                correct = 0
                total = 0

                for inputs, labels in train_loader:  
                    inputs = inputs.to(device) 
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                train_loss = running_loss / len(train_loader)
                train_acc = 100 * correct / total  # Calcular la precisión de entrenamiento
                train_losses.append(train_loss)
                train_accs.append(train_acc)

                # Validación (en cada época)
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:  # Usar val_loader para validación
                        inputs = inputs.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()

                        _, predicted = torch.max(outputs.data, 1) 

                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()

                val_loss /= len(val_loader)
                val_acc = 100 * val_correct / val_total 
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                print(f'Epoch [{epoch+1}/{epocas}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        del num_train_samples
        del train_losses
        del  train_accs 
        del val_losses
        del val_accs 

        return model