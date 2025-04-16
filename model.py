# Fix for the model.py file

from load_data import Dataloader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Set features labels 
insole_features = [
    "left_1", "left_2", "left_3", "left_4", "left_5", "left_6", "left_7", "left_8", "left_9", "left_10",
    "left_11", "left_12", "left_13", "left_14", "left_15", "left_16",
    "left_acc_x", "left_acc_y", "left_acc_z",
    "left_ang_x", "left_ang_y", "left_ang_z",
    "left_force", "left_center_x", "left_center_y",
    "right_1", "right_2", "right_3", "right_4", "right_5", "right_6", "right_7", "right_8", "right_9", "right_10",
    "right_11", "right_12", "right_13", "right_14", "right_15", "right_16",
    "right_acc_x", "right_acc_y", "right_acc_z",
    "right_ang_x", "right_ang_y", "right_ang_z",
    "right_force", "right_center_x", "right_center_y"
]
mocap_features = []
for i in range(1, 43):
    mocap_features.append(f"p{i}_x")
    mocap_features.append(f"p{i}_y")
    mocap_features.append(f"p{i}_z")
mocap_features.append("pelvis_x")
mocap_features.append("pelvis_y")
mocap_features.append("pelvis_z")

# Paramètres du modèle ajustés aux données réelles
NUM_CLASSES = 13  # 13 classes de comportements
SEQUENCE_LENGTH = 100  # 100 frames = 1 seconde
MOCAP_FEATURES = 129  # Ajusté pour correspondre à la dimension réelle des données
PRESSURE_FEATURES = 50  
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.0005  # Reduced learning rate to prevent NaN
WEIGHT_DECAY = 1e-4  # Increased weight decay

class MovementDataset(Dataset):
    def __init__(self, mocap_data, pressure_data, labels):
        self.mocap_data = mocap_data  # Shape: [N, SEQUENCE_LENGTH, MOCAP_FEATURES]
        self.pressure_data = pressure_data  # Shape: [N, SEQUENCE_LENGTH, PRESSURE_FEATURES]
        self.labels = labels  # Shape: [N]
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'mocap': torch.FloatTensor(self.mocap_data[idx]),
            'pressure': torch.FloatTensor(self.pressure_data[idx]),
            'label': torch.LongTensor([self.labels[idx]])[0]
        }

class MovementClassifier(nn.Module):
    def __init__(self, mocap_features, pressure_features, hidden_dim=128, num_classes=14):
        super(MovementClassifier, self).__init__()
        
        # Simplified architecture to avoid numerical instability
        # Branche pour les données de motion capture (positions)
        self.mocap_cnn = nn.Sequential(
            nn.Conv1d(mocap_features, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)  # Add dropout to reduce overfitting
        )
        
        # Calcul de la dimension d'entrée pour le LSTM après les couches CNN et pooling
        self.mocap_lstm_input_dim = 128
        self.mocap_lstm = nn.LSTM(
            self.mocap_lstm_input_dim, 
            hidden_dim, 
            num_layers=1,  # Reduced number of layers
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        # Branche pour les données de pression
        self.pressure_cnn = nn.Sequential(
            nn.Conv1d(pressure_features, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.3)  # Add dropout
        )
        
        self.pressure_lstm_input_dim = 64
        self.pressure_lstm = nn.LSTM(
            self.pressure_lstm_input_dim, 
            hidden_dim // 2, 
            num_layers=1,  # Reduced number of layers
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        # Feature fusion - Simplified
        self.mocap_fc = nn.Linear(hidden_dim*2, hidden_dim)
        self.pressure_fc = nn.Linear(hidden_dim, hidden_dim//2)
        
        # Combined features dimension
        combined_dim = hidden_dim + hidden_dim//2
           
        # Simplified classifier to avoid numerical instability
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, mocap, pressure):
        batch_size = mocap.size(0)
        
        # Traitement des données de motion capture
        mocap_cnn_in = mocap.transpose(1, 2)  # [batch, seq_len, features] -> [batch, features, seq_len]
        mocap_cnn_out = self.mocap_cnn(mocap_cnn_in)
        mocap_lstm_in = mocap_cnn_out.transpose(1, 2)  # [batch, features, seq_len] -> [batch, seq_len, features]
        
        # Initialisation des états cachés LSTM
        h0_mocap = torch.zeros(2, batch_size, 128).to(mocap.device)  # 1 layer * 2 directions
        c0_mocap = torch.zeros(2, batch_size, 128).to(mocap.device)
        
        # Passage dans le LSTM
        mocap_lstm_out, _ = self.mocap_lstm(mocap_lstm_in, (h0_mocap, c0_mocap))
        
        # Traitement similaire pour les données de pression
        pressure_cnn_in = pressure.transpose(1, 2)
        pressure_cnn_out = self.pressure_cnn(pressure_cnn_in)
        pressure_lstm_in = pressure_cnn_out.transpose(1, 2)
        
        h0_pressure = torch.zeros(2, batch_size, 64).to(pressure.device)  # 1 layer * 2 directions
        c0_pressure = torch.zeros(2, batch_size, 64).to(pressure.device)
        
        pressure_lstm_out, _ = self.pressure_lstm(pressure_lstm_in, (h0_pressure, c0_pressure))
        
        # Extraction des features
        mocap_last = mocap_lstm_out[:, -1, :]  # [batch, hidden_dim*2]
        pressure_last = pressure_lstm_out[:, -1, :]  # [batch, hidden_dim]
        
        # Apply feature transformation
        mocap_features = self.mocap_fc(mocap_last)
        pressure_features = self.pressure_fc(pressure_last)
        
        # Combine features
        combined_features = torch.cat((mocap_features, pressure_features), dim=1)
        
        # Classification finale
        output = self.classifier(combined_features)
        
        return output

# Improved normalization function to handle outliers and prevent NaN
def normalize_data(data):
    """Normalise les données par feature avec clipping pour éviter les valeurs extrêmes"""
    # Check for NaN values before normalization
    if np.isnan(data).any():
        print("Warning: NaN values found in data before normalization")
        # Replace NaN values with mean of non-NaN values
        mask = np.isnan(data)
        data[mask] = np.nanmean(data)
    
    # Clip extreme values to prevent numerical instability
    q_low = np.percentile(data, 1, axis=(0, 1), keepdims=True)
    q_high = np.percentile(data, 99, axis=(0, 1), keepdims=True)
    data = np.clip(data, q_low, q_high)
    
    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    # Ensure std is not too small
    std = np.maximum(std, 1e-6)
    
    normalized_data = (data - mean) / std
    
    # Check for NaN values after normalization
    if np.isnan(normalized_data).any():
        print("Warning: NaN values found after normalization")
        # Replace any remaining NaN with 0
        normalized_data = np.nan_to_num(normalized_data)
    
    return normalized_data, (mean, std)

# Fonction de prédiction à utiliser après l'entraînement
def predict(model, mocap_data, pressure_data, device, mocap_stats=None, pressure_stats=None):
    """Fonction pour prédire une classe sur de nouvelles données"""
    model.eval()
    
    # Conversion en array numpy si les données sont en DataFrame
    if isinstance(mocap_data, pd.DataFrame):
        mocap_data = mocap_data.to_numpy().astype(np.float32)
    if isinstance(pressure_data, pd.DataFrame):
        pressure_data = pressure_data.to_numpy().astype(np.float32)
    
    # S'assurer que les données ont la forme [sequence_length, features]
    # Si c'est une seule séquence, ajouter dimension batch
    if len(mocap_data.shape) == 2:
        mocap_data = mocap_data.reshape(1, *mocap_data.shape)
        pressure_data = pressure_data.reshape(1, *pressure_data.shape)
    
    # Normalisation si des statistiques sont fournies
    if mocap_stats:
        mean_mocap, std_mocap = mocap_stats
        mocap_data = (mocap_data - mean_mocap) / std_mocap
    if pressure_stats:
        mean_pressure, std_pressure = pressure_stats
        pressure_data = (pressure_data - mean_pressure) / std_pressure
    
    # Handle NaN values
    mocap_data = np.nan_to_num(mocap_data)
    pressure_data = np.nan_to_num(pressure_data)
    
    # Conversion en tensors pour PyTorch
    mocap_tensor = torch.FloatTensor(mocap_data).to(device)
    pressure_tensor = torch.FloatTensor(pressure_data).to(device)
    
    with torch.no_grad():
        output = model(mocap_tensor, pressure_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities, dim=1).item()
        pred_prob = probabilities[0][pred_class].item()
    
    return pred_class, pred_prob, probabilities.cpu().numpy()[0]

# Improved training function with gradient clipping and NaN checking
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, device='cuda', patience=15):
    model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    # Improved learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, verbose=True)
    
    for epoch in range(epochs):
        # Mode entraînement
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            mocap = batch['mocap'].to(device)
            pressure = batch['pressure'].to(device)
            labels = batch['label'].to(device)
            
            # Check for NaN values in input data
            if torch.isnan(mocap).any() or torch.isnan(pressure).any():
                print("Warning: NaN values in input batch detected")
                # Replace NaN with zeros
                mocap = torch.nan_to_num(mocap)
                pressure = torch.nan_to_num(pressure)
            
            optimizer.zero_grad()
            
            outputs = model(mocap, pressure)
            loss = criterion(outputs, labels)
            
            # Check if loss is NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss encountered in epoch {epoch+1}")
                continue  # Skip this batch
                
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else float('inf')
        train_losses.append(epoch_train_loss)
        
        # Mode évaluation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                mocap = batch['mocap'].to(device)
                pressure = batch['pressure'].to(device)
                labels = batch['label'].to(device)
                
                # Handle NaN values
                mocap = torch.nan_to_num(mocap)
                pressure = torch.nan_to_num(pressure)
                
                outputs = model(mocap, pressure)
                loss = criterion(outputs, labels)
                
                # Skip if loss is NaN
                if torch.isnan(loss):
                    continue
                    
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_losses.append(epoch_val_loss)
        
        # Skip accuracy calculation if predictions are empty
        if len(all_preds) > 0:
            accuracy = accuracy_score(all_labels, all_preds)
            val_accuracies.append(accuracy)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.4f}')
        else:
            val_accuracies.append(0.0)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Accuracy: N/A (no valid predictions)')
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Early stopping check - ensure we aren't stopping with NaN values
        if epoch_val_loss < best_val_loss and not np.isnan(epoch_val_loss):
            best_val_loss = epoch_val_loss
            # Deep copy the model state
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                # Check if best model state exists before loading
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break
    
    # If early stopping not triggered, load best model if available
    if epoch == epochs - 1 and best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return train_losses, val_losses, val_accuracies

# Fonction d'évaluation complète
def evaluate_model(model, test_loader, class_names, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            mocap = batch['mocap'].to(device)
            pressure = batch['pressure'].to(device)
            labels = batch['label'].to(device)
            
            # Handle NaN values
            mocap = torch.nan_to_num(mocap)
            pressure = torch.nan_to_num(pressure)
            
            outputs = model(mocap, pressure)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    return accuracy, conf_matrix, report

# Fonction principale
def main():
    # Définissez vos noms de classes ici (remplacez par les noms réels)
    class_names = [f"Action_{i}" for i in range(NUM_CLASSES)]
    
    # Ici, vous devrez charger vos données réelles
    print("Chargement des données...")
    
    # Initialisation du dataloader avec le bon chemin de fichiers
    dataloader = Dataloader()
    
    # Chargement des données avec les bonnes dimensions
    try:
        mocap_data = dataloader.load_data("train_mocap.h5",(6938, 100, 129))
        pressure_data = dataloader.load_data("train_insoles.h5",(6938, 100, 50))
        labels = dataloader.load_data("train_labels.h5", (6938))
        
        # Check for NaN values in the data
        if np.isnan(mocap_data).any():
            print(f"NaN values found in mocap_data: {np.isnan(mocap_data).sum()} values")
        if np.isnan(pressure_data).any():
            print(f"NaN values found in pressure_data: {np.isnan(pressure_data).sum()} values")
        if np.isnan(labels).any():
            print(f"NaN values found in labels: {np.isnan(labels).sum()} values")
            
        # Normalisation des données et récupération des stats pour usage futur
        print("Normalisation des données...")
        mocap_data, mocap_stats = normalize_data(mocap_data)
        pressure_data, pressure_stats = normalize_data(pressure_data)
        
        # Séparation train/val/test avec stratification
        print("Séparation des données...")
        mocap_train, mocap_test, pressure_train, pressure_test, labels_train, labels_test = train_test_split(
            mocap_data, pressure_data, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        mocap_train, mocap_val, pressure_train, pressure_val, labels_train, labels_val = train_test_split(
            mocap_train, pressure_train, labels_train, test_size=0.15, random_state=42, stratify=labels_train
        )
        
        # Vérification de la distribution des classes
        print(f"Distribution des classes - Train: {np.bincount(labels_train)}")
        print(f"Distribution des classes - Val: {np.bincount(labels_val)}")
        print(f"Distribution des classes - Test: {np.bincount(labels_test)}")
        
        # Création des datasets et dataloaders
        train_dataset = MovementDataset(mocap_train, pressure_train, labels_train)
        val_dataset = MovementDataset(mocap_val, pressure_val, labels_val)
        test_dataset = MovementDataset(mocap_test, pressure_test, labels_test)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
        
        # Initialisation du modèle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilisation du périphérique: {device}")
        
        model = MovementClassifier(MOCAP_FEATURES, PRESSURE_FEATURES, hidden_dim=128, num_classes=NUM_CLASSES)
        print(f"Nombre total de paramètres: {sum(p.numel() for p in model.parameters())}")
        
        # Loss et optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        # Entraînement
        print("Début de l'entraînement...")
        train_losses, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            epochs=EPOCHS, device=device, patience=15
        )
        
        # Évaluation finale
        print("Évaluation du modèle...")
        accuracy, conf_matrix, report = evaluate_model(model, test_loader, class_names, device)
        print(f"Test Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        # Visualisation de la matrice de confusion
        plt.figure(figsize=(14, 12))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        # Visualisation des courbes d'apprentissage
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig('learning_curves.png')
        plt.show()
        
        # Sauvegarde du modèle et des stats de normalisation pour réutilisation future
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'mocap_stats': mocap_stats,
            'pressure_stats': pressure_stats,
            'class_names': class_names
        }, 'movement_classifier_checkpoint.pth')
        
        print("Modèle sauvegardé avec succès!")
        
        # Exemple d'utilisation pour la prédiction (sur un échantillon du jeu de test)
        print("\nExemple de prédiction:")
        sample_idx = 0  # Premier échantillon du jeu de test
        sample_mocap = mocap_test[sample_idx]
        sample_pressure = pressure_test[sample_idx]
        true_label = labels_test[sample_idx]
        
        # Reshape pour la fonction predict
        sample_mocap = sample_mocap.reshape(1, *sample_mocap.shape) 
        sample_pressure = sample_pressure.reshape(1, *sample_pressure.shape)
        
        # Prédiction
        pred_class, confidence, all_probs = predict(model, sample_mocap, sample_pressure, device)
        print(f"Vraie classe: {class_names[true_label]}")
        print(f"Prédiction: {class_names[pred_class]} avec {confidence*100:.2f}% de confiance")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()