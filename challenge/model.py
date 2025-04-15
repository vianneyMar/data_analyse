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

#Set features labels 
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


# Paramètres du modèle adaptés à vos données spécifiques
NUM_CLASSES = 14  # 14 classes de comportements
SEQUENCE_LENGTH = 100  # 100 frames = 1 seconde
MOCAP_FEATURES = 126  # 126 features pour les données de position
PRESSURE_FEATURES = 50  # 50 features pour les données de pression
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5  # Régularisation L2 pour limiter le surapprentissage

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
        
        # Branche pour les données de motion capture (positions)
        self.mocap_cnn = nn.Sequential(
            nn.Conv1d(mocap_features, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        # Calcul de la dimension d'entrée pour le LSTM après les couches CNN et pooling
        self.mocap_lstm_input_dim = 256
        self.mocap_lstm = nn.LSTM(
            self.mocap_lstm_input_dim, 
            hidden_dim, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        # Branche pour les données de pression
        self.pressure_cnn = nn.Sequential(
            nn.Conv1d(pressure_features, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.pressure_lstm_input_dim = 128
        self.pressure_lstm = nn.LSTM(
            self.pressure_lstm_input_dim, 
            hidden_dim // 2, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.2
        )
        
        # Feature fusion
        combined_dim = hidden_dim * 2 + hidden_dim  # Dimensions combinées des deux branches
        
        # Mécanisme d'attention pour mettre en avant les features importantes
        self.attention = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, mocap, pressure):
        batch_size = mocap.size(0)
        
        # Traitement des données de motion capture
        mocap_cnn_in = mocap.transpose(1, 2)  # [batch, seq_len, features] -> [batch, features, seq_len]
        mocap_cnn_out = self.mocap_cnn(mocap_cnn_in)
        mocap_lstm_in = mocap_cnn_out.transpose(1, 2)  # [batch, features, seq_len] -> [batch, seq_len, features]
        
        # Initialisation des états cachés LSTM
        h0_mocap = torch.zeros(2 * 2, batch_size, 128).to(mocap.device)  # 2 layers * 2 directions
        c0_mocap = torch.zeros(2 * 2, batch_size, 128).to(mocap.device)
        
        # Passage dans le LSTM
        mocap_lstm_out, _ = self.mocap_lstm(mocap_lstm_in, (h0_mocap, c0_mocap))
        
        # Traitement similaire pour les données de pression
        pressure_cnn_in = pressure.transpose(1, 2)
        pressure_cnn_out = self.pressure_cnn(pressure_cnn_in)
        pressure_lstm_in = pressure_cnn_out.transpose(1, 2)
        
        h0_pressure = torch.zeros(2 * 2, batch_size, 64).to(pressure.device)  # 2 layers * 2 directions
        c0_pressure = torch.zeros(2 * 2, batch_size, 64).to(pressure.device)
        
        pressure_lstm_out, _ = self.pressure_lstm(pressure_lstm_in, (h0_pressure, c0_pressure))
        
        # Extraction des features
        # Utiliser la dernière sortie et aussi global average pooling pour capturer 
        # les informations importantes sur toute la séquence
        mocap_last = mocap_lstm_out[:, -1, :]
        pressure_last = pressure_lstm_out[:, -1, :]
        
        # Global average pooling pour obtenir des représentations globales
        mocap_global = torch.mean(mocap_lstm_out, dim=1)
        pressure_global = torch.mean(pressure_lstm_out, dim=1)
        
        # Concaténation des features des deux branches 
        # (on pourrait utiliser différentes stratégies de fusion ici)
        combined_features = torch.cat((mocap_last, pressure_last, mocap_global), dim=1)
        
        # Application du mécanisme d'attention
        attention_weights = self.attention(combined_features)
        attended_features = combined_features * attention_weights
        
        # Classification finale
        output = self.classifier(attended_features)
        
        return output

# Fonction de normalisation des données
def normalize_data(data):
    """Normalise les données par feature"""
    mean = np.mean(data, axis=(0, 1), keepdims=True)
    std = np.std(data, axis=(0, 1), keepdims=True)
    # Évite la division par zéro
    std = np.where(std < 1e-10, 1e-10, std)
    return (data - mean) / std

# Fonction d'entraînement avec early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100, device='cuda', patience=15):
    model.to(device)
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Early stopping
    best_val_loss = float('inf')
    best_model_state = None
    counter = 0
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True)
    
    for epoch in range(epochs):
        # Mode entraînement
        model.train()
        running_loss = 0.0
        
        for batch in train_loader:
            mocap = batch['mocap'].to(device)
            pressure = batch['pressure'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(mocap, pressure)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_train_loss = running_loss / len(train_loader)
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
                
                outputs = model(mocap, pressure)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_val_loss = val_loss / len(val_loader)
        val_losses.append(epoch_val_loss)
        
        accuracy = accuracy_score(all_labels, all_preds)
        val_accuracies.append(accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Update learning rate
        scheduler.step(epoch_val_loss)
        
        # Early stopping check
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                model.load_state_dict(best_model_state)
                break
    
    # Si early stopping n'a pas été déclenché, chargez le meilleur modèle
    if epoch == epochs - 1:
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
    
    # Simule des données pour l'exemple - À remplacer par le chargement de vos données réelles
    num_samples = 6938
    
    # Données simulées - À remplacer par vos données réelles
    
    dataloader = Dataloader()
    mocap_data = dataloader.load_data("train_mocap.h5",(6938,100,129))
    pressure_data = dataloader.load_data("train_insoles.h5",(6938,100,50))
    labels = dataloader.load_data("train_labels.h5", (6938))
    
    # Normalisation des données
    print("Normalisation des données...")
    mocap_data = normalize_data(mocap_data)
    pressure_data = normalize_data(pressure_data)
    



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
    
    # Sauvegarde du modèle
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
    }, 'movement_classifier_checkpoint.pth')
    
    print("Modèle sauvegardé avec succès!")

    # Fonction de prédiction pour utilisation future
    def predict(model, df_mocap, df_pressure, device, mean_std_mocap=None, mean_std_pressure=None):
        """Fonction pour prédire une classe sur de nouvelles données formatées avec DataFrame"""
        model.eval()

        # Convertir en numpy array
        mocap_np = df_mocap.to_numpy().astype(np.float32)
        pressure_np = df_pressure.to_numpy().astype(np.float32)

        # Normalisation (si stats fournies)
        if mean_std_mocap:
            mean_mocap, std_mocap = mean_std_mocap
            mocap_np = (mocap_np - mean_mocap) / std_mocap
        if mean_std_pressure:
            mean_pressure, std_pressure = mean_std_pressure
            pressure_np = (pressure_np - mean_pressure) / std_pressure

        # Ajouter la dimension batch
        mocap_tensor = torch.FloatTensor(mocap_np).unsqueeze(0).to(device)
        pressure_tensor = torch.FloatTensor(pressure_np).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(mocap_tensor, pressure_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            pred_prob = probabilities[0][pred_class].item()

        return pred_class, pred_prob, probabilities.cpu().numpy()[0]


    # # Exemple d'utilisation pour la prédiction (à adapter avec vos données réelles)
    # sample_mocap = np.random.randn(SEQUENCE_LENGTH, MOCAP_FEATURES)
    # sample_pressure = np.random.randn(SEQUENCE_LENGTH, PRESSURE_FEATURES)
    
    # Normalisation (utiliser les mêmes paramètres que pour l'entraînement)
    # Dans un cas réel, vous devriez sauvegarder les mean/std lors de l'entraînement
    # sample_mocap = (sample_mocap - np.mean(mocap_data, axis=(0, 1), keepdims=True)) / np.std(mocap_data, axis=(0, 1), keepdims=True)
    # sample_pressure = (sample_pressure - np.mean(pressure_data, axis=(0, 1), keepdims=True)) / np.std(pressure_data, axis=(0, 1), keepdims=True)
    
    # pred_class, confidence, all_probs = predict(model, sample_mocap, sample_pressure, device)
    # print(f"Prédiction: {class_names[pred_class]} avec {confidence*100:.2f}% de confiance")

if __name__ == "__main__":
    main() 