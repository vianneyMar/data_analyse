import torch
import numpy as np
import pandas as pd
from load_data import Dataloader
from model import MovementClassifier, predict

def generate_kaggle_submission(model_path, test_data_path_mocap, test_data_path_insole, output_csv_path):
    """
    Génère un fichier de soumission Kaggle à partir du modèle entraîné
    
    Args:
        model_path (str): Chemin vers le fichier de checkpoint du modèle (.pth)
        test_data_path_mocap (str): Chemin vers les données de test mocap (h5)
        test_data_path_insole (str): Chemin vers les données de test insole (h5)
        output_csv_path (str): Chemin pour sauvegarder le fichier CSV de soumission
    """
    print("Chargement du checkpoint du modèle...")
    
    # Vérifier la disponibilité du GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique: {device}")
    
    # Charger le modèle entraîné et ses paramètres
    checkpoint = torch.load(model_path, map_location=device)
    
    # Récupérer les paramètres du modèle
    mocap_stats = checkpoint['mocap_stats']
    pressure_stats = checkpoint['pressure_stats']
    
    # Initialiser le modèle avec les mêmes paramètres
    model = MovementClassifier(
        mocap_features=129,  # Ajuster selon votre modèle
        pressure_features=50,  # Ajuster selon votre modèle
        hidden_dim=128,
        num_classes=14
    )
    
    # Charger les poids du modèle
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Charger les données de test
    print("Chargement des données de test...")
    dataloader = Dataloader()
    
    # Définir la forme selon la structure de vos données de test
    # Vous devrez peut-être ajuster ces dimensions
    test_mocap_data = dataloader.load_data(test_data_path_mocap, shape=(None, 100, 129))
    test_pressure_data = dataloader.load_data(test_data_path_insole, shape=(None, 100, 50))
    
    print(f"Données de test chargées: {test_mocap_data.shape}, {test_pressure_data.shape}")
    
    # Normaliser les données de test avec les mêmes statistiques que les données d'entraînement
    mean_mocap, std_mocap = mocap_stats
    mean_pressure, std_pressure = pressure_stats
    
    # Normalisation
    test_mocap_data = (test_mocap_data - mean_mocap) / std_mocap
    test_pressure_data = (test_pressure_data - mean_pressure) / std_pressure
    
    # Remplacer les valeurs NaN par 0
    test_mocap_data = np.nan_to_num(test_mocap_data)
    test_pressure_data = np.nan_to_num(test_pressure_data)
    
    print("Génération des prédictions...")
    predictions = []
    
    # Pour chaque échantillon de test
    for i in range(test_mocap_data.shape[0]):
        # Extraire l'échantillon
        mocap_sample = test_mocap_data[i:i+1]
        pressure_sample = test_pressure_data[i:i+1]
        
        # Prédire la classe
        pred_class, _, _ = predict(model, mocap_sample, pressure_sample, device)
        predictions.append(pred_class)
        
        # Afficher la progression
        if (i + 1) % 100 == 0:
            print(f"Prédictions générées pour {i + 1}/{test_mocap_data.shape[0]} échantillons")
    
    print("Création du fichier de soumission...")
    # Créer le DataFrame avec ID et prédictions
    submission_df = pd.DataFrame({
        'ID': list(range(len(predictions))),
        'TARGET': predictions
    })
    
    # Sauvegarder en CSV
    submission_df.to_csv(output_csv_path, index=False)
    print(f"Fichier de soumission sauvegardé à {output_csv_path}")

if __name__ == "__main__":
    # Chemins des fichiers (à ajuster selon votre environnement)
    MODEL_PATH = 'movement_classifier_checkpoint.pth'
    TEST_MOCAP_PATH = 'test_mocap.h5'    # Ajuster selon le nom de votre fichier de test
    TEST_INSOLE_PATH = 'test_insoles.h5'  # Ajuster selon le nom de votre fichier de test
    OUTPUT_PATH = 'kaggle_submission.csv'
    
    generate_kaggle_submission(MODEL_PATH, TEST_MOCAP_PATH, TEST_INSOLE_PATH, OUTPUT_PATH)