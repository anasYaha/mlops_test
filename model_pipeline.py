"""
Pipeline modulaire pour la prédiction du churn client
Groupe AAZ - Mini Projet Churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Charger les données depuis un fichier CSV"""
    print(f"Chargement du fichier: {filepath}")
    
    data = pd.read_csv(filepath)
    
    print(f"  - Dimensions: {data.shape[0]} lignes × {data.shape[1]} colonnes")
    print("Données chargées avec succès\n")
    
    return data


def prepare_data(data):
    """Prétraiter les données"""
    print("Prétraitement des données...")
    
    df = data.copy()
    
    # Gestion des valeurs manquantes
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing_count = df['TotalCharges'].isnull().sum()
    if missing_count > 0:
        print(f"  - Remplacement de {missing_count} valeurs manquantes")
        df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Suppression de customerID
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("  - Colonne customerID supprimée")
    
    # Encodage de la variable cible
    if df['Churn'].dtype == 'object':
        df['Churn'] = (df['Churn'] == 'Yes').astype(int)
        print("  - Variable cible encodée (0=Non, 1=Oui)")
    
    # Encodage des variables catégorielles
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        print(f"  - Encodage de {len(categorical_cols)} variables catégorielles")
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print(f"Prétraitement terminé - Shape: {df.shape}\n")
    
    return df


def train_model(data_frame, test_size=0.2):
    """Entraîner le modèle SVM"""
    print(f"Entraînement du modèle SVM...")
    
    # Séparer X et y
    X = data_frame.drop('Churn', axis=1)
    y = data_frame['Churn']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"  - Train: {X_train.shape[0]} échantillons")
    print(f"  - Test: {X_test.shape[0]} échantillons")
    
    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  - Normalisation effectuée")
    
    # Équilibrage avec SMOTE
    print(f"  - Équilibrage avec SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    print(f"  - Après SMOTE: {len(X_train_balanced)} échantillons")
    
    # Entraînement du SVM
    print(f"  - Entraînement du SVM linéaire...")
    clf = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    clf.fit(X_train_balanced, y_train_balanced)
    
    print(f"Modèle entraîné\n")
    
    return clf, X_test_scaled, y_test, scaler


def evaluate_model(clf, x_test, y_test):
    """Évaluer les performances du modèle"""
    print(f"Évaluation du modèle...")
    
    y_pred = clf.predict(x_test)
    y_pred_proba = clf.predict_proba(x_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\nRÉSULTATS:")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc_auc:.4f}")
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nMatrice de Confusion:")
    print(f"  {cm}")
    
    print(f"\nRapport de Classification:")
    report = classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])
    print(report)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title('Matrice de Confusion', fontsize=14, fontweight='bold')
    plt.ylabel('Valeurs Réelles')
    plt.xlabel('Valeurs Prédites')
    plt.tight_layout()
    plt.show()
    
    print("Évaluation terminée\n")
    
    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'report': report
    }


def save_model(clf, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
    """Sauvegarder le modèle et le scaler"""
    print(f"Sauvegarde du modèle...")
    joblib.dump(clf, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  - Modèle sauvegardé dans {model_path}")
    print(f"  - Scaler sauvegardé dans {scaler_path}")
    print("Sauvegarde terminée\n")


def load_model(model_path='model.pkl', scaler_path='scaler.pkl'):
    """Charger un modèle et un scaler"""
    print(f"Chargement du modèle...")
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    print(f"  - Modèle chargé depuis {model_path}")
    print(f"  - Scaler chargé depuis {scaler_path}")
    print("Chargement terminé\n")
    return clf, scaler
