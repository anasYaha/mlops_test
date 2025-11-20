"""
Script principal pour exécuter le pipeline ML
Atelier 2 - Projet Churn Groupe AAZ
"""
import argparse
import sys
from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def main():
    """Fonction principale avec arguments CLI"""
    parser = argparse.ArgumentParser(
        description='Pipeline ML pour la prédiction du churn client'
    )
    
    parser.add_argument(
        '--action',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'predict', 'full'],
        help='Action à exécuter: train, evaluate, predict, ou full'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default='Telco-Customer-Churn.csv',
        help='Chemin vers le fichier CSV (default: Telco-Customer-Churn.csv)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='model.pkl',
        help='Chemin du modèle (default: model.pkl)'
    )
    
    parser.add_argument(
        '--scaler',
        type=str,
        default='scaler.pkl',
        help='Chemin du scaler (default: scaler.pkl)'
    )
    
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion pour le test (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PIPELINE DE PRÉDICTION DU CHURN CLIENT")
    print("Groupe AAZ - Mini Projet")
    print("="*60 + "\n")
    
    if args.action == 'full':
        print("Action: Pipeline complet\n")
        run_full_pipeline(args)
    
    elif args.action == 'train':
        print("Action: Entraînement du modèle\n")
        run_training(args)
    
    elif args.action == 'evaluate':
        print("Action: Évaluation du modèle\n")
        run_evaluation(args)
    
    elif args.action == 'predict':
        print("Action: Prédiction\n")
        run_prediction(args)


def run_full_pipeline(args):
    """Exécuter le pipeline complet"""
    try:
        print(f"1. Chargement des données depuis {args.data}...")
        data = load_data(args.data)
        print(f"    {len(data)} lignes chargées")
        
        print(f"\n2. Prétraitement des données...")
        data_frame = prepare_data(data)
        print(f"    Données prétraitées")
        
        print(f"\n3. Entraînement du modèle (test_size={args.test_size})...")
        clf, x_test, y_test, scaler = train_model(data_frame, test_size=args.test_size)
        print(f"    Modèle entraîné")
        
        print(f"\n4. Évaluation du modèle...")
        results = evaluate_model(clf, x_test, y_test)
        print(f"    Accuracy: {results['accuracy']:.4f}")
        
        print(f"\n5. Sauvegarde du modèle dans {args.model}...")
        save_model(clf, scaler, args.model, args.scaler)
        print(f"    Modèle sauvegardé")
        
        print("\n" + "="*60)
        print("Pipeline terminé avec succès")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        sys.exit(1)


def run_training(args):
    """Entraîner et sauvegarder le modèle"""
    try:
        print(f"Chargement des données depuis {args.data}...")
        data = load_data(args.data)
        
        print("Prétraitement des données...")
        data_frame = prepare_data(data)
        
        print("Entraînement du modèle...")
        clf, x_test, y_test, scaler = train_model(data_frame, test_size=args.test_size)
        
        print(f"Sauvegarde du modèle dans {args.model}...")
        save_model(clf, scaler, args.model, args.scaler)
        
        print("\nEntraînement terminé!\n")
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        sys.exit(1)


def run_evaluation(args):
    """Évaluer un modèle existant"""
    try:
        print(f"Chargement du modèle depuis {args.model}...")
        clf, scaler = load_model(args.model, args.scaler)
        
        print(f"Chargement des données de test depuis {args.data}...")
        data = load_data(args.data)
        data_frame = prepare_data(data)
        
        # Utiliser les mêmes données de test
        X = data_frame.drop('Churn', axis=1)
        y = data_frame['Churn']
        
        from sklearn.model_selection import train_test_split
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        
        X_test_scaled = scaler.transform(X_test)
        
        print("Évaluation du modèle...")
        results = evaluate_model(clf, X_test_scaled, y_test)
        
        print("\nÉvaluation terminée!\n")
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        sys.exit(1)


def run_prediction(args):
    """Faire des prédictions"""
    try:
        print(f"Chargement du modèle depuis {args.model}...")
        clf, scaler = load_model(args.model, args.scaler)
        
        print(f"Chargement des données depuis {args.data}...")
        data = load_data(args.data)
        data_frame = prepare_data(data)
        
        X = data_frame.drop('Churn', axis=1)
        y_true = data_frame['Churn']
        
        X_scaled = scaler.transform(X)
        
        print("Prédictions en cours...")
        y_pred = clf.predict(X_scaled)
        y_pred_proba = clf.predict_proba(X_scaled)[:, 1]
        
        print(f"\nRÉSULTATS DES PRÉDICTIONS")
        print("="*50)
        print(f"Nombre total: {len(y_pred)}")
        print(f"Clients à risque: {sum(y_pred == 1)} ({sum(y_pred == 1)/len(y_pred)*100:.1f}%)")
        print(f"Clients sans risque: {sum(y_pred == 0)} ({sum(y_pred == 0)/len(y_pred)*100:.1f}%)")
        print(f"Score moyen: {y_pred_proba.mean():.3f}")
        print("="*50)
        
        print("\nPrédictions terminées!\n")
        
    except Exception as e:
        print(f"\nERREUR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
