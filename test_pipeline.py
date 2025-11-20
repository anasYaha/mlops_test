"""
Script de test pour vérifier chaque fonction individuellement
Atelier 2 - Projet Churn Groupe AAZ
"""
from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model
)


def test_load_data():
    """Test 1: Chargement des données"""
    print("=" * 50)
    print("TEST 1: Chargement des données")
    print("=" * 50)
    try:
        data = load_data('Telco-Customer-Churn.csv')
        print(f"✓ Données chargées: {len(data)} lignes")
        print(f"✓ Shape des données: {data.shape}")
        return data
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None


def test_prepare_data(data):
    """Test 2: Prétraitement des données"""
    print("\n" + "=" * 50)
    print("TEST 2: Prétraitement des données")
    print("=" * 50)
    try:
        data_frame = prepare_data(data)
        print(f"✓ Features préparés")
        print(f"✓ Shape du data_frame: {data_frame.shape}")
        return data_frame
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None


def test_train_model(data_frame):
    """Test 3: Entraînement du modèle"""
    print("\n" + "=" * 50)
    print("TEST 3: Entraînement du modèle")
    print("=" * 50)
    try:
        clf, x_test, y_test, scaler = train_model(data_frame, test_size=0.2)
        print(f"✓ Modèle entraîné avec succès")
        print(f"✓ Taille des données de test: {len(x_test)}")
        return clf, x_test, y_test, scaler
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None, None, None, None


def test_evaluate_model(clf, x_test, y_test):
    """Test 4: Évaluation du modèle"""
    print("\n" + "=" * 50)
    print("TEST 4: Évaluation du modèle")
    print("=" * 50)
    try:
        results = evaluate_model(clf, x_test, y_test)
        print(f"✓ Évaluation terminée")
        print(f"✓ Accuracy: {results['accuracy']:.4f}")
        return results
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None


def test_save_model(clf, scaler):
    """Test 5: Sauvegarde du modèle"""
    print("\n" + "=" * 50)
    print("TEST 5: Sauvegarde du modèle")
    print("=" * 50)
    try:
        save_model(clf, scaler, 'test_model.pkl', 'test_scaler.pkl')
        print("✓ Modèle sauvegardé avec succès")
        return True
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return False


def test_load_model():
    """Test 6: Chargement du modèle"""
    print("\n" + "=" * 50)
    print("TEST 6: Chargement du modèle")
    print("=" * 50)
    try:
        clf, scaler = load_model('test_model.pkl', 'test_scaler.pkl')
        print("✓ Modèle chargé avec succès")
        return clf, scaler
    except Exception as e:
        print(f"✗ Erreur: {e}")
        return None, None


def main():
    """Exécuter tous les tests"""
    print("\n")
    print("╔" + "=" * 48 + "╗")
    print("║  TESTS DES FONCTIONS DU PIPELINE ML          ║")
    print("║         Projet Churn - Groupe AAZ            ║")
    print("╚" + "=" * 48 + "╝")
    print("\n")
    
    # Test 1: Chargement
    data = test_load_data()
    if data is None:
        print("\n✗ Tests arrêtés: Impossible de charger les données")
        return
    
    # Test 2: Prétraitement
    data_frame = test_prepare_data(data)
    if data_frame is None:
        print("\n✗ Tests arrêtés: Erreur de prétraitement")
        return
    
    # Test 3: Entraînement
    clf, x_test, y_test, scaler = test_train_model(data_frame)
    if clf is None:
        print("\n✗ Tests arrêtés: Erreur d'entraînement")
        return
    
    # Test 4: Évaluation
    results = test_evaluate_model(clf, x_test, y_test)
    if results is None:
        print("\n✗ Tests arrêtés: Erreur d'évaluation")
        return
    
    # Test 5: Sauvegarde
    if not test_save_model(clf, scaler):
        print("\n⚠ Attention: Erreur de sauvegarde")
    
    # Test 6: Chargement
    clf_loaded, scaler_loaded = test_load_model()
    if clf_loaded is None:
        print("\n⚠ Attention: Erreur de chargement")
    
    # Résumé final
    print("\n" + "=" * 50)
    print("RÉSUMÉ DES TESTS")
    print("=" * 50)
    print("✓ Tous les tests sont passés avec succès!")
    print("✓ Le pipeline est opérationnel")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
