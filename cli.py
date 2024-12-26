import argparse
from .model import ChemModel

def main():
    parser = argparse.ArgumentParser(description="GenReP: A console app for molecular property prediction.")
    parser.add_argument("smiles", help="SMILES string of the molecule to predict.")
    parser.add_argument("--model_path", default="ensemble_model.joblib", help="Path to the trained model file.")
    parser.add_argument("--scaffolds_path", default="scaffolds.npy", help="Path to the scaffolds file.")
    parser.add_argument("--scaler_path", default="scaler.joblib", help="Path to the scaler file.")
    args = parser.parse_args()

    model = ChemModel(
        model_path=args.model_path,
        scaffolds_path=args.scaffolds_path,
        scaler_path=args.scaler_path
    )
    
    prediction, probability = model.predict(args.smiles)
    print(f"Prediction for {args.smiles}: {prediction}")
    print(f"Probability of being class 1: {probability:.4f}")

if __name__ == "__main__":
    main()
