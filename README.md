
# GenReP

GenReP is a stand-alone tool for predicting the **direction of TP53 relative gene expression** in response to pharmaceutical compounds. Mutations and dysregulations in TP53 are implicated in approximately half of all detected cancers, making it a key target for drug discovery. GenReP’s ensemble machine learning approach—leveraging molecular fingerprints, descriptors, and scaffold-based features—aims to help researchers and pharmaceutical scientists assess how novel or existing compounds may affect TP53 expression.

---

## Table of Contents
1. [Key Features](#key-features)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Citation](#citation)
5. [License](#license)

---

## Key Features
- **Ensemble Model**: Combines multiple algorithms for improved predictive performance.
- **SMILES Input**: Accepts SMILES strings for rapid structure-based property prediction.
- **Scaffold-Aware**: Incorporates scaffold-based features to account for structural context in molecules.
- **Scalable Workflow**: Pre-trained with molecular descriptors and fingerprints, enabling quick inference.
- **Console App**: Easily integrate the CLI tool into your existing pipelines or computational workflows.

---

## Installation
1. **Clone the Repository**  
   ```
   git clone https://github.com/MLBC-lab/GenReP.git
   ```
2. **Install Dependencies**
   ```
   pip install -r requirements.txt
   ```
  ---
## Usage
### Prepare Your SMILES
+ Ensure you have your molecule’s SMILES string ready for prediction.
### Run the Prediction
 + The CLI accepts several arguments, including paths to the model, scaffolds file, and scaler.      
    + Parameters:
        -   `smiles`: SMILES string of the molecule.
        -   `--model_path`: Path to the trained model file. (Default: `ensemble_model.joblib`)
        -   `--scaffolds_path`: Path to the scaffolds file. (Default: `scaffolds.npy`)
        -   `--scaler_path`: Path to the scaler file. (Default: `scaler.joblib`)

    + Below is a simple example command:
       ```
       python -m genrep.cli \
        "CCOC(=O)c1ccc(cc1)NCc1ccccc1" \
         --model_path ensemble_model.joblib \
         --scaffolds_path scaffolds.npy \
         --scaler_path scaler.joblib
       ```
###  Interpret the Output
   + The script will print out the predicted TP53 regulation (class 0 or 1) and the **probability** of it being class 1.
     
---

## Additional CLI Examples

Below are more detailed usage examples for the CLI. Each example demonstrates how to customize model/scaffold/scaler paths, manage output, and handle batch processing.

 1. **Using Default Model, Scaffolds, and Scaler**
    ```
    python -m genrep.cli "CCOC(=O)c1ccc(cc1)NCc1ccccc1"
    ```
2. **Specifying a Custom Model Path**
   ```
   python -m genrep.cli "CCOC(=O)c1ccc(cc1)NCc1ccccc1" \
    --model_path ./my_models/custom_model.joblib
   ```
3. **Helpful Tips**
-   **Batch Processing**: While the CLI is designed for single SMILES input, you can easily wrap the command in a shell script or Python loop to process multiple SMILES strings.
  
-   **Experimental SMILES**: If the SMILES contains unusual or nonstandard notation, ensure it’s valid for RDKit or your underlying chemistry toolkit to avoid parsing errors.
  
-   **Log Outputs**: Pipe the output to a file for record-keeping:
    ```
    python -m genrep.cli "CCc1ccc(cc1)Cl" >> predictions.log
    ```
---
## Citation
  + Publication pending.
---
## License
MIT License

Copyright (c) 2023 Austin Spadaro

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
