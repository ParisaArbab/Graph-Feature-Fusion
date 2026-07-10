# Graph Representation Learning using Feature Fusion

This project explores a simple yet effective approach for graph learning by combining multiple graph and node features into a single feature vector. Instead of relying only on Graph Neural Networks (GNNs), the project evaluates a Multi-Layer Perceptron (MLP) as a lightweight alternative across multiple benchmark datasets.

## Features

- Graph feature fusion through feature concatenation
- MLP-based graph classification
- Evaluation on multiple graph datasets
- Performance comparison across datasets
- Modular and easy-to-extend implementation

## Technologies

- Python
- PyTorch
- NumPy
- Scikit-learn
- Jupyter Notebook

## Workflow

1. Load graph dataset.
2. Extract multiple graph features.
3. Concatenate features into a unified representation.
4. Train an MLP classifier.
5. Evaluate model performance.

## Goal

Investigate whether a simple feature engineering approach combined with an MLP can achieve competitive performance on graph learning tasks while reducing model complexity.

## Author

**Parisa Arbab**

## Create Environment
It is better to create a virtual environment for this project.

```bash
python3 -m venv <myenvpath>
```
Install all required packages
```bash
pip install -r requirements.txt
```
