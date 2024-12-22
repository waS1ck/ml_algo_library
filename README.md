
# ML Library

An open-source machine learning library implementing core algorithms from scratch in Python. This library is designed for educational purposes and includes tools for data handling, visualization, and model evaluation.


## Features

- Algorithms:
    - Linear Regression
    - Logistic Regression
    - k-Nearest Neighbors (kNN)
    - k-Means Clustering
- Visualization: Matplotlib for data and model visualizations.
- Data Handling: Easy dataset loading and splitting using Pandas.
- Testing and Benchmarking: Scripts to verify algorithm correctness and runtime.
## Getting Started

1. Prerequisites

Make sure you have Python 3.7+ installed along with the following libraries:

- Pandas
- NumPy
- Matplotlib

Install them using:

    pip install -r requirements.txt

2. Installation

Clone this repository to your local machine:

    git clone https://github.com/yourusername/ml-library.git
    cd ml-library
3. Usage

Each algorithm can be tested with its corresponding dataset. Run the main.py script and specify the algorithm and dataset:

    python main.py --algorithm linear_regression --dataset your_dataset.csv

Replace your_dataset.csv with the dataset provided for the specific algorithm. Supported algorithms include:

- linear_regression
- logistic_regression
- knn
- kmeans

    


## Example

Here’s an example usage for Linear Regression:

    from algorithms.linear_regression import LinearRegression
    from utils.data_loader import load_csv
    from utils.train_test_split import train_test_split

    # Load dataset
    data = load_csv("your_dataset.csv")
    X = data.drop(columns="target").values
    y = data["target"].values

    # Train-test split
    X_train, X_test = train_test_split(data)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)
    print(predictions)
## Folder Structure

    ml_library/
    │
    ├── algorithms/                # Core ML algorithms
    ├── utils/                     # Utilities for data handling
    ├── tests/                     # Test scripts
    ├── notebooks/                 # Jupyter notebooks
    ├── README.md                  # Project documentation
    ├── requirements.txt           # Dependencies
    └── main.py                    # Entry point script

## Handling Different Datasets

This library supports flexibility for testing algorithms with unique datasets:
- Ensure each dataset matches the algorithm's expected format (e.g., labeled columns, no missing values).
- Update the main.py script to dynamically load and preprocess datasets for each algorithm.
