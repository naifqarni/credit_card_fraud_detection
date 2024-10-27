# Credit Card Fraud Detection


## Project Overview

This project aims to detect fraudulent transactions within credit card datasets using Jupyter Notebook as part of the Math 557 course requirements. It applies linear algebra techniques to demonstrate how they can enhance machine learning and classification in fraud detection.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset contains anonymized credit card transactions labeled as fraudulent or legitimate. Various features were transformed and selected to build an effective fraud detection model. These features are essential for identifying patterns in the data that may signify fraud.

## Approach

This project uses linear algebra techniques within Jupyter Notebook to facilitate analysis, including:

- **Data Preprocessing**: Standardizing and normalizing features to improve model performance.
- **Dimensionality Reduction**: Principal Component Analysis (PCA) was applied to reduce the dataset's dimensionality, making the data more manageable and helping the model focus on the most critical aspects.
- **Modeling**: A classification model was developed, trained, and evaluated within the notebook to identify fraudulent transactions with accuracy. The model’s results include accuracy, precision, and recall metrics to gauge the effectiveness of the fraud detection model.

## Usage

1. **Install Dependencies**  
   Ensure you have Python, Jupyter Notebook, and the required packages installed. To install the necessary dependencies, run:
   ```bash
   pip install -r requirements.txt

2. **Open the Notebook**  
   Start Jupyter Notebook and open fraud_detection.ipynb:
   ```bash
   jupyter notebook fraud_detection.ipynb

Run the Cells
Execute each cell in the notebook to preprocess the data, train the model, and evaluate its performance. Results, including accuracy, precision, and recall metrics, will be displayed within the notebook.


## Acknowledgments

This project was developed as part of the Math 557 Linear Algebra course. 
Special thanks to the course instructor for his guidance and to the providers of the dataset for enabling this exploration of linear algebra in machine learning and fraud detection.


