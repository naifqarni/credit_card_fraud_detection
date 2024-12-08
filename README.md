# Credit Card Fraud Detection

## Project Overview

This project aims to detect fraudulent transactions within credit card datasets using both an interactive Streamlit application and Jupyter Notebook as part of the Math 557 course requirements. It applies linear algebra techniques to demonstrate how they can enhance machine learning and classification in fraud detection.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgments](#acknowledgments)

## Dataset

The dataset contains anonymized credit card transactions labeled as fraudulent or legitimate. Various features were transformed and selected to build an effective fraud detection model. These features are essential for identifying patterns in the data that may signify fraud.

## Features

- Interactive web interface for real-time fraud detection
- Custom implementation of machine learning algorithms:
  - Logistic Regression with gradient descent
  - K-Nearest Neighbors (KNN)
  - XGBoost for high performance
- Feature reduction techniques:
  - Singular Value Decomposition (SVD)
  - Correlation-based Feature Selection
- Comprehensive visualization of model performance
- Real-time parameter tuning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Interactive Web Application

1. Navigate to the src directory:
   ```bash
   cd src
   ```

2. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your web browser and go to the URL shown in the terminal (typically http://localhost:8501)

### Using the Jupyter Notebook

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Open `project_notebook.ipynb` in your browser

3. Execute the cells to:
   - Explore the data analysis
   - View the mathematical concepts
   - See detailed implementation of algorithms
   - Examine model performance metrics

## Acknowledgments

This project was developed as part of the Math 557 Applied Linear Algebra course under the supervision of Dr. Mohammed Alshrani. 

Special thanks to:
- Dr. Mohammed Alshrani for course guidance and project supervision
- The providers of the dataset for enabling this exploration
- NumPy and Streamlit documentation for implementation references



