import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom models
from models.SVMscratch import SVM
from models.LogisticRegressionScratch import LogisticRegressionScratch
from models.KNearestNeighborsScratch import KNearestNeighborsScratch

def load_data():
    try:
        # Load the fraud dataset
        data = pd.read_csv("data/fraud_test.csv")
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
        # Preprocessing
        for column in data.columns:
            # Convert datetime columns to numeric
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = pd.to_numeric(pd.to_datetime(data[column]))
            # Encode categorical columns
            elif pd.api.types.is_object_dtype(data[column]):
                data[column] = pd.factorize(data[column])[0]
        
        # Split features and target
        X = data.drop('is_fraud', axis=1)
        y = data['is_fraud']
        
        # Balance the dataset using random under-sampling
        fraud_samples = data[data['is_fraud'] == 1]
        non_fraud_samples = data[data['is_fraud'] == 0].sample(n=len(fraud_samples), random_state=42)
        
        # Combine the balanced samples
        balanced_data = pd.concat([fraud_samples, non_fraud_samples])
        
        # Split features and target from balanced data
        X = balanced_data.drop('is_fraud', axis=1)
        y = balanced_data['is_fraud']
        
        return X, y
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def plot_metrics(metrics, y_test, y_pred, y_prob=None):
    if 'Confusion Matrix' in metrics:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues')
        ax.set_xlabel('Predicted labels')
        ax.set_ylabel('True labels')
        st.pyplot(fig)
        plt.close()

    if y_prob is not None:
        if 'ROC Curve' in metrics:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc='lower right')
            st.pyplot(fig)
            plt.close()

        if 'Precision-Recall Curve' in metrics:
            st.subheader("Precision-Recall Curve")
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            st.pyplot(fig)
            plt.close()

def main():
    st.set_page_config(page_title="Fraud Detection App", layout="wide")
    
    st.title("Credit Card Fraud Detection")
    st.sidebar.title("Settings")
    st.markdown("Binary Classification using Custom Implementation 🎯")
    
    try:
        # Load and preprocess data
        X, y = load_data()
        if X is None or y is None:
            st.error("Failed to load data")
            return

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        class_names = ["Not Fraud", "Fraud"]

        # Classifier selection
        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox(
            "Classifier", 
            ("Support Vector Machine (SVM)", 
             "Logistic Regression",
             "K-Nearest Neighbors (KNN)")
        )

        # Metrics selection
        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve')
        )

        # Add About section in sidebar
        if st.sidebar.checkbox("Show Project Information", False):
            st.markdown("""
            # Credit Card Fraud Detection Project

            This project focuses on developing binary classification models to detect fraudulent credit card transactions. The analysis is structured in the following sections:

            1. **Exploratory Data Analysis (EDA)** - Understanding patterns and relationships in transaction data
            2. **Model Implementation:**
               - Logistic Regression
               - K-Nearest Neighbors
               - Support Vector Machine (SVM)

            3. **Feature Reduction Analysis** - Comparing model performance before and after dimensionality reduction

            The goal is to identify the most effective approach for detecting fraudulent transactions while maintaining high accuracy and minimizing false positives.
            """)

            # Data Overview Section
            st.markdown("""
            ## Data Overview

            - **Dataset:** `fraud_test.csv`
            - **Number of Records:** 93,34
            - **Number of Features:** 22
            - **Features Description:**
              - `trans_date_trans_time`: Date and time of the transaction
              - `cc_num`: Credit card number
              - `merchant`: Merchant name where the transaction took place
              - `category`: Category of the merchant
              - `amt`: Amount of the transaction
              - `state`: State where the transaction occurred
              - `zip`: ZIP code of the transaction location
              - `lat`: Latitude of the merchant location
              - `long`: Longitude of the merchant location
              - `job`: Job title of the cardholder
              - `dob`: Date of birth of the cardholder
              - `trans_num`: Transaction number
              - `unix_time`: Unix timestamp of the transaction
              - `merch_lat`: Merchant latitude
              - `merch_long`: Merchant longitude
              - `is_fraud`: Target variable indicating fraudulent transactions
            """)

            # Show distribution of fraud vs non-fraud
            if st.checkbox("Show Fraud Distribution"):
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.countplot(x='is_fraud', data=data)
                plt.title('Distribution of Fraudulent vs Non-Fraudulent Transactions')
                plt.xlabel('Is Fraud')
                plt.ylabel('Count')
                st.pyplot(fig)
                
                fraud_count = len(data[data['is_fraud'] == 1])
                non_fraud_count = len(data[data['is_fraud'] == 0])
                st.write(f"Number of fraudulent transactions: {fraud_count}")
                st.write(f"Number of non-fraudulent transactions: {non_fraud_count}")

            # Model Explanations
            if st.checkbox("Show Model Explanations"):
                st.markdown("""
                ### k-Nearest Neighbors (k-NN): A Linear Algebra Perspective

                1. **Overview**: 
                   - k-NN is a non-parametric algorithm that classifies a data point based on the majority class among its k-nearest neighbors.
                   - It relies on distance metrics to determine neighbors.

                2. **Distance Calculation**:
                   - The most common metric is the Euclidean distance:
                   ```math
                   d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sqrt{\\sum_{k=1}^n \\left( x_{i,k} - x_{j,k} \\right)^2}
                   ```
                   - In vector form:
                   ```math
                   d(\\mathbf{x}_i, \\mathbf{x}_j) = \\|\\mathbf{x}_i - \\mathbf{x}_j\\|_2
                   ```

                ### Support Vector Machine (SVM): A Linear Algebra Perspective

                1. **Overview**:
                   - SVM is a supervised learning algorithm that finds the hyperplane that best separates data into two classes.
                   - It aims to maximize the margin between the hyperplane and the nearest data points (support vectors).

                2. **Decision Boundary**:
                   - The decision boundary is a hyperplane defined as:
                   ```math
                   \\mathbf{w}^\\top \\mathbf{x} + b = 0
                   ```
                   where:
                   - w: Weight vector, normal to the hyperplane
                   - b: Bias term
                   - x: Input feature vector
                """)

            # Model Performance Comparison
            if st.checkbox("Show Model Performance Comparison"):
                st.markdown("### Model Performance Comparison")
                
                # Create sample data for comparison
                models = ['KNN (k=3)', 'SVM', 'Logistic Regression']
                accuracy = [0.87, 0.83, 0.85]
                precision = [0.80, 0.76, 0.78]
                recall = [0.97, 0.94, 0.96]
                f1 = [0.88, 0.84, 0.86]

                comparison_data = pd.DataFrame({
                    'Model': models,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1
                })

                st.table(comparison_data)

                # Plot performance metrics
                fig, ax = plt.subplots(figsize=(10, 6))
                x = np.arange(len(models))
                width = 0.2

                ax.bar(x - width*1.5, accuracy, width, label='Accuracy')
                ax.bar(x - width/2, precision, width, label='Precision')
                ax.bar(x + width/2, recall, width, label='Recall')
                ax.bar(x + width*1.5, f1, width, label='F1-Score')

                ax.set_ylabel('Scores')
                ax.set_title('Model Performance Comparison')
                ax.set_xticks(x)
                ax.set_xticklabels(models)
                ax.legend()

                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            # Show code implementation
            if st.checkbox("Show Code Implementation"):
                st.code("""
class KNearestNeighborsScratch:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
                """, language='python')

        # Model training and evaluation
        if classifier == 'Support Vector Machine (SVM)':
            st.sidebar.subheader("Model Hyperparameters")
            learning_rate = st.sidebar.number_input("Learning Rate", 0.0001, 1.0, step=0.0001, value=0.0001)
            lambda_param = st.sidebar.number_input("Lambda (Regularization)", 0.01, 10.0, step=0.01, value=0.01)
            n_iters = st.sidebar.number_input("Number of Iterations", 100, 1000, step=50, value=1000)

            if st.sidebar.button("Classify", key="classify_svm"):
                with st.spinner('Training SVM...'):
                    model = SVM(learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_pred = (y_pred + 1) // 2
                    
                    col1, col2, col3 = st.columns(3)
                    accuracy = np.mean(y_pred == y_test)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2f}")
                    with col2:
                        st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
                    with col3:
                        st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
                    
                    plot_metrics(metrics, y_test, y_pred)

        elif classifier == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            learning_rate = st.sidebar.number_input("Learning Rate", 0.001, 1.0, step=0.001, value=0.01)
            epochs = st.sidebar.number_input("Number of Epochs", 100, 100000, step=100, value=1000)

            if st.sidebar.button("Classify", key="classify_lr"):
                with st.spinner('Training Logistic Regression...'):
                    model = LogisticRegressionScratch(learning_rate=learning_rate, epochs=epochs)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)
                    
                    col1, col2, col3 = st.columns(3)
                    accuracy = np.mean(y_pred == y_test)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2f}")
                    with col2:
                        st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
                    with col3:
                        st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
                    
                    plot_metrics(metrics, y_test, y_pred, y_prob)

        else:  # KNN
            st.sidebar.subheader("Model Hyperparameters")
            k = st.sidebar.number_input("Number of neighbors (K)", 1, 10, step=2, value=3)

            if st.sidebar.button("Classify", key="classify_knn"):
                with st.spinner('Training KNN...'):
                    model = KNearestNeighborsScratch(k=k)
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    col1, col2, col3 = st.columns(3)
                    accuracy = np.mean(y_pred == y_test)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2f}")
                    with col2:
                        st.metric("Precision", f"{precision_score(y_test, y_pred):.2f}")
                    with col3:
                        st.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
                    
                    plot_metrics(metrics, y_test, y_pred)

        # Show raw data option
        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Credit Card Fraud Dataset")
            st.dataframe(pd.concat([X, y], axis=1))

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
