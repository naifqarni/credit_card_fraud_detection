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
from feature_reduction.PCAScratch import PCA, find_optimal_components
from feature_reduction.SVDscratch import SVD
from feature_reduction.CorrelationFilter import CorrelationFilter

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


        # Add feature reduction selection before classifier selection
        st.sidebar.subheader("Feature Reduction")
        reduction_method = st.sidebar.selectbox(
            "Reduction Method",
            ("None", "PCA", "SVD", "Correlation Filter")
        )

        # Apply feature reduction if selected
        if reduction_method != "None":
            with st.spinner(f'Applying {reduction_method}...'):
                if reduction_method == "PCA":
                    n_components = find_optimal_components(X_train_scaled)
                    reducer = PCA(n_components=n_components)
                    X_train_reduced = reducer.fit_transform(X_train_scaled)
                    X_test_reduced = reducer.transform(X_test_scaled)
                    st.info(f"Reduced features from {X_train_scaled.shape[1]} to {n_components} components")
                
                elif reduction_method == "SVD":
                    n_components = min(X_train_scaled.shape[1] // 2, 10)  # Example: reduce to half or max 10
                    reducer = SVD(n_components=n_components)
                    X_train_reduced = reducer.fit_transform(X_train_scaled)
                    X_test_reduced = reducer.transform(X_test_scaled)
                    st.info(f"Reduced features from {X_train_scaled.shape[1]} to {n_components} components")
                
                elif reduction_method == "Correlation Filter":
                    reducer = CorrelationFilter(threshold=0.8)
                    X_train_reduced = reducer.fit_transform(X_train_scaled)
                    X_test_reduced = reducer.transform(X_test_scaled)
                    st.info(f"Reduced features from {X_train_scaled.shape[1]} to {X_train_reduced.shape[1]} features")
                
                # Update the training and test data
                X_train_scaled = X_train_reduced
                X_test_scaled = X_test_reduced

        # Move classifier and metrics selection up
        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox(
            "Classifier", 
            ("Support Vector Machine (SVM)", 
             "Logistic Regression",
             "K-Nearest Neighbors (KNN)")
        )

        metrics = st.sidebar.multiselect(
            "What metrics to plot?",
            ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve')
        )

        # Model hyperparameters sections remain here...

       

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
                    
                    # Add feature reduction information
                    st.info(f"Original number of features: {X.shape[1]}")
                    if reduction_method != "None":
                        st.info(f"Number of features after {reduction_method}: {X_train_scaled.shape[1]}")
                        st.info(f"Feature reduction: {(1 - X_train_scaled.shape[1]/X.shape[1])*100:.1f}% reduction")
                    
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
                    
                    # Add feature reduction information
                    st.info(f"Original number of features: {X.shape[1]}")
                    if reduction_method != "None":
                        st.info(f"Number of features after {reduction_method}: {X_train_scaled.shape[1]}")
                        st.info(f"Feature reduction: {(1 - X_train_scaled.shape[1]/X.shape[1])*100:.1f}% reduction")
                    
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
                    
                    # Add feature reduction information
                    st.info(f"Original number of features: {X.shape[1]}")
                    if reduction_method != "None":
                        st.info(f"Number of features after {reduction_method}: {X_train_scaled.shape[1]}")
                        st.info(f"Feature reduction: {(1 - X_train_scaled.shape[1]/X.shape[1])*100:.1f}% reduction")
                    
                    plot_metrics(metrics, y_test, y_pred)

        # Show raw data option
        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Credit Card Fraud Dataset")
            st.dataframe(pd.concat([X, y], axis=1))

        # Move About section to the bottom of sidebar
        if st.sidebar.checkbox("Show Project Information", False):
            st.markdown("""
            # Credit Card Fraud Detection Project

            📘 For detailed implementation and analysis, check out our [GitHub Jupyter Notebook](https://github.com/yourusername/fraud-detection/blob/main/analysis.ipynb)

            ## Project Overview
            This project implements custom machine learning algorithms from scratch to detect fraudulent credit card transactions. We focus on three core algorithms: SVM, KNN, and Logistic Regression, each built using fundamental mathematical principles without relying on existing ML libraries.

            ## Key Features
            - Custom implementation of ML algorithms
            - Interactive visualization of model performance
            - Real-time model parameter tuning
            - Comprehensive performance metrics

            ## Acknowledgments
            - Dataset provided by Kaggle
            - Inspired by research from IEEE papers on fraud detection
            - Special thanks to the scikit-learn documentation for algorithm references

            ## Resources
            - [Scikit-learn Documentation](https://scikit-learn.org/)
            - [Research Paper: "Credit Card Fraud Detection Using Machine Learning"](https://example.com)
            - [Mathematics of Machine Learning](https://example.com)
            - [Dataset Source](https://kaggle.com)
            """)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
