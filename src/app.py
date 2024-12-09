import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Import custom models
from models.LogisticRegressionScratch import LogisticRegressionScratch
from models.KNearestNeighborsScratch import KNearestNeighborsScratch
from feature_reduction.CorrelationFilter import CorrelationFilter
from sklearn.preprocessing import LabelEncoder

def load_data():
    try:
        # Load the fraud dataset
        data = pd.read_csv("data/fraud_test.csv")
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        
        # Preprocessing
# Preprocessing: Convert date/time columns to numerical format or drop them
        for column in data.columns:
            # Check if column is datetime-like
            if pd.api.types.is_datetime64_any_dtype(data[column]):
                data[column] = data[column].astype(int)  # Convert datetime to numerical (timestamp)
            elif pd.api.types.is_object_dtype(data[column]):
                # Encode categorical text columns
                encoder = LabelEncoder()
                data[column] = encoder.fit_transform(data[column])

        # Check for missing values and handle them
        data = data.fillna(0)
        
        # Split features and target
        X = data.drop('is_fraud', axis=1)
        y = data['is_fraud']
        
        return X, y
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def main():
    st.set_page_config(page_title="Fraud Detection App", layout="wide")
    
    # Create a container for the sidebar
    with st.sidebar:
        st.title("Settings")
        # Replace checkbox with button
        if st.button("📖 Project Information", use_container_width=True):
            st.session_state.show_info = True
        elif not "show_info" in st.session_state:
            st.session_state.show_info = False
    
    if st.session_state.show_info:
        st.markdown("""
        # Credit Card Fraud Detection Project

        ## Course Information
        This project was developed for MATH-557 Applied Linear Algebra under the supervision of Dr. Mohammed Alshrani.

        📘 For a detailed walkthrough of the implementation and mathematical concepts, check out our [Project Notebook](https://github.com/naifqarni/credit_card_fraud_detection/blob/main/project_notbook.ipynb)

        ## Project Overview
        This project implements custom machine learning algorithms from scratch to detect fraudulent credit card transactions. The implementation focuses on two core algorithms: Logistic Regression and K-Nearest Neighbors (KNN), built using fundamental mathematical principles and linear algebra concepts.

        ## Key Features
        - Custom implementation of machine learning algorithms:
          - Logistic Regression with gradient descent
          - K-Nearest Neighbors (KNN)
        - Advanced ML implementation:
          - XGBoost for high performance
        - Feature reduction techniques:
          - Singular Value Decomposition (SVD)
          - Correlation-based Feature Selection
        - Real-time parameter tuning:
          - Learning rate and epochs for Logistic Regression
          - Number of neighbors for KNN
          - Correlation threshold for feature selection
          - Number of components for SVD

        ## Implementation Details
        - All algorithms are implemented from scratch using NumPy
        - Feature reduction methods for dimensionality reduction
        - Interactive web interface using Streamlit
        - Comprehensive performance metrics and visualizations

        ## Acknowledgments
        - Dr. Mohammed Alshrani for course guidance and project supervision
        - Dataset provided by Kaggle
        - NumPy and Streamlit documentation

        ## Resources
        - [Course Materials MATH-557]
        - [NumPy Documentation](https://numpy.org/doc/)
        - [Streamlit Documentation](https://docs.streamlit.io)
        - [Dataset Source](https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction/code)
        """)
        
        # Add a button to go back to the main app
        if st.button("← Back to App"):
            st.session_state.show_info = False
            st.rerun()
            
        return  # Exit the function here if showing project info

    # Rest of the application code
    st.title("Credit Card Fraud Detection")
    st.markdown("Binary Classification using Custom Implementation")
    
    try:
        # Load and preprocess data
        X, y = load_data()
        if X is None or y is None:
            st.error("Failed to load data")
            return

        # First split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Balance only the training data
        fraud_mask = y_train == 1
        non_fraud_mask = y_train == 0
        
        # Get the number of fraud cases
        n_fraud = fraud_mask.sum()
        
        # Get indices of non-fraud cases and sample them
        non_fraud_indices = np.where(non_fraud_mask)[0]
        sampled_non_fraud_indices = np.random.choice(
            non_fraud_indices,
            size=n_fraud,
            replace=False
        )
        
        # Create mask for selected samples
        selected_indices = np.concatenate([
            np.where(fraud_mask)[0],
            sampled_non_fraud_indices
        ])
        
        # Balance the training data
        X_train = X_train.iloc[selected_indices]
        y_train = y_train.iloc[selected_indices]
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)


        # Add feature reduction selection before classifier selection
        st.sidebar.subheader("Feature Reduction")
        reduction_method = st.sidebar.selectbox(
            "Reduction Method",
            ("None", "SVD", "Correlation Filter")
        )

        # Apply feature reduction if selected
        if reduction_method != "None":
            with st.spinner(f'Applying {reduction_method}...'):
                if reduction_method == "SVD":
                    # Add user input for number of components
                    max_components = min(X_train_scaled.shape[1], X_train_scaled.shape[0])
                    n_components = st.sidebar.slider(
                        "Number of SVD components",
                        min_value=2,
                        max_value=max_components,
                        value=min(10, max_components),
                        help="Choose the number of components to keep after SVD reduction"
                    )
                    
                    reducer = TruncatedSVD(n_components=n_components, random_state=42)
                    X_train_reduced = reducer.fit_transform(X_train_scaled)
                    X_test_reduced = reducer.transform(X_test_scaled)
                    
                    # Display explained variance information
                    explained_var_ratio = reducer.explained_variance_ratio_
                    cumulative_var_ratio = np.cumsum(explained_var_ratio)
                    st.info(f"Reduced features from {X_train_scaled.shape[1]} to {n_components} components")
                    st.info(f"Total explained variance ratio: {cumulative_var_ratio[-1]:.2%}")
                
                elif reduction_method == "Correlation Filter":
                    # Add threshold selection slider
                    correlation_threshold = st.sidebar.slider(
                        "Correlation Threshold",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.8,
                        step=0.1,
                        help="Features with correlation above this threshold will be removed"
                    )
                    
                    # Convert DataFrame to numpy array for CorrelationFilter
                    X_train_array = X_train_scaled
                    if isinstance(X_train_scaled, pd.DataFrame):
                        X_train_array = X_train_scaled.values
                    
                    reducer = CorrelationFilter(threshold=correlation_threshold)
                    X_train_reduced = reducer.fit_transform(X_train_array)
                    X_test_reduced = reducer.transform(X_test_scaled)
                    
                    # Calculate number of features removed
                    original_features = X_train_array.shape[1]
                    reduced_features = X_train_reduced.shape[1]
                    
                    st.info(f"Reduced features from {original_features} to {reduced_features} features")
                    st.info(f"Removed {original_features - reduced_features} features due to high correlation")
                
                # Update the training and test data
                X_train_scaled = X_train_reduced
                X_test_scaled = X_test_reduced

        # Move classifier and metrics selection up
        st.sidebar.subheader("Choose Classifier")
        classifier = st.sidebar.selectbox(
            "Classifier", 
            ("Logistic Regression",
             "K-Nearest Neighbors (KNN)",
             "XGBoost")
        )

        # Model hyperparameters sections remain here...

       

        # Model training and evaluation
        if classifier == 'Logistic Regression':
            st.sidebar.subheader("Model Hyperparameters")
            learning_rate = st.sidebar.number_input(
                "Learning Rate",
                min_value=0.001,
                max_value=1.0,
                value=0.01,
                step=0.001,
                format="%.3f"
            )
            epochs = st.sidebar.number_input(
                "Number of Epochs",
                min_value=100,
                max_value=100000,
                value=1000,
                step=100
            )

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
                    
                    # Add feature reduction information
                    st.info(f"Original number of features: {X.shape[1]}")
                    if reduction_method != "None":
                        st.info(f"Number of features after {reduction_method}: {X_train_scaled.shape[1]}")
                        st.info(f"Feature reduction: {(1 - X_train_scaled.shape[1]/X.shape[1])*100:.1f}% reduction")
                    

        elif classifier == 'K-Nearest Neighbors (KNN)':
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
                    
                    # Add feature reduction information
                    st.info(f"Original number of features: {X.shape[1]}")
                    if reduction_method != "None":
                        st.info(f"Number of features after {reduction_method}: {X_train_scaled.shape[1]}")
                        st.info(f"Feature reduction: {(1 - X_train_scaled.shape[1]/X.shape[1])*100:.1f}% reduction")
                    

        elif classifier == 'XGBoost':
            st.sidebar.subheader("Model Hyperparameters")
            learning_rate = st.sidebar.number_input(
                "Learning Rate",
                min_value=0.01,
                max_value=0.3,
                value=0.1,
                step=0.01,
                format="%.2f"
            )
            max_depth = st.sidebar.number_input(
                "Max Depth",
                min_value=3,
                max_value=10,
                value=6,
                step=1
            )
            n_estimators = st.sidebar.number_input(
                "Number of Trees",
                min_value=50,
                max_value=1000,
                value=100,
                step=50
            )

            if st.sidebar.button("Classify", key="classify_xgb"):
                with st.spinner('Training XGBoost...'):
                    model = xgb.XGBClassifier(
                        learning_rate=learning_rate,
                        max_depth=max_depth,
                        n_estimators=n_estimators,
                        random_state=42
                    )
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_prob = model.predict_proba(X_test_scaled)[:, 1]
                    
                    col1, col2, col3 = st.columns(3)
                    accuracy = np.mean(y_pred == y_test)
                    with col1:
                        st.metric("Accuracy", f"{accuracy:.2f}")
                    
                    # Add feature reduction information
                    st.info(f"Original number of features: {X.shape[1]}")
                    if reduction_method != "None":
                        st.info(f"Number of features after {reduction_method}: {X_train_scaled.shape[1]}")
                        st.info(f"Feature reduction: {(1 - X_train_scaled.shape[1]/X.shape[1])*100:.1f}% reduction")
                    

        # Show raw data option
        if st.sidebar.checkbox("Show raw data", False):
            st.subheader("Credit Card Fraud Dataset")
            if reduction_method == "None":
                # Show original data
                st.dataframe(pd.concat([X, y], axis=1))
            else:
                # Show reduced data
                if reduction_method == "SVD":
                    # For SVD, use generic feature names since original features are transformed
                    reduced_df = pd.DataFrame(
                        X_train_scaled,
                        columns=[f'Component_{i+1}' for i in range(X_train_scaled.shape[1])]
                    )
                elif reduction_method == "Correlation Filter":
                    # For Correlation Filter, keep the original feature names that weren't removed
                    if isinstance(X_train_scaled, np.ndarray):
                        # Get the indices of kept features using the keep_mask
                        kept_features = X_train.columns[reducer.keep_mask]
                        reduced_df = pd.DataFrame(X_train_scaled, columns=kept_features)
                    else:
                        reduced_df = pd.DataFrame(X_train_scaled)
                
                # Add target column
                reduced_df['is_fraud'] = y_train
                
                st.write("Data after feature reduction:")
                st.dataframe(reduced_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
