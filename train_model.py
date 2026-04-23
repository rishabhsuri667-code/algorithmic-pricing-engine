import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

def train():
    print("Loading datasets...")
    try:
        listings = pd.read_csv('data/listings.csv')
        demand = pd.read_csv('data/daily_demand.csv')
    except FileNotFoundError:
        print("Data files not found. Please run data_generator.py first.")
        return

    # Merge datasets
    df = pd.merge(demand, listings, on='property_id', how='left')

    # Feature Engineering
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    
    # One-hot encode categorical variables
    categorical_cols = ['neighborhood', 'weather']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Define features and target
    target = 'booked'
    
    # Drop columns not used for prediction
    drop_cols = ['property_id', 'date', 'name', target]
    features = [col for col in df_encoded.columns if col not in drop_cols]
    
    X = df_encoded[features]
    y = df_encoded[target]

    print(f"Features used for training: {features}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Random Forest model...")
    # Use class_weight='balanced' if there is an imbalance, but we generated a balanced-ish dataset
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model and feature names
    print("Saving model to model/pricing_model.pkl...")
    os.makedirs('model', exist_ok=True)
    
    # Save the model
    with open('model/pricing_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    # Save the expected feature names to ensure the app passes them in the correct order
    with open('model/features.pkl', 'wb') as f:
        pickle.dump(features, f)
        
    print("Training complete!")

if __name__ == "__main__":
    train()
