import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load and prepare the data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def prepare_features(df):
    # Use only winning numbers as features
    feature_cols = ['WN1', 'WN2', 'WN3', 'WN4', 'WN5', 'WN6', 'EN']
    
    # No need for additional feature engineering
    return df, feature_cols

def train_models():
    # Load and prepare data
    df = load_data('Mark_Six.csv')
    df, feature_cols = prepare_features(df)
    
    # Create 7 different models (6 winning numbers + 1 extra number)
    models = []
    for i in range(1, 7):  # First 6 models for winning numbers
        X = df[feature_cols]
        y = df[f'WN{i}']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        models.append((model, scaler))
        
        # Print model performance
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        print(f"Model {i} - Training R² Score: {train_score:.4f}, Test R² Score: {test_score:.4f}")
    
    # Add model for extra number (EN)
    X = df[feature_cols]
    y = df['EN']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    models.append((model, scaler))
    
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    print(f"Extra Number Model - Training R² Score: {train_score:.4f}, Test R² Score: {test_score:.4f}")
    
    return models, feature_cols

def predict_next_draw(models, feature_cols, last_draw_features):
    predictions = []
    
    # Scale and predict for each model (including extra number)
    for model, scaler in models:
        scaled_features = scaler.transform([last_draw_features])
        pred = model.predict(scaled_features)[0]
        # Round to nearest integer and ensure it's within 1-49 range
        pred = max(1, min(49, round(pred)))
        predictions.append(pred)
    
    # Ensure no duplicate numbers
    final_predictions = []
    for pred in predictions:
        while pred in final_predictions:
            pred = np.random.randint(1, 50)
        final_predictions.append(pred)
    
    # First 6 numbers should be sorted, but keep the extra number at the end
    return sorted(final_predictions[:6]) + [final_predictions[6]]

def main():
    # Train models
    models, feature_cols = train_models()
    
    # Example of last draw features (WN1-WN6, EN)
    last_draw_features = [8,13,18,23,31,47,16]  # Example values
    
    # Predict next draw
    predictions = predict_next_draw(models, feature_cols, last_draw_features)
    print("\nPredicted numbers for next draw:", predictions)

if __name__ == "__main__":
    main()



