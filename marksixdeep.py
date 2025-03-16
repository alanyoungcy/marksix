import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Load and clean data
def load_data(file_path):
    df = pd.read_csv(file_path)
    
    # Print columns to help debug
    print("Available columns in CSV file:", df.columns.tolist())
    
    # Handle possible column name differences
    numeric_columns = []
    for col in df.columns:
        if col.startswith('Num') or col.startswith('WN') or col.startswith('EN') or any(x in col for x in ['Low', 'High', 'Odd', 'Even', 'From Last']):
            numeric_columns.append(col)
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')
    
    print(f"Processed numeric columns: {numeric_columns}")
    df = df.dropna()
    return df

# Prepare features and targets
def prepare_features(df):
    # Dynamically determine which columns to use based on what's available
    
    # For features, look for statistical columns
    potential_feature_cols = ['From Last', 'Low', 'High', 'Odd', 'Even', 
                            '1-10', '11-20', '21-30', '31-40', '41-50']
    feature_cols = [col for col in potential_feature_cols if col in df.columns]
    
    # If no feature columns match exactly, try to find similar ones
    if not feature_cols:
        for col in df.columns:
            if any(keyword in col for keyword in ['Low', 'High', 'Odd', 'Even', 'Last']):
                feature_cols.append(col)
    
    # For targets, look for number columns
    number_cols = []
    # Try to find columns containing winning numbers and extra number
    for col in df.columns:
        if col.startswith('Num') or col.startswith('WN') or col.startswith('N') or col.startswith('No'):
            number_cols.append(col)
    
    # Ensure we have enough columns for prediction
    if len(feature_cols) < 1:
        raise ValueError(f"Not enough feature columns found. Available columns: {df.columns.tolist()}")
    
    if len(number_cols) < 7:  # We need at least 7 columns for 6 winning numbers + 1 extra
        # Use the first 7 numeric columns as number columns if we can't identify them clearly
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 7:
            number_cols = numeric_cols[:7]
        else:
            raise ValueError(f"Cannot identify enough number columns for targets. Available columns: {df.columns.tolist()}")
    
    print(f"Using feature columns: {feature_cols}")
    print(f"Using target columns: {number_cols}")
    
    return df[feature_cols], df[number_cols[:7]]  # Use first 7 number columns

# Build deep learning model
def build_model(input_shape, output_shape):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(output_shape)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

# Train deep learning model
def train_deep_learning_model():
    # Load and prepare data
    df = load_data('Mark_Six.csv')
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Build and train model
    model = build_model(X_train_scaled.shape[1], y_train.shape[1])
    history = model.fit(X_train_scaled, y_train, 
                        validation_data=(X_test_scaled, y_test), 
                        epochs=300, batch_size=32, verbose=1)
    
    return model, scaler, X.columns.tolist()  # Return feature column names

# Predict next draw using deep learning
def predict_next_draw_deep(model, scaler, last_draw_features):
    scaled_features = scaler.transform([last_draw_features])
    predictions = model.predict(scaled_features)[0]
    
    # Round and ensure numbers are within 1-49 range
    predictions = [max(1, min(49, round(pred))) for pred in predictions]
    
    # Ensure no duplicate numbers
    final_predictions = []
    for pred in predictions:
        while pred in final_predictions:
            pred = np.random.randint(1, 50)
        final_predictions.append(pred)
    
    # First 6 numbers sorted, extra number at the end
    return sorted(final_predictions[:6]) + [final_predictions[6]]

def main():
    # Train deep learning model
    print("Training deep learning model...")
    model, scaler, feature_names = train_deep_learning_model()
    
    print(f"\nTo predict the next draw, you need to provide values for: {feature_names}")
    print("The model is now trained. You can use: model, scaler = train_deep_learning_model()")
    
    # Example of last draw features - this should be updated based on actual column names
    # Use average values if not sure what to input
    last_draw_features = [1] * len(feature_names)  # Default to 1 for each feature
    
    # Predict next draw
    predictions = predict_next_draw_deep(model, scaler, last_draw_features)
    print("\nPredicted numbers for next draw (Deep Learning):", predictions)
    print("(Note: These predictions use placeholder feature values. Update the 'last_draw_features' with actual values for better predictions)")

if __name__ == "__main__":
    main()