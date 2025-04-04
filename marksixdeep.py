# Please be aware, this code is not working. I am still working on it.
# THis is already destroyed by cursor, the other part of the code should work
# still it is worth to observer the logic and i keep it here for future reference.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.data_processor import DataProcessor
from src.feature_engineering import FeatureEngineer
from src.model import MarkSixDataset, ModelTrainer
from src.monte_carlo import MonteCarloSimulator
from src.visualizer import Visualizer

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
    # 1. 数据加载和预处理
    data_processor = DataProcessor('data/Mark_Six.csv')
    df = data_processor.load_data()
    
    # 2. 特征工程
    feature_engineer = FeatureEngineer()
    df_with_features = feature_engineer.process_dataframe(df)
    
    # 打印前5行特征
    print("特征提取结果前5行：")
    print(df_with_features[['odd_count', 'even_count', 'numbers_sum', 
                           'numbers_std', 'consecutive_count']].head())
    
    # 3. 准备训练数据
    train_df, test_df = data_processor.split_train_test()
    
    feature_columns = ['odd_count', 'even_count', 'numbers_sum', 
                      'numbers_std', 'consecutive_count']
    target_columns = [f'Winning No. {i}' for i in range(1, 7)]
    
    X_train = train_df[feature_columns].values
    y_train = train_df[target_columns].values
    
    # 4. 训练模型
    train_dataset = MarkSixDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    trainer = ModelTrainer(input_size=len(feature_columns))
    trainer.train(train_loader)
    
    # 5. 蒙特卡洛模拟
    simulator = MonteCarloSimulator(trainer.model)
    mean_return, std_return = simulator.run_simulation(
        torch.FloatTensor(X_train[-1:])  # 使用最后一期数据进行预测
    )
    
    print(f"\n蒙特卡洛模拟结果：")
    print(f"平均收益: {mean_return:.2f} HKD")
    print(f"收益标准差: {std_return:.2f} HKD")
    
    # 6. 可视化
    visualizer = Visualizer()
    visualizer.plot_numbers_sum_trend(df_with_features)
    visualizer.plot_feature_distributions(df_with_features)

if __name__ == "__main__":
    main()