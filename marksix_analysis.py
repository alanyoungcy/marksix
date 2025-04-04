import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from typing import List, Tuple, Dict
from marksix_simulation import MonteCarloSimulator
from marksix_model import MarkSixLSTM, MarkSixMLP, ModelTrainer
import warnings
warnings.filterwarnings('ignore')

class MarkSixDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        # Convert to float32 for better compatibility
        self.features = torch.FloatTensor(features.astype(np.float32))
        self.targets = torch.FloatTensor(targets.astype(np.float32))
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

class FeatureExtractor:
    def __init__(self):
        self.scaler = MinMaxScaler()
    
    def count_consecutive_numbers(self, numbers: List[int]) -> int:
        """Count pairs of consecutive numbers in sorted list."""
        consecutive_count = 0
        sorted_numbers = sorted(numbers)
        for i in range(len(sorted_numbers) - 1):
            if sorted_numbers[i + 1] - sorted_numbers[i] == 1:
                consecutive_count += 1
        return consecutive_count
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all required features from the raw data."""
        features = pd.DataFrame()
        
        # Basic features directly from data
        features['Draw'] = df['Draw']  # 期数
        
        # Convert date to numerical features
        features['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
        features['DayOfMonth'] = pd.to_datetime(df['Date']).dt.day
        features['Month'] = pd.to_datetime(df['Date']).dt.month
        
        # Get winning numbers (已经排序的搅出号码)
        winning_numbers = df[['Winning Number 1', 'Winning Number 2', 'Winning Number 3', 
                            'Winning Number 4', 'Winning Number 5', 'Winning Number 6']]
        
        # Use existing features from Excel
        features['OddCount'] = df['Odd']  # 单数号码总数
        features['EvenCount'] = df['Even']  # 偶数号码总数
        features['MinNumber'] = df['Low']  # 最小值
        features['MaxNumber'] = df['High']  # 最大值
        
        # Calculate sum and standard deviation
        features['NumberSum'] = winning_numbers.sum(axis=1)
        features['NumberStd'] = winning_numbers.std(axis=1)
        features['NumberRange'] = features['MaxNumber'] - features['MinNumber']
        
        # Calculate consecutive numbers
        features['ConsecutiveCount'] = winning_numbers.apply(
            lambda x: self.count_consecutive_numbers(x), axis=1
        )
        
        # Number range distributions (号码出现频次)
        features['Range1_10'] = df['1-10']
        features['Range11_20'] = df['11-20']
        features['Range21_30'] = df['21-30']
        features['Range31_40'] = df['31-40']
        features['Range41_50'] = df['41-50']
        
        # Add supplementary number as feature (特别号码)
        features['SupplementaryNumber'] = df['Supplementary Number']
        
        # Add from last number as feature (上次搅出的号码)
        features['FromLast'] = df['From Last']
        
        # Calculate additional features
        features['MeanNumber'] = winning_numbers.mean(axis=1)
        features['MedianNumber'] = winning_numbers.median(axis=1)
        
        # Calculate gaps between numbers
        gaps = []
        for i in range(len(winning_numbers.columns)-1):
            gap = winning_numbers.iloc[:, i+1] - winning_numbers.iloc[:, i]
            gaps.append(gap)
        features['MeanGap'] = pd.concat(gaps, axis=1).mean(axis=1)
        features['MaxGap'] = pd.concat(gaps, axis=1).max(axis=1)
        
        return features

class MarkSixAnalyzer:
    def __init__(self, excel_path: str):
        # Read Excel with correct column names
        self.df = pd.read_excel(excel_path)
        
        # Rename columns to match the structure
        winning_number_cols = {
            str(i): f'Winning Number {i}' 
            for i in range(1, 7)
        }
        self.df = self.df.rename(columns=winning_number_cols)
        
        self.feature_extractor = FeatureExtractor()
        self.features_df = None
        self.scaler = MinMaxScaler()
    
    def prepare_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and targets for model training."""
        self.features_df = self.feature_extractor.extract_features(self.df)
        
        # Clean data: remove any non-numeric characters and convert to float
        def clean_numeric(x):
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, str):
                # Remove any non-numeric characters except decimal point
                cleaned = ''.join(c for c in x if c.isdigit() or c == '.')
                return float(cleaned) if cleaned else 0.0
            return 0.0
        
        # Apply cleaning to all numeric columns
        numeric_features = self.features_df.select_dtypes(include=[np.number, object]).columns
        for col in numeric_features:
            self.features_df[col] = self.features_df[col].apply(clean_numeric)
        
        # Scale numerical features
        self.features_df[numeric_features] = self.scaler.fit_transform(self.features_df[numeric_features])
        
        # Prepare targets (next draw's numbers)
        # Clean winning numbers first
        winning_cols = [f'Winning Number {i}' for i in range(1, 7)]
        for col in winning_cols:
            self.df[col] = self.df[col].apply(clean_numeric)
        
        targets = self.df[winning_cols].values[1:]
        features = self.features_df.values[:-1]  # Remove last row as we don't have next draw for it
        
        return features, targets
    
    def visualize_number_sum_trend(self):
        """Visualize the trend of number sums over time."""
        plt.figure(figsize=(15, 6))
        sns.lineplot(data=self.features_df, x='Draw', y='NumberSum')
        plt.title('Number Sum Trend Over Time (期数号码和趋势)')
        plt.xlabel('Draw Number (期数)')
        plt.ylabel('Sum of Winning Numbers (号码和)')
        plt.show()
        
        # Additional visualization for number distribution
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        winning_cols = [f'Winning Number {i}' for i in range(1, 7)]
        sns.boxplot(data=self.df[winning_cols])
        plt.title('Distribution of Winning Numbers (中奖号码分布)')
        plt.xlabel('Position')
        plt.ylabel('Number Value')
        
        plt.subplot(1, 2, 2)
        range_cols = ['1-10', '11-20', '21-30', '31-40', '41-50']
        sns.barplot(x=range_cols, y=self.df[range_cols].mean())
        plt.title('Average Frequency by Number Range (号码范围平均频率)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def print_feature_summary(self):
        """Print summary of extracted features."""
        print("\n特征摘要 (Feature Summary) - 前5行:")
        print(self.features_df.head())
        
        print("\n特征统计 (Feature Statistics):")
        print(self.features_df.describe())
        
        print("\n号码范围分布 (Number Range Distribution):")
        range_cols = ['Range1_10', 'Range11_20', 'Range21_30', 'Range31_40', 'Range41_50']
        print(self.features_df[range_cols].describe())

def prepare_sequence_data(features: np.ndarray, targets: np.ndarray, sequence_length: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare sequential data for LSTM."""
    X, y = [], []
    for i in range(len(features) - sequence_length):
        X.append(features[i:i + sequence_length])
        y.append(targets[i + sequence_length])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def main():
    # Initialize analyzer and simulator
    analyzer = MarkSixAnalyzer('Mark_Six.xlsx')
    simulator = MonteCarloSimulator()
    
    # Prepare and analyze data
    features, targets = analyzer.prepare_data()
    
    # Ensure features and targets are float32
    features = features.astype(np.float32)
    targets = targets.astype(np.float32)
    
    # Print feature summary
    analyzer.print_feature_summary()
    
    # Prepare data for models
    sequence_length = 5
    X_seq, y_seq = prepare_sequence_data(features, targets, sequence_length)
    
    # Split data into train and validation sets
    train_size = int(0.8 * len(X_seq))
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]
    
    # Create data loaders
    train_dataset = MarkSixDataset(X_train, y_train)
    val_dataset = MarkSixDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Initialize and train LSTM model
    print("\nTraining LSTM model...")
    input_size = features.shape[1]  # Number of features
    lstm_model = MarkSixLSTM(input_size=input_size)
    lstm_trainer = ModelTrainer(lstm_model)
    lstm_trainer.train(train_loader, val_loader, epochs=50)
    
    # Initialize and train MLP model
    print("\nTraining MLP model...")
    mlp_model = MarkSixMLP(input_size=input_size)
    mlp_trainer = ModelTrainer(mlp_model)
    mlp_trainer.train(train_loader, val_loader, epochs=50)
    
    # Track best performing combinations
    best_combination = None
    best_mean_return = float('-inf')
    best_results = None
    
    print("\nRunning simulations to find the best combination...")
    
    for i in range(10000):
        if i % 10 == 0:  # Progress indicator
            print(f"Completed {i} simulations...")
        
        # Prepare the last sequence for prediction
        last_sequence = features[-sequence_length:]
        
        # Get predictions from both models
        lstm_pred = lstm_trainer.predict(torch.FloatTensor(last_sequence).unsqueeze(0))[0]
        mlp_pred = mlp_trainer.predict(torch.FloatTensor(last_sequence).unsqueeze(0))[0]
        
        # Generate numbers using both model predictions and statistical features
        latest_features = analyzer.features_df.iloc[-1]
        variation = np.random.normal(0, 2, 6)
        
        # Combine predictions from all methods
        predicted_numbers = list(set(
            list(lstm_pred) +
            list(mlp_pred) +
            [
                max(1, min(49, round(latest_features['MinNumber'] + variation[0]))),
                max(1, min(49, round(latest_features['MeanNumber'] + variation[1]))),
                max(1, min(49, round(latest_features['MedianNumber'] + variation[2]))),
                max(1, min(49, round(latest_features['MaxNumber'] + variation[3]))),
                max(1, min(49, round((latest_features['MinNumber'] + latest_features['MaxNumber'])/2 + variation[4]))),
                max(1, min(49, round(latest_features['NumberSum']/6 + variation[5])))
            ]
        ))
        
        # Select 6 numbers randomly from the combined predictions
        if len(predicted_numbers) > 6:
            predicted_numbers = sorted(np.random.choice(predicted_numbers, 6, replace=False))
        else:
            # Fill up to 6 numbers if we don't have enough
            while len(predicted_numbers) < 6:
                new_num = np.random.randint(1, 50)
                if new_num not in predicted_numbers:
                    predicted_numbers.append(new_num)
            predicted_numbers = sorted(predicted_numbers)
        
        # Run simulation
        results = simulator.run_simulation(predicted_numbers, num_simulations=10000)
        
        # Track if this is the best combination so far
        if results['mean_return'] > best_mean_return:
            best_mean_return = results['mean_return']
            best_combination = predicted_numbers
            best_results = results
    
    print("\n=== Best Combination Found ===")
    print(f"Numbers: {best_combination}")
    print(f"Expected Return: HKD {best_mean_return:.2f}")
    print("\nDetailed Statistics for Best Combination:")
    simulator.print_simulation_summary(best_results)
    
    print("\n数据准备完成 (Data preparation completed)")
    print(f"特征形状 (Features shape): {features.shape}")
    print(f"目标形状 (Targets shape): {targets.shape}")

if __name__ == "__main__":
    main()