import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch

# Import everything else fro_m your existing modules
from marksix_simulation import MonteCarloSimulator
from marksix_model import MarkSixLSTM, MarkSixMLP, ModelTrainer
from marksix_analysis import (
    MarkSixAnalyzer,
    prepare_sequence_data,
    MarkSixDataset,
    DataLoader
)

def visualize_plots(analyzer: MarkSixAnalyzer):
    """
    Rather than showing each plot in a new window,
    capture them as figures and display them via Streamlit.
    """
    # First plot: Number Sum Trend
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    analyzer.features_df.plot(
        x='Draw', y='NumberSum', kind='line', ax=ax1,
        title='Number Sum Trend Over Time (期数号码和趋势)'
    )
    ax1.set_xlabel('Draw Number (期数)')
    ax1.set_ylabel('Sum of Winning Numbers (号码和)')
    st.pyplot(fig1)

    # Second figure with subplots
    fig2, (ax_box, ax_bar) = plt.subplots(1, 2, figsize=(15, 6))

    winning_cols = [f'Winning Number {i}' for i in range(1, 7)]
    analyzer.df[winning_cols].boxplot(ax=ax_box)
    ax_box.set_title('Distribution of Winning Numbers (中奖号码分布)')
    ax_box.set_xlabel('Position')
    ax_box.set_ylabel('Number Value')

    range_cols = ['1-10', '11-20', '21-30', '31-40', '41-50']
    mean_vals = analyzer.df[range_cols].mean()
    ax_bar.bar(range_cols, mean_vals)
    ax_bar.set_title('Average Frequency by Number Range (号码范围平均频率)')
    ax_bar.set_xticklabels(range_cols, rotation=45)

    fig2.tight_layout()
    st.pyplot(fig2)

def run_analysis_and_prediction():
    """
    This function encapsulates your entire analysis + model training
    + simulation logic. The results (plots & predicted numbers) will
    then be displayed in the Streamlit app.
    """
    # Initialize analyzer and simulator
    analyzer = MarkSixAnalyzer('Mark_Six.xlsx')
    simulator = MonteCarloSimulator()
    
    # Prepare and analyze data
    features, targets = analyzer.prepare_data()
    
    # Ensure features and targets are float32
    features = features.astype(np.float32)
    targets = targets.astype(np.float32)
    
    # Show a small summary
    st.write("特征摘要 (Feature Summary) - 前5行:")
    st.write(analyzer.features_df.head())
    
    # Plot some figures in Streamlit
    visualize_plots(analyzer)

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
    st.write("Training LSTM model...")
    input_size = features.shape[1]  # Number of features
    lstm_model = MarkSixLSTM(input_size=input_size)
    lstm_trainer = ModelTrainer(lstm_model)
    lstm_trainer.train(train_loader, val_loader, epochs=10)  # reduce epochs in a demo
    
    # Initialize and train MLP model
    st.write("Training MLP model...")
    mlp_model = MarkSixMLP(input_size=input_size)
    mlp_trainer = ModelTrainer(mlp_model)
    mlp_trainer.train(train_loader, val_loader, epochs=10)  # reduce epochs in a demo
    
    # Track best performing combinations
    best_combination = None
    best_mean_return = float('-inf')
    best_results = None
    
    st.write("Running simulations to find the best combination...")
    
    for i in range(20):  # reduce iteration count for a faster example
        if i % 5 == 0:
            st.write(f"Completed {i} simulations out of 20...")
        
        # Prepare the last sequence for prediction
        last_sequence = features[-sequence_length:]
        
        # Get predictions from both models
        lstm_pred = lstm_trainer.predict(torch.FloatTensor(last_sequence).unsqueeze(0))[0]
        mlp_pred = mlp_trainer.predict(torch.FloatTensor(last_sequence).unsqueeze(0))[0]
        
        # Generate numbers using both model predictions and statistical features
        latest_features = analyzer.features_df.iloc[-1]
        variation = np.random.normal(0, 2, 6)
        
        # Combine predictions
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
        
        # Select or fill up to 6 numbers
        if len(predicted_numbers) > 6:
            predicted_numbers = sorted(np.random.choice(predicted_numbers, 6, replace=False))
        else:
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
    
    # Output final best combination
    st.write("=== Best Combination Found ===")
    st.write(f"Numbers: {best_combination}")
    st.write(f"Expected Return: HKD {best_mean_return:.2f}")
    
    # Print out final stats
    st.write("Detailed Statistics for Best Combination:")
    summary_text = (
        f"Average Return: HKD {best_results['mean_return']:.2f}\n"
        f"Standard Deviation: HKD {best_results['std_return']:.2f}\n"
        f"Maximum Return: HKD {best_results['max_return']:.2f}\n"
        f"Minimum Return: HKD {best_results['min_return']:.2f}\n"
        f"Probability of Positive Return: {best_results['positive_return_prob'] * 100:.2f}%\n"
    )
    st.text(summary_text)
    
    st.write("Matches Distribution (3-6 matches):", best_results['matches_distribution'])

    # After you find best_results...
    plot_fig = simulator.plot_simulation_results(best_results)
    st.pyplot(plot_fig)

def main():
    st.title("Mark Six Analysis and Prediction")
    st.write("This Streamlit app runs an analysis, trains models, and attempts to predict Mark Six draws.")
    
    # Only run the analysis if the user clicks the button
    if st.button("Run Analysis & Prediction"):
        run_analysis_and_prediction() 