# Mark Six Lottery Prediction System

## Project Overview
This project implements an advanced prediction system for Hong Kong's Mark Six lottery using three separate Convolutional Neural Networks (CNNs). The system analyzes historical lottery data patterns and statistically relevant features to generate potential winning number combinations.

## How It Works

This prediction system leverages deep learning to analyze patterns in Mark Six historical data. The approach includes:

1. **Feature Engineering**: Extracting statistically relevant features from past draws:
   - Distribution patterns (Low/High numbers)
   - Odd/Even number balances
   - Number ranges (1-10, 11-20, 21-30, 31-40, 41-50)
   - Historical frequency patterns
   - Recency of numbers (draws since last appearance)

2. **Triple CNN Architecture**:
   - **CNN 1**: Focuses on overall pattern recognition across the entire dataset
   - **CNN 2**: Specializes in recency and frequency patterns
   - **CNN 3**: Analyzes number distribution and balance features
   
3. **Ensemble Prediction**: The three models work together to generate more robust predictions than any single model could provide.

## Technical Implementation

The prediction engine uses:
- TensorFlow/Keras for deep learning implementation
- Pandas/NumPy for data handling and preprocessing
- Ensemble techniques to combine model outputs
- Probabilistic analysis to rank potential number combinations

## Dataset Requirements

The system requires a CSV file with historical Mark Six draws, containing:
- Draw dates
- Winning numbers (6 regular numbers + 1 extra)
- Statistical features (can be generated from raw data)

## Setup and Installation

```bash
# Clone the repository
git clone [repository-url]

# Navigate to the project directory
cd marksix

# Install dependencies
pip install -r requirements.txt

# Run the prediction system
python marksixdeep.py
```

## Dependencies

- Python 3.6+
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Scikit-learn

## Usage

1. Ensure your Mark Six historical data is in a CSV file named `Mark_Six.csv`
2. Run the prediction script:
   ```
   python marksixdeep.py
   ```
3. The system will:
   - Load and process historical data
   - Train the three CNN models
   - Generate predictions for potential winning combinations
   - Output probability-ranked number combinations

## Caution

This system is designed for educational and research purposes. While it uses advanced machine learning techniques, lottery predictions remain probabilistic and cannot guarantee winnings. Always gamble responsibly.

## Contributing

Contributions to improve the prediction algorithms or feature engineering approaches are welcome. Please feel free to submit pull requests or open issues with suggestions.

## License

[license good] 