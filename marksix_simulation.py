import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass

@dataclass
class PrizeStructure:
    """Prize structure for Mark Six lottery."""
    first_prize: float = 8_000_000  # Approximate HKD for matching all 6 numbers
    second_prize: float = 1_000_000  # Match 5 numbers + extra
    third_prize: float = 50_000     # Match 5 numbers
    fourth_prize: float = 2_000     # Match 4 numbers + extra
    fifth_prize: float = 640        # Match 4 numbers
    sixth_prize: float = 320        # Match 3 numbers + extra
    seventh_prize: float = 40       # Match 3 numbers
    ticket_cost: float = 10         # Cost per bet in HKD

class MonteCarloSimulator:
    def __init__(self, prize_structure: PrizeStructure = None):
        self.prize_structure = prize_structure or PrizeStructure()
        self.results_history = []
    
    def generate_random_draw(self) -> Tuple[List[int], int]:
        """Generate a random draw of 6 numbers plus extra number."""
        numbers = np.random.choice(49, size=7, replace=False) + 1
        return sorted(numbers[:6]), numbers[6]
    
    def calculate_prize(self, 
                       predicted_numbers: List[int], 
                       actual_numbers: List[int], 
                       extra_number: int) -> float:
        """Calculate prize based on matching numbers."""
        matches = len(set(predicted_numbers) & set(actual_numbers))
        matches_extra = extra_number in predicted_numbers
        
        if matches == 6:
            return self.prize_structure.first_prize
        elif matches == 5 and matches_extra:
            return self.prize_structure.second_prize
        elif matches == 5:
            return self.prize_structure.third_prize
        elif matches == 4 and matches_extra:
            return self.prize_structure.fourth_prize
        elif matches == 4:
            return self.prize_structure.fifth_prize
        elif matches == 3 and matches_extra:
            return self.prize_structure.sixth_prize
        elif matches == 3:
            return self.prize_structure.seventh_prize
        return 0
    
    def run_simulation(self, 
                      predicted_numbers: List[int], 
                      num_simulations: int = 10000) -> Dict:
        """Run Monte Carlo simulation for betting strategy analysis."""
        returns = []
        matches_history = []
        
        for _ in range(num_simulations):
            actual_numbers, extra_number = self.generate_random_draw()
            prize = self.calculate_prize(predicted_numbers, actual_numbers, extra_number)
            net_return = prize - self.prize_structure.ticket_cost
            returns.append(net_return)
            
            matches = len(set(predicted_numbers) & set(actual_numbers))
            matches_history.append(matches)
        
        results = {
            'mean_return': np.mean(returns),
            'std_return': np.std(returns),
            'max_return': np.max(returns),
            'min_return': np.min(returns),
            'positive_return_prob': np.mean(np.array(returns) > 0),
            'matches_distribution': np.bincount(matches_history, minlength=7)[3:],  # Count matches 3-6
        }
        
        self.results_history.append(results)
        return results
    
    def plot_simulation_results(self, results: Dict):
        """Visualize simulation results and return the figure."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot 1: Distribution of Returns (matches_distribution in results_history)
        ax1.set_title('Distribution of Returns')
        ax1.hist(
            self.results_history[-1]['matches_distribution'],
            bins=4,
            label=['3 matches', '4 matches', '5 matches', '6 matches']
        )
        ax1.set_xlabel('Number of Matches')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Plot 2: Match distribution
        match_labels = ['3', '4', '5', '6']
        ax2.bar(match_labels, results['matches_distribution'])
        ax2.set_title('Distribution of Number Matches')
        ax2.set_xlabel('Number of Matches')
        ax2.set_ylabel('Frequency')
        
        fig.tight_layout()
        return fig  # <-- Return the figure object instead of calling plt.show()
    
    def print_simulation_summary(self, results: Dict):
        """Print summary statistics from simulation."""
        print("\nMonte Carlo Simulation Results:")
        print(f"Average Return: HKD {results['mean_return']:.2f}")
        print(f"Standard Deviation: HKD {results['std_return']:.2f}")
        print(f"Maximum Return: HKD {results['max_return']:.2f}")
        print(f"Minimum Return: HKD {results['min_return']:.2f}")
        print(f"Probability of Positive Return: {results['positive_return_prob']*100:.2f}%")
        
        print("\nMatching Numbers Distribution:")
        for i, count in enumerate(results['matches_distribution']):
            print(f"{i+3} matches: {count} times ({count/100:.2f}%)")

def main():
    # Example usage
    simulator = MonteCarloSimulator()
    
    # Example predicted numbers
    predicted_numbers = [1, 10, 20, 30, 40, 49]
    
    # Run simulation
    results = simulator.run_simulation(predicted_numbers)
    
    # Display results
    simulator.print_simulation_summary(results)
    plot_fig = simulator.plot_simulation_results(results)

if __name__ == "__main__":
    main() 