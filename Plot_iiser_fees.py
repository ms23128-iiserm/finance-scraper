import pandas as pd
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Configuration ---
DATA_FILE = 'iiser_fees.csv'
# ---------------------

def load_data(filepath):
    """Loads the fee data."""
    print(f"--- Loading Data from '{filepath}' ---")
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"❌ ERROR: '{filepath}' is empty.")
            return None
        print("✅ Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: File not found: '{filepath}'")
        return None

def plot_data(df):
    """
    Plots the historical fee and inflation data on two separate y-axes.
    """
    print(f"--- Plotting historical data... ---")
    
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- Plot 1: Tuition Fee ---
    color = 'tab:blue'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Tuition Fee (INR)', color=color)
    ax1.plot(df['Year'], df['Tuition_Fee'], color=color, marker='o', label='Tuition Fee')
    ax1.tick_params(axis='y', labelcolor=color)
    
    # --- Plot 2: Inflation (CPI) ---
    ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('India CPI (Annual %)', color=color)
    ax2.plot(df['Year'], df['India_CPI'], color=color, marker='s', linestyle='--', label='India CPI')
    ax2.tick_params(axis='y', labelcolor=color)

    # --- Final Plot ---
    plt.title('IISER Mohali Tuition Fee vs. India Inflation Rate (2016-2024)')
    fig.tight_layout()  # Adjust plot to prevent y-label overlap
    
    # Add legends (we need to get labels from both axes)
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.grid(True)
    
    output_file = 'historical_fee_vs_cpi_plot.png'
    plt.savefig(output_file)
    print(f"✅ Graph saved to '{output_file}'")

def main():
    df = load_data(DATA_FILE)
    if df is not None:
        plot_data(df)

if __name__ == "__main__":
    main()

