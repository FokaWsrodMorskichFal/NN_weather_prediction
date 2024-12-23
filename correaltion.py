import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(df, output_file="correlation_heatmap.png"):
    # Ensure all columns are numeric
    df = df.astype(float)

    # Calculate Pearson correlation matrix
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))

    # Create a heatmap using seaborn
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True,
        square=True
    )
    plt.title("Feature Correlation Heatmap", fontsize=16)

    # Save the heatmap as an image file
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to {output_file}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the DataFrame
    df = pd.read_csv("./proc_data/concat_clean_data_simulate_middle_day.csv", delimiter='\t')
    
    # Generate the heatmap
    plot_correlation_heatmap(df)
