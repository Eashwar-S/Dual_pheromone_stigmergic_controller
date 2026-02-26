import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_centralized(centralized_path):
    if centralized_path.exists():
        df_cent = pd.read_excel(centralized_path, sheet_name="detailed")
        # For centralized, we just take grid_size (x) and t_targets (y)
        # If there are multiple runs, we take the mean to plot a clean line
        df_cent_agg = df_cent.groupby("grid_size")["t_targets"].mean().reset_index()
        plt.plot(df_cent_agg["grid_size"], df_cent_agg["t_targets"], 
                 label="Centralized", linestyle="--", marker="o", color="blue")
    else:
        print(f"Warning: File not found {centralized_path}")

def plot_eff(search_eff_path):
    if search_eff_path.exists():
        df_eff = pd.read_excel(search_eff_path, sheet_name="detailed")
        # Calculate mean and std for error bars
        df_eff_agg = df_eff.groupby("grid_size")["t_targets"].agg(["mean", "std"]).reset_index()
        df_eff_agg["std"] = df_eff_agg["std"].fillna(0)
        plt.errorbar(df_eff_agg["grid_size"], df_eff_agg["mean"], yerr=df_eff_agg["std"], 
                     label="Stigmergy Repulsive (Our approach)", linestyle="--", marker="s", 
                     color="green", capsize=5)
    else:
        print(f"Warning: File not found {search_eff_path}")

def plot_rand(random_path):
    if random_path.exists():
        df_rand = pd.read_excel(random_path, sheet_name="detailed")
        # Calculate mean and std for error bars
        df_rand_agg = df_rand.groupby("grid_size")["t_targets"].agg(["mean", "std"]).reset_index()
        df_rand_agg["std"] = df_rand_agg["std"].fillna(0)
        plt.errorbar(df_rand_agg["grid_size"], df_rand_agg["mean"], yerr=df_rand_agg["std"], 
                     label="Stigmergy Random", linestyle="--", marker="^", 
                     color="red", capsize=5)
    else:
        print(f"Warning: File not found {random_path}")


def main():
    # Base directory for experiment results
    base_path = Path(__file__).parent.parent / "experimentation" / "experiment_results"
    
    # Paths to the specific result files
    centralized_path = base_path / "centralized" / "E1" / "results.xlsx"
    search_eff_path = base_path / "stigmergy_search_efficient" / "E1" / "results.xlsx"
    random_path = base_path / "stigmergy_random" / "E1" / "results.xlsx"

    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Centralized vs Stigmergy Repulsive (Our approach)
    plt.figure(figsize=(10, 6))
    plot_centralized(centralized_path)
    plot_eff(search_eff_path)
    
    plt.xlabel("Grid Size")
    plt.ylabel("Timesteps to Find All Targets (t_targets)")
    plt.title("Target Discovery Time vs Grid Size (E1) - Centralized vs Our Approach")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()
    
    plot1_filename = output_dir / "t_targets_vs_grid_size_E1_two.png"
    plt.savefig(plot1_filename, dpi=300)
    print(f"Plot saved successfully to {plot1_filename}")
    plt.close()

    # Plot 2: All three approaches
    plt.figure(figsize=(10, 6))
    plot_centralized(centralized_path)
    plot_eff(search_eff_path)
    plot_rand(random_path)
    
    plt.xlabel("Grid Size")
    plt.ylabel("Timesteps to Find All Targets (t_targets)")
    plt.title("Target Discovery Time vs Grid Size (E1) - All Approaches")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    plot2_filename = output_dir / "t_targets_vs_grid_size_E1_all.png"
    plt.savefig(plot2_filename, dpi=300)
    print(f"Plot saved successfully to {plot2_filename}")
    
    # Show the last plot
    plt.show()

if __name__ == "__main__":
    main()
