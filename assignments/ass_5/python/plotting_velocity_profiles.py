import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# File paths
file1 = Path("assignments/ass_5/python/velocity_profile_data/k_epsilon/x1.csv")
file2 = Path("assignments/ass_5/python/velocity_profile_data/k_omega/x1.csv")
file_paths = [file1, file2]

filex2_1 = Path("assignments/ass_5/python/velocity_profile_data/k_epsilon/x2.csv")
filex2_2 = Path("assignments/ass_5/python/velocity_profile_data/k_omega/x2.csv")
file_paths2 = [filex2_1, filex2_2]

filex3_1 = Path("assignments/ass_5/python/velocity_profile_data/k_epsilon/x3.csv") 
filex3_2 = Path("assignments/ass_5/python/velocity_profile_data/k_omega/x3.csv")
file_paths3 = [filex3_1, filex3_2]

filex4_1 = Path("assignments/ass_5/python/velocity_profile_data/k_epsilon/x4.csv")
filex4_2 = Path("assignments/ass_5/python/velocity_profile_data/k_omega/x4.csv")
file_paths4 = [filex4_1, filex4_2]


def plot_offset_velocity_profile(x, file_paths, U_b, scale=10):

    plt.figure(figsize=(9, 6))

    colors = ["#1f77b4", "#d62728"]
    markers = ["o", "s"]
    labels = ["k-ε model", "k-ω model"]

    for i, file_path in enumerate(file_paths):

        data = pd.read_csv(file_path, skiprows=4)
        data.columns = data.columns.str.strip()

        u = data["Velocity [ m s^-1 ]"]
        y = data["Y [ m ]"]

        H = y.max() - y.min()

        y_non_dim = y / H
        u_non_dim = u / U_b

        x_offset = (x / H) + scale * u_non_dim

        plt.plot(
            x_offset,
            y_non_dim,
            color=colors[i],
            linewidth=2.5,
            marker=markers[i],
            markersize=4,
            markevery=4,
            label=labels[i]
        )

    plt.xlabel(r"$x/H + {}\,u/U_b$".format(scale), fontsize=12, weight="bold")
    plt.ylabel(r"$y/H$", fontsize=12, weight="bold")
    plt.title(f"Offset Mean Velocity Profiles at x = {x}", fontsize=14, weight="bold")

    plt.legend(
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=10
    )

    plt.minorticks_on()

    plt.grid(which="major", linestyle="--", linewidth=0.8)
    plt.grid(which="minor", linestyle=":", linewidth=0.5)

    plt.tick_params(axis="both", which="major", labelsize=10)

    plt.tight_layout()
    plt.show()


plot_offset_velocity_profile(x=5.5, file_paths=file_paths, U_b=1.0, scale=10)
plot_offset_velocity_profile(x=10, file_paths=file_paths2, U_b=1.0, scale=10)
plot_offset_velocity_profile(x=20, file_paths=file_paths2, U_b=1.0, scale=10)
plot_offset_velocity_profile(x=30, file_paths=file_paths2, U_b=1.0, scale=10)