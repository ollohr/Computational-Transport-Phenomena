import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = r'assignments\ass_3\plot_drag-1.tsv'

# Read file while skipping the first 3 header rows
data = pd.read_csv(
    file_path,
    delim_whitespace=True,
    skiprows=3,
    names=["TimeStep", "Drag", "FlowTime"]
)

print(data.head())

# Plot Drag vs Flow Time
plt.figure()
plt.plot(data["FlowTime"], data["Drag"])
plt.xlabel("Flow Time")
plt.ylabel("Drag")
plt.title("Drag vs Flow Time")
plt.grid(True)
plt.show()