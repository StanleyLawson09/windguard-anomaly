# Generate a monitoring dashboard
import pandas as pd
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os

os.makedirs("monitoring", exist_ok=True)

# Load the prediction logs
df = pd.read_csv(os.path.join("monitoring", "logs", "prediction.csv"), parse_dates=["timestamp"])


print(f"Total predictions: {len(df)}")
print(df["label"].value_counts())
print(f"Overall failure rate: {(df.label == 'Failure').mean() * 100:.2f}%")
print("\nProbability summary statistics:")
print(df["failure_prob"].describe())

# Failure rate rolling average over time
df["is_failure"] = (df["label"] == "Failure").astype(int)
df["rolling_rate"] = df["is_failure"].rolling(window=20).mean()

# Plot the rolling failure rate over time
fig, axes = plt.subplots(2, 1,figsize=(12, 7))

axes[0].plot(df.index, df["rolling_rate"], color="#E94560", linewidth=1.5)
axes[0].set_title("Figure 2a: Rolling Failure Rate Over Time (20-prediction window)")
axes[0].set_ylabel("Failure Rate")
axes[0].set_ylim(0, 0.5)


axes[1].hist(df["failure_prob"], bins=40, color="#0F3460", alpha=0.8)
axes[1].axvline(0.5, color="red", linestyle="--", label="Decision Threshold (0.5)")
axes[1].set_title("Figure 2b: Predicted Failure Probability Distribution")
axes[1].set_xlabel("P(failure)")
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join("monitoring", "dashboard.png"), dpi=120)
print("\nDashboard saved as monitoring/dashboard.png")