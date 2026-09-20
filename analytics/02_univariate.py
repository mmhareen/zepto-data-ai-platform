import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("analytics/titanic.csv")

# Histogram + box plot for age
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["age"], bins=20, edgecolor="black")
axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age")

axes[1].boxplot(df["age"])
axes[1].set_title("Age Box Plot")

plt.tight_layout()
plt.savefig("analytics/age_distribution.png")
plt.close()

# Histogram + box plot for fare
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(df["fare"], bins=20, edgecolor="black")
axes[0].set_title("Fare Distribution")
axes[0].set_xlabel("Fare")

axes[1].boxplot(df["fare"])
axes[1].set_title("Fare Box Plot")

plt.tight_layout()
plt.savefig("analytics/fare_distribution.png")
plt.close()

print("Saved age_distribution.png and fare_distribution.png")


def count_outliers_iqr(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = series[(series < lower_bound) | (series > upper_bound)]
    return len(outliers), lower_bound, upper_bound

age_outlier_count, age_low, age_high = count_outliers_iqr(df["age"])
fare_outlier_count, fare_low, fare_high = count_outliers_iqr(df["fare"])

print(f"Age outliers: {age_outlier_count} (bounds: {age_low:.2f} to {age_high:.2f})")
print(f"Fare outliers: {fare_outlier_count} (bounds: {fare_low:.2f} to {fare_high:.2f})")


fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode()[0]

print(f"\nFare — mean: {fare_mean:.2f}, median: {fare_median:.2f}, mode: {fare_mode:.2f}")