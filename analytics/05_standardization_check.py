import pandas as pd

df = pd.read_csv("analytics/titanic.csv")

# Before: original stats
print("BEFORE standardization:")
print(f"age  -> mean: {df['age'].mean():.2f}, std: {df['age'].std():.2f}")
print(f"fare -> mean: {df['fare'].mean():.2f}, std: {df['fare'].std():.2f}")

# Apply z-score manually: z = (x - mean) / std
df["age_zscore"] = (df["age"] - df["age"].mean()) / df["age"].std()
df["fare_zscore"] = (df["fare"] - df["fare"].mean()) / df["fare"].std()

# After: standardized stats
print("\nAFTER standardization:")
print(f"age_zscore  -> mean: {df['age_zscore'].mean():.4f}, std: {df['age_zscore'].std():.4f}")
print(f"fare_zscore -> mean: {df['fare_zscore'].mean():.4f}, std: {df['fare_zscore'].std():.4f}")