import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("analytics/titanic.csv")

# (a) Survival rate by sex
for sex_value in df["sex"].unique():
    mask = df["sex"] == sex_value
    survival_rate = df[mask]["survived"].mean()
    print(f"Survival rate for sex={sex_value}: {survival_rate:.2%}")

print()

# (b) Survival rate by pclass
for pclass_value in sorted(df["pclass"].unique()):
    mask = df["pclass"] == pclass_value
    survival_rate = df[mask]["survived"].mean()
    print(f"Survival rate for pclass={pclass_value}: {survival_rate:.2%}")

print()

# (c) Survival rate by sex AND pclass together
for sex_value in df["sex"].unique():
    for pclass_value in sorted(df["pclass"].unique()):
        mask = (df["sex"] == sex_value) & (df["pclass"] == pclass_value)
        survival_rate = df[mask]["survived"].mean()
        print(f"Survival rate for sex={sex_value}, pclass={pclass_value}: {survival_rate:.2%}")



# Correlation matrix on exactly these 6 numeric columns
corr_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[corr_columns].corr()

print("\nCorrelation matrix:")
print(corr_matrix)

plt.figure(figsize=(7, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap (6 numeric columns)")
plt.tight_layout()
plt.savefig("analytics/correlation_heatmap.png")
plt.close()

print("Saved correlation_heatmap.png")