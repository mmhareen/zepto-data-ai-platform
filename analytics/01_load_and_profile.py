import seaborn as sns

df = sns.load_dataset("titanic")

print(df.shape)
print(df.info())
print(df.describe())


missing_pct = df.isnull().mean() * 100
missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
print("\nMissing value percentages:")
print(missing_pct)


# Drop rows with missing embarked/embark_town (< 5% missing)
df = df.dropna(subset=["embarked", "embark_town"])

# Impute age with median (5-30% missing)
df["age"] = df["age"].fillna(df["age"].median())

# Drop deck entirely (>30% missing, unreliable to impute)
df = df.drop(columns=["deck"])

print("\nShape after cleaning:", df.shape)
print("\nRemaining missing values:\n", df.isnull().sum().sum())


df.to_csv("analytics/titanic.csv", index=False)
print("\nSaved cleaned data to analytics/titanic.csv")