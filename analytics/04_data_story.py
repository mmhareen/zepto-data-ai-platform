import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("analytics/titanic.csv")

# Chart 1: Survival rate by sex (bar chart)
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="sex", y="survived", errorbar=None)
plt.title("Survival Rate by Sex")
plt.ylabel("Survival Rate")
plt.savefig("analytics/chart1_survival_by_sex.png")
plt.close()

# Chart 2: Survival rate by pclass (bar chart)
plt.figure(figsize=(5, 4))
sns.barplot(data=df, x="pclass", y="survived", errorbar=None)
plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.savefig("analytics/chart2_survival_by_pclass.png")
plt.close()

# Chart 3: Survival rate by sex AND pclass together (grouped bar chart)
plt.figure(figsize=(6, 4))
sns.barplot(data=df, x="pclass", y="survived", hue="sex", errorbar=None)
plt.title("Survival Rate by Class and Sex")
plt.ylabel("Survival Rate")
plt.savefig("analytics/chart3_survival_by_pclass_sex.png")
plt.close()

# Chart 4: Age distribution split by survival outcome (box plot)
plt.figure(figsize=(5, 4))
sns.boxplot(data=df, x="survived", y="age")
plt.title("Age Distribution by Survival Outcome")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.savefig("analytics/chart4_age_by_survival.png")
plt.close()

print("Saved 4 charts.")