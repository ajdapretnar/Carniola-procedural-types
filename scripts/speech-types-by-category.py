import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import zscore, chi2_contingency
import seaborn as sns
import numpy as np


plt.rcParams['svg.fonttype'] = 'none'

df = pd.read_csv("data/Kranjska2.0_procedural_annotated.csv")
# Maps curia names to English
df["Curia"] = df["Curia"].map({
    "kmečka": "Rural",
    "mestna": "Urban",
    "veleposestniki": "Great Landowners",
    "trgovsko-obrtna": "Chamber of Commerce",
    "splošna": "General"
})

## CURIA

# Relationship between curia and procedural type

ct = pd.crosstab(
    df["Curia"],
    df["Procedural Type"]
)

print("Curia crosstab:")
print(ct)

plt.figure(figsize=(8,4))

sns.heatmap(
    ct,
    cmap="viridis"
)

plt.title("Distribution of procedural speech types within curia")
plt.show()


# Strength of association between curia and procedural type

chi2, p, dof, expected = chi2_contingency(ct)

n = ct.to_numpy().sum()
min_dim = min(ct.shape) - 1

cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Chi-square = {chi2:.2f}")
print(f"p-value = {p:.5f}")
print(f"Cramer's V = {cramers_v:.5f}")

## LANGUAGE

# Relationship between language and procedural type

lang = pd.crosstab(
    df["Language"],
    df["Procedural Type"]
)

print("Language crosstab:")
print(lang)

sns.heatmap(
    lang,
    annot=True,
    fmt=".0f",
    cmap="viridis"
)

plt.title("Distribution of procedural speech types within laguages")
plt.show()

# Strength of association between language and procedural type

chi2, p, dof, expected = chi2_contingency(lang)

n = lang.to_numpy().sum()
min_dim = min(lang.shape) - 1

cramers_v = np.sqrt(chi2 / (n * min_dim))

print(f"Chi-square = {chi2:.2f}")
print(f"p-value = {p:.5f}")
print(f"Cramer's V = {cramers_v:.5f}")


# Zscore vizualization of curia by year

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

ct = pd.crosstab(
    df["Year"],
    df["Curia"]
)

ct_pct = ct.apply(zscore)

print("Curia crosstab:")
print(ct_pct)

plt.figure(figsize=(4,6))
plt.rcParams['svg.fonttype'] = 'none'
sns.heatmap(ct_pct, center=0, cmap="RdBu_r")

plt.title("Z-score of procedural speech by year within curia")
plt.show()