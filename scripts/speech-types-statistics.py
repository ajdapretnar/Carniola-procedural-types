import pandas as pd
import matplotlib.pyplot as plt


plt.rcParams['svg.fonttype'] = 'none'

df_all = pd.read_csv("data/Kranjska2.0_regular_MPs.csv", parse_dates=["Date"])
df_procedural = pd.read_csv("data/Kranjska2.0_procedural_annotated.csv", parse_dates=["Date"])

# Data exploration

print("Number of speeches by regular MPs:", len(df_all))
print("Number of procedural speeches:", len(df_procedural))
print("Types of procedural speeches:")
print(df_procedural["Procedural Type"].value_counts())

# Procedural weight by year

df_all["Year"] = pd.to_datetime(df_all["Date"]).dt.year
mean_proc = df_all.groupby("Year")["Procedural"].mean()
print("Mean procedural weight by year:", mean_proc)

# Procedural type shares by year

df_procedural["Year"] = pd.to_datetime(df_procedural["Date"]).dt.year

type_shares = pd.crosstab(
    df_procedural["Year"],
    df_procedural["Procedural Type"],
    normalize="index"
)

print("Procedural type shares by year:")
print(type_shares)

# Visualize procedural speech types over time

type_shares = type_shares.reindex(mean_proc.index)
area_data = type_shares.mul(mean_proc,axis=0)

ax = area_data.plot(
    kind="area",
    stacked=True,
    figsize=(12, 6)
)

plt.ylabel("Mean procedural weight")
plt.xlabel("Year")
plt.title("Procedural speech types over time")

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()
plt.show()