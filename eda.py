from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ===============================
# PATH CONFIGURATION (GitHub Safe)
# ===============================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "processed" / "clean_workers_data.csv"
OUTPUT_DIR = BASE_DIR / "data" / "EDA_charts"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(INPUT_FILE)

# ===============================
# FEATURE ENGINEERING FOR EDA
# ===============================
df["Total_Workers"] = df["Main_Total"] + df["Marginal_Total"]

df["Rural_Workers"] = df["Main_Rural_Total"] + df["Marginal_Rural_Total"]
df["Urban_Workers"] = df["Main_Urban_Total"] + df["Marginal_Urban_Total"]

df["Male_Workers"] = df["Main_Male"] + df["Marginal_Male"]
df["Female_Workers"] = df["Main_Female"] + df["Marginal_Female"]

# ===============================
# 1️⃣ Industry-wise Total Workers
# ===============================
industry_total = (
    df.groupby("Industry_Name")["Total_Workers"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(12, 6))
sns.barplot(
    x=industry_total.values,
    y=industry_total.index,
    palette="viridis"
)
plt.title("Total Workers by Industry")
plt.xlabel("Number of Workers")
plt.ylabel("Industry")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "industry_total_workers.png")
plt.close()

# ===============================
# 2️⃣ Gender Distribution (Overall)
# ===============================
gender_total = pd.Series({
    "Male": df["Male_Workers"].sum(),
    "Female": df["Female_Workers"].sum()
})

plt.figure(figsize=(6, 6))
gender_total.plot.pie(
    autopct="%1.1f%%",
    colors=["#4c72b0", "#55a868"],
    startangle=90
)
plt.title("Overall Gender Distribution")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "gender_distribution.png")
plt.close()

# ===============================
# 3️⃣ Rural vs Urban Workers
# ===============================
rural_urban_sum = pd.Series({
    "Rural": df["Rural_Workers"].sum(),
    "Urban": df["Urban_Workers"].sum()
})

plt.figure(figsize=(6, 6))
rural_urban_sum.plot.pie(
    autopct="%1.1f%%",
    colors=["#ff9999", "#66b3ff"],
    startangle=90
)
plt.title("Rural vs Urban Workers")
plt.ylabel("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "rural_urban_workers.png")
plt.close()

# ===============================
# 4️⃣ Top 5 States by Workforce
# ===============================
state_total = (
    df.groupby("State_Name")["Total_Workers"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=state_total.index,
    y=state_total.values,
    palette="magma"
)
plt.title("Top 5 States by Total Workforce")
plt.xlabel("State")
plt.ylabel("Number of Workers")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top5_states.png")
plt.close()

# ===============================
# 5️⃣ Top 5 Industries by Workforce
# ===============================
top5_industry = industry_total.head(5)

plt.figure(figsize=(8, 5))
sns.barplot(
    x=top5_industry.values,
    y=top5_industry.index,
    palette="coolwarm"
)
plt.title("Top 5 Industries by Workforce")
plt.xlabel("Number of Workers")
plt.ylabel("Industry")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "top5_industries.png")
plt.close()

print("✅ EDA completed successfully.")
print(f"📁 Charts saved in: {OUTPUT_DIR}")
