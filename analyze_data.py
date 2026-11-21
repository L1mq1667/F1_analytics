import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("charts", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")

df_res = pd.read_csv("f1_drivers_results.csv")
df_aud = pd.read_csv("f1_audience.csv")
df_2026 = df_res[df_res["season"] == 2026].copy()

top_gp = df_aud[df_aud["season"] == 2026].sort_values("circuit_weekend_attendance").tail(10)
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(top_gp["grand_prix"], top_gp["circuit_weekend_attendance"] / 1000, color="#e10600")
ax.set_title("Топ-10 самых посещаемых Гран-при Формулы-1 (2026)", fontsize=13, pad=12)
ax.set_xlabel("Посещаемость за уикенд, тыс. зрителей")
ax.set_ylabel("Этап Гран-при")
ax.set_xlim(0, top_gp["circuit_weekend_attendance"].max() / 1000 * 1.15)
for bar in bars:
    width = bar.get_width()
    ax.text(width + 8, bar.get_y() + bar.get_height() / 2, f"{width:.0f} тыс.", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("charts/top_attendance.png", dpi=150)
plt.close()

top_drivers = df_2026.groupby("driver_code")["points"].sum().nlargest(5).index
fig, ax = plt.subplots(figsize=(11, 6))
for driver in top_drivers:
    driver_data = df_2026[df_2026["driver_code"] == driver].sort_values("round")
    cum_points = driver_data["points"].cumsum()
    ax.plot(driver_data["round"], cum_points, marker="o", linewidth=2, label=driver)
ax.set_title("Динамика набора очков топ-5 пилотов по ходу сезона 2026", fontsize=13, pad=12)
ax.set_xlabel("Номер этапа (раунд)")
ax.set_ylabel("Суммарные очки")
ax.set_xticks(sorted(df_2026["round"].unique()))
ax.legend(title="Пилот")
plt.tight_layout()
plt.savefig("charts/championship_battle.png", dpi=150)
plt.close()

valid = df_2026[(df_2026["grid_position"] > 0) & (df_2026["status"] == "Finished")]
corr = valid["grid_position"].corr(valid["finish_position"])
fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(valid["grid_position"], valid["finish_position"], alpha=0.55, color="#1f77b4", edgecolors="white", s=55)
ax.plot([1, 22], [1, 22], color="#e10600", linestyle="--", label="Финиш = Старт")
ax.set_title(f"Зависимость финишной позиции от старта (2026), корреляция r = {corr:.2f}", fontsize=12, pad=12)
ax.set_xlabel("Стартовая позиция (квалификация)")
ax.set_ylabel("Финишная позиция (гонка)")
ax.set_xticks(range(1, 23))
ax.set_yticks(range(1, 23))
ax.legend()
plt.tight_layout()
plt.savefig("charts/grid_vs_finish.png", dpi=150)
plt.close()

def group_status(status):
    if status == "Finished":
        return "Финишировал в круге"
    if "Lap" in str(status):
        return "Отстал на круг+"
    return "Сход (DNF)"

status_counts = df_2026["status"].apply(group_status).value_counts()
fig, ax = plt.subplots(figsize=(8, 6))
ax.pie(
    status_counts.values,
    labels=status_counts.index,
    autopct="%1.1f%%",
    startangle=140,
    colors=["#2ecc71", "#f39c12", "#e74c3c"],
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
ax.set_title("Распределение исходов гонок для пилотов в сезоне 2026", fontsize=13, pad=12)
plt.tight_layout()
plt.savefig("charts/finish_status.png", dpi=150)
plt.close()

team_pts = df_res.groupby(["season", "team"])["points"].sum().unstack(level=0)
rounds_count = df_res.groupby("season")["round"].nunique()

team_avg = pd.DataFrame({
    2025: team_pts[2025] / rounds_count[2025],
    2026: team_pts[2026] / rounds_count[2026]
}).dropna().sort_values(2026, ascending=True)

y = np.arange(len(team_avg))
height = 0.38

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(y - height / 2, team_avg[2025], height, label="Сезон 2025", color="#95a5a6")
ax.barh(y + height / 2, team_avg[2026], height, label="Сезон 2026", color="#e10600")

ax.set_title("Сравнение команд: средние очки за этап (2025 vs 2026)", fontsize=13, pad=12)
ax.set_xlabel("Среднее количество очков за Гран-при")
ax.set_ylabel("Команда")
ax.set_yticks(y)
ax.set_yticklabels(team_avg.index)
ax.set_xlim(0, team_avg[2026].max() * 1.18)

for i, (_, row) in enumerate(team_avg.iterrows()):
    diff = row[2026] - row[2025]
    sign = "+" if diff > 0 else ""
    ax.text(row[2026] + 0.5, i + height / 2, f"{row[2026]:.1f} ({sign}{diff:.1f})", va="center", fontsize=8.5, fontweight="bold")

ax.legend()
plt.tight_layout()
plt.savefig("charts/season_comparison_2025_2026.png", dpi=150)
plt.close()

piv = df_aud.pivot_table(index="grand_prix", columns="season", values="circuit_weekend_attendance").dropna()
piv["pct_change"] = (piv[2026] - piv[2025]) / piv[2025] * 100
piv = piv.sort_values("pct_change", ascending=True)

y = np.arange(len(piv))
height = 0.38

fig, ax = plt.subplots(figsize=(11, 7))
ax.barh(y - height / 2, piv[2025] / 1000, height, label="2025 год", color="#95a5a6")
ax.barh(y + height / 2, piv[2026] / 1000, height, label="2026 год", color="#e10600")

ax.set_title("Рост посещаемости одних и тех же Гран-при (2025 vs 2026)", fontsize=13, pad=12)
ax.set_xlabel("Посещаемость за уикенд, тыс. зрителей")
ax.set_ylabel("Этап Гран-при")
ax.set_yticks(y)
ax.set_yticklabels(piv.index)
ax.set_xlim(0, piv[2026].max() / 1000 * 1.18)

for i, (_, row) in enumerate(piv.iterrows()):
    sign = "+" if row["pct_change"] > 0 else ""
    ax.text(row[2026] / 1000 + 8, i + height / 2, f"{sign}{row['pct_change']:.1f}%", va="center", fontsize=8.5, fontweight="bold")

ax.legend()
plt.tight_layout()
plt.savefig("charts/relevance_yoy.png", dpi=150)
plt.close()