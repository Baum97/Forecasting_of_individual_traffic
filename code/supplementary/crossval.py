import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

CSV = r"d:/Projects/Forecasting_of_individual_traffic/predictions/cross_validation/all_sources_T28-06-2026/cross_validation_matrix.csv"
OUT = r"d:/Projects/Forecasting_of_individual_traffic/thesis/images/cv_f1_5x5_lgbm.png"
ORDER = ["emobpy", "real_world_ev", "ved", "routine", "yjmob"]

df = pd.read_csv(CSV)
df = df[df["algo"] == "lgbm"]
mat = (df.pivot(index="model_trained_on", columns="evaluated_on", values="f1")
         .reindex(index=ORDER, columns=ORDER))

fig, ax = plt.subplots(figsize=(5.6, 4.5))
im = ax.imshow(mat.values, cmap="viridis", vmin=0.0, vmax=1.0)

ax.set_xticks(range(len(ORDER)), ORDER, rotation=30, ha="right")
ax.set_yticks(range(len(ORDER)), ORDER)
ax.set_xlabel("getestet auf")
ax.set_ylabel("trainiert auf")

for i in range(len(ORDER)):
    for j in range(len(ORDER)):
        v = mat.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="white" if v < 0.5 else "black")

cb = fig.colorbar(im, ax=ax)
cb.set_label("F1")
fig.tight_layout()
fig.savefig(OUT, dpi=200)
print("written:", OUT)
print(mat.round(2))