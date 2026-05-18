"""Generate project-page screenshots from the adversarial-ML notebook setup."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

FIG_W_IN = 1400 / 150  # 1400 px wide at 150 DPI
SPECIES = ["setosa", "versicolor", "virginica"]
SPECIES_COLORS = {"setosa": "#1f77b4", "versicolor": "#2ca02c", "virginica": "#d62728"}
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def save(fig, name):
    out = ASSETS / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out}")


# -------------------------------------------------------------------- data
df = pd.read_csv(ROOT / "iris_extended.csv")

np.random.seed(1)
training_rows = np.random.choice(df.index, size=1000, replace=False)
test_rows = df.index.difference(training_rows)
train_df = df.loc[training_rows].reset_index(drop=True)
test_df = df.loc[test_rows].reset_index(drop=True)

# Baseline 4-feature model (sanity check the 93% figure)
clean_model = LogisticRegression(random_state=1, max_iter=1000)
clean_model.fit(train_df[FEATURES], train_df["species"])
clean_acc = accuracy_score(test_df["species"], clean_model.predict(test_df[FEATURES]))
print(f"clean 4D acc = {clean_acc:.3f}")

# Availability poisoning: 500 chaff samples near setosa, mislabeled as versicolor
np.random.seed(1)
setosa_mean = train_df.loc[train_df["species"] == "setosa", FEATURES].mean()
train_std = train_df[FEATURES].std()
n_poison = 500
poison_X = pd.DataFrame({
    col: setosa_mean[col] + 0.3 * train_std[col] * np.random.randn(n_poison)
    for col in FEATURES
})
poison_y = pd.Series(["versicolor"] * n_poison, name="species")
poison_df = pd.concat([poison_X, poison_y], axis=1)
poisoned_train = pd.concat([train_df, poison_df], ignore_index=True)

poisoned_model = LogisticRegression(random_state=1, max_iter=1000)
poisoned_model.fit(poisoned_train[FEATURES], poisoned_train["species"])
poisoned_acc = accuracy_score(test_df["species"], poisoned_model.predict(test_df[FEATURES]))
print(f"poisoned 4D acc = {poisoned_acc:.3f}")


# ---------------------------------------------- 1) decision boundary 2D
# Train two 2D models (petal_length, petal_width) so the boundary is drawable.
TWOD = ["petal_length", "petal_width"]
clean_2d = LogisticRegression(random_state=1, max_iter=1000)
clean_2d.fit(train_df[TWOD], train_df["species"])

poisoned_2d = LogisticRegression(random_state=1, max_iter=1000)
poisoned_2d.fit(poisoned_train[TWOD], poisoned_train["species"])

fig, axes = plt.subplots(1, 2, figsize=(FIG_W_IN, FIG_W_IN * 0.42))
class_to_int = {c: i for i, c in enumerate(SPECIES)}
cmap_light = plt.matplotlib.colors.ListedColormap(["#cfe2f3", "#d9ead3", "#f4cccc"])

for ax, model, title, src in [
    (axes[0], clean_2d, f"Clean model ({clean_acc * 100:.0f}%)", train_df),
    (axes[1], poisoned_2d, f"After poisoning ({poisoned_acc * 100:.1f}%)", poisoned_train),
]:
    DecisionBoundaryDisplay.from_estimator(
        model, src[TWOD], response_method="predict",
        cmap=cmap_light, alpha=0.6, ax=ax,
        xlabel="petal_length (cm)", ylabel="petal_width (cm)",
    )
    for sp in SPECIES:
        pts = src[src["species"] == sp]
        ax.scatter(pts["petal_length"], pts["petal_width"],
                   c=SPECIES_COLORS[sp], label=sp, edgecolor="white",
                   linewidth=0.4, s=22, alpha=0.85)
    ax.set_title(title)
    ax.grid(False)
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

fig.tight_layout()
save(fig, "decision-boundary-before-after.png")


# ------------------------------------------------- 2) poisoned overlay
fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_W_IN * 0.55))
muted = {"setosa": "#1f5f99", "versicolor": "#5cb85c", "virginica": "#e8a6a6"}
ax.scatter(poison_df["petal_length"], poison_df["petal_width"],
           c="#d7191c", marker="x", s=30, linewidth=1.0,
           label=f"Poisoned samples (n={n_poison})", alpha=0.45, zorder=1)
for sp in SPECIES:
    pts = train_df[train_df["species"] == sp]
    ax.scatter(pts["petal_length"], pts["petal_width"],
               c=muted[sp], label=sp.capitalize(), s=32,
               edgecolor="white", linewidth=0.5, alpha=0.95, zorder=3)
ax.set_xlabel("petal_length (cm)")
ax.set_ylabel("petal_width (cm)")
ax.set_title("Availability poisoning: chaff injection near setosa cluster")
ax.grid(False)
ax.legend(loc="upper left", fontsize=11, framealpha=0.95)
fig.tight_layout()
save(fig, "poisoned-data-overlay.png")


# --------------------------------------------- 3) confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(FIG_W_IN, FIG_W_IN * 0.42))
for ax, model, title in [
    (axes[0], clean_model, "Clean model"),
    (axes[1], poisoned_model, "After poisoning"),
]:
    preds = model.predict(test_df[FEATURES])
    cm = confusion_matrix(test_df["species"], preds, labels=SPECIES)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=SPECIES, yticklabels=SPECIES,
                cbar=False, annot_kws={"size": 13})
    ax.set_title(title)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")

fig.tight_layout()
save(fig, "confusion-matrix-before-after.png")


# --------------------------------------------- 4) extraction comparison
# Rebuild the target/shadow exactly as in the notebook for reproducibility.
target_model = LogisticRegression(random_state=1, max_iter=1000)
target_model.fit(train_df[FEATURES], train_df["species"])

np.random.seed(3)
n_q = 2000
X_queries = pd.DataFrame({
    "sepal_length": train_df["sepal_length"].mean() + train_df["sepal_length"].std() * np.random.randn(n_q),
    "sepal_width": train_df["sepal_width"].mean() + train_df["sepal_length"].std() * np.random.randn(n_q),
    "petal_length": train_df["petal_length"].mean() + train_df["sepal_length"].std() * np.random.randn(n_q),
    "petal_width": train_df["petal_width"].mean() + train_df["sepal_length"].std() * np.random.randn(n_q),
})
y_queries = pd.Series([target_model.predict(X_queries.iloc[[i]])[0] for i in range(n_q)],
                     name="species")
shadow_model = RandomForestClassifier(random_state=1)
shadow_model.fit(X_queries, y_queries)

target_preds = target_model.predict(test_df[FEATURES])
shadow_preds = shadow_model.predict(test_df[FEATURES])
agreement = accuracy_score(target_preds, shadow_preds)
print(f"target/shadow agreement = {agreement:.3f}")

target_correct = [
    int(((test_df["species"] == sp) & (target_preds == sp)).sum()) for sp in SPECIES
]
shadow_correct = [
    int(((test_df["species"] == sp) & (shadow_preds == sp)).sum()) for sp in SPECIES
]

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_W_IN * 0.5))
x = np.arange(len(SPECIES))
width = 0.36
b1 = ax.bar(x - width / 2, target_correct, width, label="Target model",
            color="#1f77b4", edgecolor="white")
b2 = ax.bar(x + width / 2, shadow_correct, width, label="Shadow model",
            color="#ff7f0e", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in SPECIES])
ax.set_ylabel("Correct predictions on test set")
ax.set_title("Model extraction: target vs shadow model predictions on unseen data")
ax.legend(loc="upper left", fontsize=11)
for bars in (b1, b2):
    for bar in bars:
        ax.annotate(f"{int(bar.get_height())}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=10)
ax.text(0.98, 0.95, f"{agreement * 100:.1f}% agreement",
        transform=ax.transAxes, ha="right", va="top", fontsize=13,
        bbox=dict(facecolor="white", edgecolor="#888", boxstyle="round,pad=0.4"))
ax.set_ylim(0, max(max(target_correct), max(shadow_correct)) * 1.18)
ax.grid(axis="y", linestyle="--", alpha=0.5)
fig.tight_layout()
save(fig, "extraction-comparison.png")
