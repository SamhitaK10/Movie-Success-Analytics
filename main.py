# Movie Success Analytics — compact, portfolio-ready (pandas + 1 SQL query)
import os, sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", None)

CSV = "rotten_tomatoes_movies.csv"
assert os.path.exists(CSV), "Put rotten_tomatoes_movies.csv next to this file."

# ----------------------------
# 1) LOAD + LIGHT NORMALIZATION
# ----------------------------
df = pd.read_csv(CSV)

# normalize common column names gracefully
rename = {
    "title": "movie_title",
    "movieTitle": "movie_title",
    "tomatometer_rating": "critic_rating",
    "tomatometer_score": "critic_rating",
    "audience_score": "audience_rating",
    "audience_rating (%)": "audience_rating",
    "domestic_box_office": "box_office",
    "box_office_gross_usd": "box_office",
    "genres": "genres",
}
for k, v in rename.items():
    if k in df.columns and v not in df.columns:
        df.rename(columns={k: v}, inplace=True)

# ensure numeric
for c in ["critic_rating", "audience_rating", "box_office", "runtime", "year"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# normalize audience to 0–100 if file uses 0–5
if "audience_rating" in df.columns:
    max_val = df["audience_rating"].max(skipna=True)
    if pd.notna(max_val) and max_val <= 5:
        df["audience_rating"] = df["audience_rating"] * 20.0

# standardize a "ratings count" column if present
possible_count_cols = ["audience_count", "audience_num_ratings", "audience_ratings", "audience_rating_count"]
count_col = next((c for c in possible_count_cols if c in df.columns), None)
if count_col:
    df[count_col] = pd.to_numeric(df[count_col], errors="coerce")
else:
    df["aud_count_proxy"] = np.nan  # keeps code simple when count missing
    count_col = "aud_count_proxy"

# minimal required columns
need = {"movie_title", "critic_rating", "audience_rating"}
missing = [c for c in need if c not in df.columns]
assert not missing, f"Missing required columns: {missing}"

# drop rows without ratings
df = df.dropna(subset=["critic_rating", "audience_rating"]).copy()

# primary genre (first token)
if "genres" in df.columns:
    df["genres"] = df["genres"].fillna("")
    df["primary_genre"] = (
        df["genres"].astype(str).str.split(",").str[0].str.strip().replace({"": "Unknown"})
    )
else:
    df["primary_genre"] = "Unknown"

print("Loaded:", df.shape)
print(df[["movie_title", "primary_genre", "critic_rating", "audience_rating"]].head())

# ----------------------------
# 2) SIMPLE ANALYSIS (pandas)
# ----------------------------
# Top 10 by audience rating with a minimum votes threshold if we have one
MIN_VOTES = 500
if df[count_col].notna().any():
    df_top_candidates = df[(df["audience_rating"].notna()) & (df[count_col] >= MIN_VOTES)]
else:
    df_top_candidates = df[df["audience_rating"].notna()]  # no votes column available

top_movies = df_top_candidates.nlargest(10, ["audience_rating", count_col])[["movie_title", "audience_rating"]]

# Genre-level audience – critic gap (only if enough samples per genre)
genre_counts = df["primary_genre"].value_counts()
genre_gap = (
    df.groupby("primary_genre")[["audience_rating", "critic_rating"]]
      .mean()
      .assign(avg_gap=lambda t: t["audience_rating"] - t["critic_rating"])
      .loc[genre_counts[genre_counts >= 20].index]
      .sort_values("avg_gap", ascending=False)
)

# Most “controversial” = largest absolute disagreement
df["gap_abs"] = (df["audience_rating"] - df["critic_rating"]).abs()
controversial = df.nlargest(15, "gap_abs")[["movie_title", "audience_rating", "critic_rating", "gap_abs"]]

print("\nTop movies (audience, filtered):\n", top_movies.head(10))
print("\nGenres with largest audience-over-critic gap:\n", genre_gap.head(10))
print("\nMost controversial (biggest disagreement):\n", controversial.head(10))

# ----------------------------
# 3) VISUALS → saved PNGs
# ----------------------------
plt.figure(figsize=(7,5))
plt.hist(df["audience_rating"].dropna(), bins=20, range=(0,100))
plt.title("Audience Rating Distribution"); plt.xlabel("Audience Rating"); plt.ylabel("# Movies"); plt.grid(True)
plt.tight_layout(); plt.savefig("audience_hist.png"); plt.close()

plt.figure(figsize=(7,5))
plt.scatter(df["critic_rating"], df["audience_rating"], alpha=0.5)
plt.title("Audience vs Critic Ratings"); plt.xlabel("Critic"); plt.ylabel("Audience")
plt.xlim(0,100); plt.ylim(0,100); plt.grid(True)
plt.tight_layout(); plt.savefig("aud_vs_crit_scatter.png"); plt.close()

print("\nSaved plots: audience_hist.png, aud_vs_crit_scatter.png")

# ----------------------------
# 4) TINY ML: predict if audience “liked” (>=70)
# ----------------------------
df_ml = df.copy()
df_ml["liked"] = (df_ml["audience_rating"] >= 70).astype(int)

# numeric features (use what exists)
num_feats = [c for c in ["critic_rating", "runtime", "year", "box_office"] if c in df_ml.columns]
# engineered feature: # of genres
if "genres" in df_ml.columns:
    df_ml["genre_count"] = df_ml["genres"].fillna("").apply(lambda s: 0 if s == "" else len([x for x in s.split(",") if x.strip()]))
    num_feats.append("genre_count")

cat_feats = ["primary_genre"] if "primary_genre" in df_ml.columns else []

X = df_ml[num_feats + cat_feats].copy()
y = df_ml["liked"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_feats),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_feats),
    ],
    remainder="drop"
)

clf = Pipeline([("prep", preprocess), ("model", LogisticRegression(max_iter=300))])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred, zero_division=0)
rec = recall_score(y_test, pred, zero_division=0)
f1  = f1_score(y_test, pred, zero_division=0)

print("\nML — Logistic Regression (predict audience liked ≥70)")
print(f"Accuracy:  {acc:.3f}  Precision: {prec:.3f}  Recall: {rec:.3f}  F1: {f1:.3f}")

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=["Not Liked","Liked"]).plot()
plt.title("Confusion Matrix — Liked Classifier")
plt.tight_layout(); plt.savefig("cm_liked_classifier.png"); plt.close()
print("Saved: cm_liked_classifier.png")

# ----------------------------
# 5) 1 SQL QUERY (for resume)
# ----------------------------
con = sqlite3.connect("movies.db")
df.to_sql("movies", con, if_exists="replace", index=False)

q_top_sql = f"""
SELECT movie_title, audience_rating, {count_col} AS audience_count
FROM movies
WHERE audience_rating IS NOT NULL
  AND ( {count_col} IS NULL OR {count_col} >= {MIN_VOTES} )
ORDER BY audience_rating DESC, COALESCE({count_col}, 0) DESC
LIMIT 10;
"""
top_sql = pd.read_sql(q_top_sql, con)
con.close()

print("\n[SQL] Top 10 by audience (filtered):\n", top_sql)

# ----------------------------
# 6) 3 QUICK INSIGHTS (paste in README)
# ----------------------------
corr = df[["critic_rating", "audience_rating"]].corr().iloc[0,1]
print("\nInsights:")
print(f"- Critic vs audience correlation: {corr:.2f} (positive → they usually agree).")
if not top_movies.empty:
    print(f"- Highest audience-rated (filtered): {top_movies.iloc[0]['movie_title']} ({top_movies.iloc[0]['audience_rating']:.0f}).")
if not genre_gap.empty:
    gname = genre_gap.index[0]; gdiff = genre_gap.iloc[0]["avg_gap"]
    print(f"- Biggest audience-over-critic genre gap: {gname} (avg +{gdiff:.1f} pts).")

print("\nDone. Generated plots + model metrics + one SQL query. ✅")
