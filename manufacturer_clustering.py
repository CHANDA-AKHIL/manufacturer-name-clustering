# ============================================================
# Manufacturer Name Clustering
# Author: [Chanda Akhil]
# Description: This script clusters manufacturer names to reduce heterogeneity (e.g., typos, mergers)
# using KNN and connected components on TF-IDF features. Includes visualizations for analysis.
# Dependencies: pandas, numpy, re, matplotlib, seaborn, scikit-learn, wordcloud, scipy
# ============================================================

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from wordcloud import WordCloud
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Load and Prepare Data
# ============================================================
# Load datasets; assumes CSV files are in 'data/' folder.
# Download from https://kaggle.com/competitions/manufacturer-name-clustering/data
try:
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
except FileNotFoundError as e:
    print(f"Error: {e}. Please download datasets and place in 'data/' folder.")
    exit(1)

# Automatically detect text column (assumes object type, excluding ID/target)
def detect_text_column(df):
    for col in df.columns:
        if df[col].dtype == "object" and col.lower() not in ["id", "target"]:
            return col
    raise ValueError("No text column found!")

text_col = detect_text_column(train)
print(f" Detected text column: {text_col}")

# Combine train and test for consistent preprocessing
all_data = pd.concat([train, test], axis=0, ignore_index=True)

# ============================================================
# Text Cleaning
# ============================================================
# Clean text by converting to lowercase, removing special characters, and normalizing spaces
def clean_text(txt):
    if not isinstance(txt, str):
        txt = str(txt)
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

all_data["clean_name"] = all_data[text_col].apply(clean_text)

# ============================================================
# TF-IDF Vectorization
# ============================================================
# Use character n-grams (2-4) to capture spelling variations
vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
X = vectorizer.fit_transform(all_data["clean_name"])

# ============================================================
# KNN Clustering
# ============================================================
# Use KNN to find neighbors and connected components for clustering
k = 5
threshold = 0.25  # Cosine distance threshold for connectivity

nbrs = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
nbrs.fit(X)
distances, indices = nbrs.kneighbors(X)

rows, cols, vals = [], [], []
for i in range(len(all_data)):
    for j, d in zip(indices[i], distances[i]):
        if d < threshold:
            rows.append(i)
            cols.append(j)
            vals.append(1)

adj = csr_matrix((vals, (rows, cols)), shape=(len(all_data), len(all_data)))
n_clusters, labels = connected_components(csgraph=adj, directed=False)
all_data["cluster"] = labels

print(f" Total clusters found: {n_clusters}")

# ============================================================
# Submission File
# ============================================================
# Ensure test set has an ID column; create if missing
if "ID" not in test.columns:
    test["ID"] = range(len(test))

test_clusters = all_data.iloc[len(train):][["ID", "cluster"]]
test_clusters.columns = ["ID", "TARGET"]
test_clusters.to_csv('submission.csv', index=False)
print(" submission.csv saved successfully!")

# ============================================================
# Visualization Section
# ============================================================
# (A) Cluster Size Distribution
plt.figure(figsize=(10, 5))
sns.countplot(x="cluster", data=all_data, palette="viridis")
plt.title(" Cluster Size Distribution", fontsize=16, weight='bold')
plt.xlabel("Cluster Label")
plt.ylabel("Count of Manufacturers")
plt.xticks(rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('visualizations/cluster_size_distribution.png')
plt.show()

# (B) WordClouds for Top Clusters
top_clusters = all_data["cluster"].value_counts().head(3).index
for c in top_clusters:
    cluster_text = " ".join(all_data.loc[all_data["cluster"] == c, "clean_name"])
    plt.figure(figsize=(6, 4))
    wc = WordCloud(width=800, height=400, background_color="white", colormap="plasma").generate(cluster_text)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"☁️ WordCloud for Cluster {c}", fontsize=14, weight='bold')
    plt.savefig(f'visualizations/wordcloud_cluster_{c}.png')
    plt.show()

# (C) PCA 2D Cluster Visualization
pca = PCA(n_components=2, random_state=42)
X_reduced = pca.fit_transform(X.toarray())

viz_df = pd.DataFrame({
    "x": X_reduced[:, 0],
    "y": X_reduced[:, 1],
    "cluster": labels
})

plt.figure(figsize=(10, 7))
palette = sns.color_palette("husl", n_clusters)
sns.scatterplot(data=viz_df, x="x", y="y", hue="cluster", palette=palette, s=50, alpha=0.7, edgecolor="k")
plt.title("🌈 PCA Visualization of Manufacturer Clusters", fontsize=16, weight='bold')
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('visualizations/pca_clusters.png')
plt.show()

# (D) Cluster Heatmap
cluster_summary = all_data.groupby("cluster")["clean_name"].apply(lambda x: " ".join(x)).reset_index()
cluster_summary["length"] = cluster_summary["clean_name"].apply(len)
cluster_summary["count"] = all_data["cluster"].value_counts().sort_index().values

corr_data = cluster_summary[["length", "count"]].corr()
plt.figure(figsize=(5, 4))
sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f", square=True)
plt.title(" Correlation Heatmap: Cluster Size vs Name Length", fontsize=14, weight='bold')
plt.tight_layout()
plt.savefig('visualizations/cluster_heatmap.png')
plt.show()


print(" Visualization Complete — Ready to Present!")
