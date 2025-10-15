# Manufacturer Name Clustering

Clustering project to group manufacturer names and reduce heterogeneity using KNN and TF-IDF

## Overview
This project is based on the Kaggle Manufacturer Name Clustering competition. The goal is to cluster manufacturer names to reduce heterogeneity caused by spelling mistakes, inconsistent legal forms, and additional information, improving data quality for analysis. The dataset includes manufacturer names with variations, and the task is to group identical manufacturers into the same cluster.

Key skills demonstrated:
- Text preprocessing and cleaning
- TF-IDF vectorization with character n-grams
- Clustering using KNN and connected components
- Visualization of cluster distributions and patterns
- Evaluation with F1-Score

Developed as part of my BTech in Computer Science to practice unsupervised learning.

Citation: Georg Vetter. Manufacturer Name Clustering. https://kaggle.com/competitions/manufacturer-name-clustering, 2024. Kaggle.

## Dataset
- **Source**: [Kaggle Manufacturer Name Clustering](https://kaggle.com/competitions/manufacturer-name-clustering/data)
- **Description**: Contains manufacturer names with inconsistencies (e.g., typos, legal forms). Train and test sets combined for preprocessing.
- **Preprocessing**:
  - Cleaned text (lowercase, remove special characters).
  - Vectorized using TF-IDF with char n-grams (2-4).

Note: Datasets not included due to size. Download from Kaggle.

## Methodology
1. **Data Preparation**: Load CSVs, detect text column, clean names.
2. **Feature Extraction**: Apply TF-IDF vectorization.
3. **Clustering**: Use KNN to find neighbors, connected components to form clusters with a 0.25 cosine threshold.
4. **Visualization**: Cluster size distribution, word clouds, PCA scatter plot, correlation heatmap.
5. **Submission**: Generate cluster labels for test set.

Results: Identified ~X clusters (adjust based on run); top clusters show consistent name patterns.


