import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
# Keep this CSV file in the same folder as this Python script.
df = pd.read_csv("movie_ratings_analysis_synthetic_dataset.csv")

print("First 5 rows")
print(df.head())

print(f"\nDataset shape (Rows, Columns): {df.shape}")

print("\n--- Column Info and Types ---")
df.info()

print("\n--- Missing value count ---")
print(df.isnull().sum())


# Data cleaning
df["Genre"] = df["Genre"].fillna("unknown")
df["Director"] = df["Director"].fillna("unknown")

df["Rating"] = df["Rating"].fillna(df["Rating"].median())
df["Budget_Million_USD"] = df["Budget_Million_USD"].fillna(
    df["Budget_Million_USD"].median()
)
df["Revenue_Million_USD"] = df["Revenue_Million_USD"].fillna(
    df["Revenue_Million_USD"].median()
)
df["Year"] = df["Year"].astype(int)

print("\n--- Missing values after cleaning ---")
print(df.isnull().sum())


# Genre analysis
genre_analysis = (
    df.groupby("Genre")[["Rating", "Budget_Million_USD", "Revenue_Million_USD"]]
    .mean()
    .sort_values("Rating", ascending=False)
)

print("\n--- Average Metrics by Genre ---")
print(genre_analysis.round(2))


# Director analysis
director_analysis = df.groupby("Director").agg(
    Avg_Rating=("Rating", "mean"),
    Total_Movies=("Movie_Title", "count")
).sort_values(by="Avg_Rating", ascending=False)

print("\n--- Top Directors by Average Rating (Min 10 Movies) ---")
top_directors = director_analysis[director_analysis["Total_Movies"] >= 10]
print(top_directors.head(10).round(2))


# Yearly analysis
year_analysis = df.groupby("Year")["Rating"].mean()

print("\n--- Yearly Average Rating Trend (First 5 & Last 5 Years) ---")
print(year_analysis.to_frame(name="Avg_Rating").round(2))


# Average rating by genre
genre_avg_rating = df.groupby("Genre")["Rating"].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 4.5))
plt.bar(
    genre_avg_rating.index,
    genre_avg_rating.values,
    color="royalblue",
    edgecolor="black"
)
plt.title("Average Rating by Genre", fontsize=13, fontweight="bold")
plt.xlabel("Genre", fontsize=11)
plt.ylabel("Average Rating (0-10)", fontsize=11)
plt.xticks(rotation=45)
plt.ylim(0, 10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# Top 10 directors by average rating (minimum 10 movies)
plt.figure(figsize=(11, 4.5))
plt.bar(
    top_directors.head(10).index,
    top_directors.head(10)["Avg_Rating"],
    color="teal",
    edgecolor="black"
)
plt.title(
    "Top 10 Directors by Average Rating (Min 10 Movies)",
    fontsize=13,
    fontweight="bold"
)
plt.xlabel("Director", fontsize=11)
plt.ylabel("Average Rating", fontsize=11)
plt.xticks(rotation=45)
plt.ylim(0, 10)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# Movie rating trends over time
yearly_avg = df.groupby("Year")["Rating"].mean()

plt.figure(figsize=(11, 4.5))
plt.plot(
    yearly_avg.index,
    yearly_avg.values,
    marker="o",
    color="crimson",
    linewidth=2,
    markersize=5
)
plt.title("Movie Rating Trends Over Time (2000 - 2025)", fontsize=13, fontweight="bold")
plt.xlabel("Year", fontsize=11)
plt.ylabel("Average Rating", fontsize=11)
plt.xticks(yearly_avg.index, rotation=45)
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# Budget vs. rating and revenue vs. rating
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(
    df["Budget_Million_USD"],
    df["Rating"],
    alpha=0.6,
    color="darkorange",
    edgecolor="black"
)
axes[0].set_title("Budget vs. Rating", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Budget (Million USD)", fontsize=11)
axes[0].set_ylabel("Rating", fontsize=11)
axes[0].grid(True, linestyle="--", alpha=0.6)

axes[1].scatter(
    df["Revenue_Million_USD"],
    df["Rating"],
    alpha=0.6,
    color="forestgreen",
    edgecolor="black"
)
axes[1].set_title("Revenue vs. Rating", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Revenue (Million USD)", fontsize=11)
axes[1].set_ylabel("Rating", fontsize=11)
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()


# Top 10 highest-rated movies
top_10_movies = df.sort_values(by="Rating", ascending=False).head(10)

plt.figure(figsize=(10, 5))
plt.barh(
    top_10_movies["Movie_Title"],
    top_10_movies["Rating"],
    color="mediumpurple",
    edgecolor="black"
)
plt.title("Top 10 Highest-Rated Movies", fontsize=13, fontweight="bold")
plt.xlabel("Rating (0-10)", fontsize=11)
plt.ylabel("Movie Title", fontsize=11)
plt.xlim(0, 10)
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
