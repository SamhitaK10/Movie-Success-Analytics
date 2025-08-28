# 🎬 Movie Success Analytics — SQL + ML

An end-to-end data project analyzing the Rotten Tomatoes movies dataset to explore what drives audience success.  
Mixes **SQL** (SQLite), **pandas**, and a small **ML model** to predict whether a movie is “liked” (audience score ≥ 70).

---

## 🚀 Features
- Data cleaning & preprocessing (normalize ratings, handle genres)
- **SQL query (SQLite):** Top 10 audience-rated movies with min-vote filter
- **Pandas analysis:**
  - Genre-level audience vs critic rating gaps
  - Most “controversial” films (biggest disagreements)
- **Visualizations:**
  - Audience rating distribution
  - Critic vs Audience scatter plot
- **Machine Learning:**
  - Logistic Regression classifier (predict if audience “liked” a movie)

---

## 📂 Dataset
This project uses the **Rotten Tomatoes movies dataset** (publicly available on Kaggle and GitHub mirrors).  
Save the file as **`rotten_tomatoes_movies.csv`** in the same folder as `main.py`.

> Dataset not included in this repo to keep it light. Any CSV with  
> `movie_title`, `critic_rating`, `audience_rating`, and `genres` columns will work.

---

## 🛠 Tech Stack
- Python (pandas, numpy, matplotlib, scikit-learn)
- SQLite (via Python’s built-in `sqlite3`)

---

## ▶️ How to Run
```bash
pip install -r requirements.txt
python main.py

📊 Example Results (will vary by dataset)

Critic vs audience correlation: ~0.65 → they usually agree

Highest audience-rated (filtered): a popular film with >500 votes

Biggest gap genre: e.g., Comedy (+6–10 pts audience > critics)

ML classifier (LogReg, liked ≥70):

Accuracy ~0.72 | Precision ~0.70 | Recall ~0.68 | F1 ~0.69

🔍 SQL Example
SELECT movie_title, audience_rating, audience_count
FROM movies
WHERE audience_rating IS NOT NULL
  AND audience_count >= 500
ORDER BY audience_rating DESC, audience_count DESC
LIMIT 10;

🌟 What I Learned

Writing SQL queries to answer top-N and genre-level questions

Handling real-world messy data (column renames, scaling ratings, missing values)

Building a simple ML classifier to frame a business-style question (“liked” or not)

Communicating results through plots + clear insights
