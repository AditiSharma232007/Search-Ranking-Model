# 🔍 Search Ranking Model

A machine learning system that ranks search results by relevance using weighted user signals — built in **pure Python with zero ML dependencies**.

> Think of it as a mini version of the algorithm behind every search engine.

---

## 🧠 What it does

When you search for something, not all results are equally relevant. This model scores every document across 4 real-world signals and sorts them from most to least relevant — just like Google, but transparent and fully controllable.

```
final_score = w₀·tfidf + w₁·(ctr × engagement) + w₂·authority + w₃·recency
```

| Signal | Default Weight | What it measures |
|---|---|---|
| TF-IDF | 40% | Do the search words appear in the document? (title matches score 1.8× higher) |
| CTR × Engagement | 25% | Do users click it — and actually stay to read it? |
| Authority | 20% | Is the source a trusted, high-quality domain? |
| Recency | 15% | Is the content fresh or years old? |

Weights are **fully adjustable at runtime** — crank up recency to surface the freshest results, or boost authority to favor trusted sources.

---

## 🚀 Quick start

```bash
git clone https://github.com/YOUR_USERNAME/search-ranking-model.git
cd search-ranking-model
pip install -r requirements.txt
python app.py
```

Then open **http://localhost:5000** in your browser.

---

## 💻 Use as a Python library

```python
from src.ranking_model import SearchRankingModel

model = SearchRankingModel()

# Basic search
results = model.search("machine learning", top_k=5)
for r in results:
    print(f"#{r.rank}  {r.document.title}  →  {r.final_score:.3f}")

# Adjust weights at runtime
model.update_weights(recency=0.5, tfidf=0.2)
results = model.search("neural networks")
```

---

## 🌐 Live demo

Open `templates/index.html` directly in any browser — **no server needed**.

Features:
- Live search with instant re-ranking
- Drag sliders to change signal weights in real time
- Analytics charts (CTR, session time, authority vs recency)
- Full corpus signals table
- Annotated source code view

---

## 📁 Project structure

```
search-ranking-model/
├── src/
│   ├── __init__.py
│   └── ranking_model.py        # Core model — Document, TFIDFEngine, SearchRankingModel
├── tests/
│   └── test_ranking_model.py   # 19 unit tests
├── templates/
│   └── index.html              # Standalone demo website
├── app.py                      # Flask API server
├── requirements.txt
└── README.md
```

---

## 🔌 API endpoints (Flask)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Web demo UI |
| GET | `/api/search?q=machine+learning` | JSON ranked results |
| GET | `/api/search?q=python&w_tfidf=0.5&w_recency=0.3` | Search with custom weights |
| GET | `/api/corpus` | All documents + raw engagement signals |

**Example response:**
```json
{
  "query": "machine learning",
  "total": 7,
  "results": [
    {
      "rank": 1,
      "title": "Introduction to Machine Learning",
      "final_score": 72.4,
      "signals": {
        "tfidf": 88.1,
        "ctr": 65.3,
        "authority": 88.0,
        "recency": 92.5
      }
    }
  ]
}
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

```
19 passed in 0.04s ✓
```

Tests cover signal score bounds, TF-IDF accuracy, title boosting, weight normalization, recency decay, and ranking order correctness.

---

## ⚙️ How the scoring works

**TF-IDF** counts how often your search terms appear in a document relative to how common they are across all documents. Terms in the title get a 1.8× boost.

**CTR × Engagement** takes the raw click-through rate and multiplies it by a quality score — average session time normalized to 5 minutes, blended with inverse bounce rate. A document that gets clicked but immediately abandoned scores low.

**Authority** is a domain trust score between 0 and 1 representing backlink quality and site reputation.

**Recency** applies linear decay from 1.0 (published today) to 0.0 (600+ days old).

All weights are normalized to sum to 1.0 before scoring, so you can set raw values freely without worrying about scale.

---

## 🛠️ Built with

- Python 3.12
- Flask (web server)
- Chart.js (analytics charts in the frontend)
- Zero ML dependencies — just `math` and `json` from the standard library

---

## 📄 License

MIT — free to use, modify, and distribute.
