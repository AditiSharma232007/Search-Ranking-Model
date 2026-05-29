"""
Flask web server for the Search Ranking Model demo.
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from src.ranking_model import SearchRankingModel, SignalWeights

app = Flask(__name__)
model = SearchRankingModel()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()
    top_k = int(request.args.get("top_k", 10))

    # Update weights from query params
    try:
        model.update_weights(
            tfidf=float(request.args.get("w_tfidf", model.weights.tfidf)),
            ctr=float(request.args.get("w_ctr", model.weights.ctr)),
            authority=float(request.args.get("w_authority", model.weights.authority)),
            recency=float(request.args.get("w_recency", model.weights.recency)),
        )
    except ValueError:
        pass

    if not query:
        return jsonify({"results": [], "query": query, "total": 0})

    results = model.search(query, top_k=top_k)
    return jsonify({
        "query": query,
        "total": len(results),
        "weights": {
            "tfidf": model.weights.tfidf,
            "ctr": model.weights.ctr,
            "authority": model.weights.authority,
            "recency": model.weights.recency,
        },
        "results": [r.to_dict() for r in results],
    })


@app.route("/api/corpus")
def corpus():
    docs = []
    for doc in model.corpus:
        docs.append({
            "id": doc.id,
            "title": doc.title,
            "url": doc.url,
            "clicks": doc.clicks,
            "impressions": doc.impressions,
            "ctr_pct": round(doc.ctr * 100, 1),
            "avg_time_s": doc.avg_time_seconds,
            "bounce_pct": round(doc.bounce_rate * 100, 1),
            "authority": doc.authority,
            "days_old": doc.days_old,
        })
    return jsonify({"corpus": docs})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
