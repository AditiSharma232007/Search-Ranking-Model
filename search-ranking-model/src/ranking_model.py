"""
Search Ranking Model
====================
Ranks documents by relevance using weighted user signals:
  - TF-IDF (lexical relevance)
  - Click-Through Rate × Engagement quality
  - Authority score
  - Recency decay

Usage:
    from src.ranking_model import SearchRankingModel
    model = SearchRankingModel()
    results = model.search("machine learning")
"""

import math
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class Document:
    id: int
    title: str
    url: str
    content: str
    clicks: int
    impressions: int
    avg_time_seconds: float
    bounce_rate: float        # 0.0 – 1.0
    authority: float          # 0.0 – 1.0
    days_old: int

    @property
    def ctr(self) -> float:
        return self.clicks / max(self.impressions, 1)

    @property
    def engagement(self) -> float:
        """Quality multiplier: blend of session depth and bounce avoidance."""
        time_score = min(self.avg_time_seconds / 300, 1.0)
        bounce_score = 1.0 - self.bounce_rate
        return (time_score + bounce_score) / 2


@dataclass
class SignalWeights:
    tfidf: float = 0.40
    ctr: float = 0.25
    authority: float = 0.20
    recency: float = 0.15

    def normalized(self) -> "SignalWeights":
        total = self.tfidf + self.ctr + self.authority + self.recency
        if total == 0:
            return SignalWeights(0.25, 0.25, 0.25, 0.25)
        return SignalWeights(
            tfidf=self.tfidf / total,
            ctr=self.ctr / total,
            authority=self.authority / total,
            recency=self.recency / total,
        )


@dataclass
class ScoredResult:
    document: Document
    final_score: float
    tfidf_score: float
    ctr_score: float
    authority_score: float
    recency_score: float
    rank: int = 0

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "id": self.document.id,
            "title": self.document.title,
            "url": self.document.url,
            "snippet": " ".join(self.document.content.split()[:20]) + "...",
            "final_score": round(self.final_score * 100, 2),
            "signals": {
                "tfidf": round(self.tfidf_score * 100, 2),
                "ctr": round(self.ctr_score * 100, 2),
                "authority": round(self.authority_score * 100, 2),
                "recency": round(self.recency_score * 100, 2),
            },
            "raw": {
                "clicks": self.document.clicks,
                "ctr_pct": round(self.document.ctr * 100, 1),
                "avg_time_s": self.document.avg_time_seconds,
                "bounce_pct": round(self.document.bounce_rate * 100, 1),
                "authority": self.document.authority,
                "days_old": self.document.days_old,
            },
        }


# ─── TF-IDF Engine ────────────────────────────────────────────────────────────

class TFIDFEngine:
    """Lightweight TF-IDF with title boosting (no external dependencies)."""

    TITLE_BOOST = 1.8
    RECENCY_MAX_DAYS = 600

    def __init__(self, corpus: List[Document]):
        self.corpus = corpus
        self._idf_cache: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in text.split() if len(w) > 1]

    def _idf(self, term: str) -> float:
        if term in self._idf_cache:
            return self._idf_cache[term]
        df = sum(1 for d in self.corpus if term in d.content.lower())
        idf = math.log(len(self.corpus) / (df + 1)) + 1
        self._idf_cache[term] = idf
        return idf

    def score(self, doc: Document, query: str) -> float:
        terms = self._tokenize(query)
        if not terms:
            return 0.0

        words = self._tokenize(doc.content)
        title_words = doc.title.lower()
        total_words = max(len(words), 1)
        score = 0.0

        for term in terms:
            tf = sum(1 for w in words if term in w) / total_words
            boost = self.TITLE_BOOST if term in title_words else 1.0
            score += tf * self._idf(term) * boost

        return min(score / len(terms), 1.0)


# ─── Ranking Model ────────────────────────────────────────────────────────────

class SearchRankingModel:
    """
    Weighted linear ranking model combining:
        score = Σ(wᵢ × signalᵢ) / Σwᵢ
    """

    RECENCY_MAX_DAYS = 600

    def __init__(
        self,
        corpus: Optional[List[Document]] = None,
        weights: Optional[SignalWeights] = None,
    ):
        self.corpus = corpus or self._default_corpus()
        self.weights = weights or SignalWeights()
        self._tfidf = TFIDFEngine(self.corpus)

    # ── Signal Scorers ────────────────────────────────────────────────────────

    def _score_tfidf(self, doc: Document, query: str) -> float:
        return self._tfidf.score(doc, query)

    def _score_ctr(self, doc: Document) -> float:
        """Normalize CTR to [0, 1]; penalised by engagement quality."""
        return min(doc.ctr * 2.5, 1.0) * doc.engagement

    def _score_authority(self, doc: Document) -> float:
        return doc.authority

    def _score_recency(self, doc: Document) -> float:
        return max(0.0, 1.0 - doc.days_old / self.RECENCY_MAX_DAYS)

    # ── Main Interface ────────────────────────────────────────────────────────

    def score_document(self, doc: Document, query: str) -> ScoredResult:
        w = self.weights.normalized()
        tfidf_s = self._score_tfidf(doc, query)
        ctr_s = self._score_ctr(doc)
        auth_s = self._score_authority(doc)
        rec_s = self._score_recency(doc)
        final = w.tfidf * tfidf_s + w.ctr * ctr_s + w.authority * auth_s + w.recency * rec_s
        return ScoredResult(
            document=doc,
            final_score=final,
            tfidf_score=tfidf_s,
            ctr_score=ctr_s,
            authority_score=auth_s,
            recency_score=rec_s,
        )

    def search(self, query: str, top_k: int = 10) -> List[ScoredResult]:
        """Return top-k ranked results for a query."""
        scored = [self.score_document(doc, query) for doc in self.corpus]
        scored.sort(key=lambda r: r.final_score, reverse=True)
        for i, result in enumerate(scored):
            result.rank = i + 1
        return scored[:top_k]

    def update_weights(self, **kwargs) -> None:
        """Dynamically update signal weights. e.g. update_weights(recency=0.4)"""
        for key, val in kwargs.items():
            if hasattr(self.weights, key):
                setattr(self.weights, key, float(val))

    # ── Default Corpus ────────────────────────────────────────────────────────

    @staticmethod
    def _default_corpus() -> List[Document]:
        return [
            Document(1, "Introduction to Machine Learning", "ml-intro.com",
                "machine learning algorithms supervised unsupervised neural networks deep learning training data features classification regression",
                8420, 12000, 187, 0.18, 0.88, 45),
            Document(2, "Deep Neural Networks Explained", "deeplearn.io",
                "neural networks deep learning layers backpropagation gradient descent machine learning model training",
                6100, 9800, 234, 0.12, 0.92, 12),
            Document(3, "Database Indexing Strategies", "dbguide.net",
                "database index query optimization sql performance b-tree hash indexing data structures",
                4200, 7600, 156, 0.31, 0.74, 380),
            Document(4, "Python for Data Science", "pydata.org",
                "python pandas numpy machine learning scikit-learn data analysis visualization matplotlib",
                9800, 18000, 211, 0.22, 0.95, 90),
            Document(5, "Reinforcement Learning from Human Feedback", "rlhf-paper.ai",
                "reinforcement learning reward model human feedback language models fine-tuning alignment machine learning",
                3300, 4100, 312, 0.08, 0.87, 8),
            Document(6, "Graph Neural Networks Survey", "gnn-survey.edu",
                "graph neural networks node classification link prediction graph learning deep learning machine learning",
                1800, 3200, 278, 0.14, 0.81, 60),
            Document(7, "SQL vs NoSQL Databases", "dbcompare.com",
                "sql nosql database comparison relational document store performance scalability data storage",
                5600, 10200, 143, 0.27, 0.71, 520),
            Document(8, "Transformer Architecture Deep Dive", "transformers.ml",
                "transformer attention mechanism self-attention bert gpt language model neural network deep learning",
                7200, 9600, 298, 0.09, 0.93, 30),
            Document(9, "Feature Engineering Techniques", "mlcraft.dev",
                "feature engineering selection machine learning preprocessing normalization encoding classification regression",
                3900, 6800, 189, 0.24, 0.76, 150),
            Document(10, "Big Data Processing with Spark", "sparkguide.io",
                "apache spark big data processing distributed computing machine learning mllib pipeline",
                4500, 8300, 167, 0.29, 0.79, 200),
        ]
