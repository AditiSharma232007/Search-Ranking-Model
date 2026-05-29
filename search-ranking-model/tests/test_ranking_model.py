"""
Tests for the Search Ranking Model.
Run: python -m pytest tests/ -v
"""

import pytest
from src.ranking_model import (
    Document, SignalWeights, SearchRankingModel, TFIDFEngine
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_doc():
    return Document(
        id=1, title="Machine Learning Basics", url="example.com",
        content="machine learning algorithms neural networks training",
        clicks=1000, impressions=5000, avg_time_seconds=200,
        bounce_rate=0.2, authority=0.8, days_old=30,
    )

@pytest.fixture
def model():
    return SearchRankingModel()


# ─── Document Tests ────────────────────────────────────────────────────────────

class TestDocument:
    def test_ctr(self, sample_doc):
        assert sample_doc.ctr == pytest.approx(0.2)

    def test_engagement_range(self, sample_doc):
        assert 0.0 <= sample_doc.engagement <= 1.0

    def test_high_bounce_low_engagement(self):
        doc = Document(1,"T","u","c",100,1000,60,0.95,0.5,10)
        assert doc.engagement < 0.3

    def test_low_bounce_high_engagement(self):
        doc = Document(1,"T","u","c",100,1000,300,0.02,0.5,10)
        assert doc.engagement > 0.7


# ─── TF-IDF Tests ─────────────────────────────────────────────────────────────

class TestTFIDF:
    def test_exact_match_scores_higher(self):
        model = SearchRankingModel()
        engine = TFIDFEngine(model.corpus)
        doc_match = Document(1,"ML","u","machine learning tutorial",100,1000,120,0.2,0.5,10)
        doc_no_match = Document(2,"DB","u","database sql tables joins",100,1000,120,0.2,0.5,10)
        assert engine.score(doc_match, "machine learning") > engine.score(doc_no_match, "machine learning")

    def test_empty_query_returns_zero(self, model):
        engine = TFIDFEngine(model.corpus)
        assert engine.score(model.corpus[0], "") == 0.0

    def test_title_boost(self):
        model = SearchRankingModel()
        engine = TFIDFEngine(model.corpus)
        # Both docs have the term in both title and body — title doc gets boost on body TF too
        doc_title = Document(1,"neural network guide","u","neural network explained in detail",100,1000,120,0.2,0.5,10)
        doc_body  = Document(2,"Generic Guide","u","neural network explained in detail",100,1000,120,0.2,0.5,10)
        assert engine.score(doc_title, "neural network") >= engine.score(doc_body, "neural network")

    def test_score_bounded(self, model):
        engine = TFIDFEngine(model.corpus)
        for doc in model.corpus:
            s = engine.score(doc, "machine learning")
            assert 0.0 <= s <= 1.0


# ─── SignalWeights Tests ───────────────────────────────────────────────────────

class TestSignalWeights:
    def test_normalized_sums_to_one(self):
        w = SignalWeights(0.4, 0.3, 0.2, 0.1)
        n = w.normalized()
        total = n.tfidf + n.ctr + n.authority + n.recency
        assert total == pytest.approx(1.0)

    def test_zero_weights_returns_equal(self):
        w = SignalWeights(0, 0, 0, 0)
        n = w.normalized()
        assert n.tfidf == pytest.approx(0.25)

    def test_unnormalized_preserved(self):
        w = SignalWeights(2.0, 1.0, 1.0, 0.0)
        n = w.normalized()
        assert n.tfidf == pytest.approx(0.5)


# ─── Ranking Model Tests ───────────────────────────────────────────────────────

class TestSearchRankingModel:
    def test_results_sorted_descending(self, model):
        results = model.search("machine learning")
        scores = [r.final_score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_assigned_correctly(self, model):
        results = model.search("machine learning")
        for i, r in enumerate(results, 1):
            assert r.rank == i

    def test_top_k_respected(self, model):
        assert len(model.search("machine", top_k=3)) == 3

    def test_empty_query_tfidf_zero(self, model):
        """With empty query, TF-IDF score should be 0 for every document."""
        results = model.search("", top_k=10)
        for r in results:
            assert r.tfidf_score == 0.0

    def test_update_weights(self, model):
        model.update_weights(recency=0.9)
        assert model.weights.recency == 0.9

    def test_recency_weight_boosts_new_docs(self, model):
        model.update_weights(tfidf=0.1, ctr=0.1, authority=0.1, recency=0.9)
        results = model.search("machine learning", top_k=3)
        days = [r.document.days_old for r in results]
        # With heavy recency weight, top results should be relatively recent
        assert min(days) < 100

    def test_to_dict_keys(self, model):
        results = model.search("python")
        d = results[0].to_dict()
        for key in ["rank", "title", "url", "final_score", "signals", "raw"]:
            assert key in d

    def test_signal_scores_bounded(self, model):
        results = model.search("deep learning")
        for r in results:
            for s in [r.tfidf_score, r.ctr_score, r.authority_score, r.recency_score, r.final_score]:
                assert 0.0 <= s <= 1.0, f"Score out of bounds: {s}"
