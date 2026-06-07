"""Tests for the persona / user-model layer of the memory graph store.

Reinforcement (repeat evidence strengthens, no duplicates), time-decay,
non-lossy consolidation, and the persona reader — all exercised directly against
the store (no extractor, no LLM, real in-memory SQLite).
"""

import types as _types
from datetime import datetime, timedelta

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
if not isinstance(sqlalchemy, _types.ModuleType):
    pytest.skip("sqlalchemy is stubbed in this environment", allow_module_level=True)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base, GraphEdge
from src.memory_graph import MemoryGraph

if type(Base).__name__ == "MagicMock":
    pytest.skip("core.database is stubbed — run this file in isolation", allow_module_level=True)


def _graph():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return MemoryGraph(session_factory=sessionmaker(bind=engine))


# ── reinforcement (no duplicates; evidence accumulates) ──

def test_repeat_observation_reinforces_not_duplicates():
    g = _graph()
    s = g.upsert_node("alice", "You", "person")
    tea = g.upsert_node("alice", "tea", "preference")
    g.add_edge("alice", s, "HAS_PREFERENCE", tea, fact="You like tea")
    g.add_edge("alice", s, "HAS_PREFERENCE", tea, fact="You like tea")
    g.add_edge("alice", s, "HAS_PREFERENCE", tea, fact="You like tea")
    current = g.current_edges("alice", predicate="HAS_PREFERENCE")
    assert len(current) == 1          # not duplicated
    assert current[0].salience == 3.0  # reinforced thrice


# ── decay + consolidation (non-lossy) ──

def test_effective_salience_decays_with_time():
    g = _graph()
    s = g.upsert_node("alice", "You", "person")
    o = g.upsert_node("alice", "tea", "preference")
    t0 = datetime(2026, 1, 1)
    g.add_edge("alice", s, "HAS_PREFERENCE", o, fact="You like tea", now=t0)
    db = g._session()
    try:
        edge = db.query(GraphEdge).first()
        fresh = g._effective_salience(edge, t0)
        decayed = g._effective_salience(edge, t0 + timedelta(days=90))
        assert fresh == pytest.approx(1.0)
        assert decayed < fresh
    finally:
        db.close()


def test_consolidate_retires_stale_low_salience_nonlossy():
    g = _graph()
    s = g.upsert_node("alice", "You", "person")
    o = g.upsert_node("alice", "ska music", "preference")
    t0 = datetime(2026, 1, 1)
    g.add_edge("alice", s, "HAS_PREFERENCE", o, fact="You like ska music", now=t0)
    # Far in the future, a single-evidence preference has decayed below the floor.
    later = t0 + timedelta(days=400)
    retired = g.consolidate_persona("alice", now=later)
    assert retired == 1
    assert g.current_edges("alice", predicate="HAS_PREFERENCE") == []
    # Non-lossy: the row still exists, just marked invalid.
    db = g._session()
    try:
        assert db.query(GraphEdge).count() == 1
    finally:
        db.close()


def test_consolidate_never_retires_functional_state():
    g = _graph()
    s = g.upsert_node("alice", "You", "person")
    rome = g.upsert_node("alice", "Rome", "place")
    t0 = datetime(2026, 1, 1)
    g.add_edge("alice", s, "LIVES_IN", rome, fact="You live in Rome", now=t0)
    retired = g.consolidate_persona("alice", now=t0 + timedelta(days=1000))
    assert retired == 0  # functional state persists until superseded
    assert len(g.current_edges("alice", predicate="LIVES_IN")) == 1


# ── persona reader ──

def test_get_persona_sections_and_ordering():
    g = _graph()
    you = g.upsert_node("alice", "You", "person")
    tea = g.upsert_node("alice", "tea", "preference")
    coffee = g.upsert_node("alice", "coffee", "preference")
    onions = g.upsert_node("alice", "onions", "preference")
    # Reinforce tea so it outranks coffee in the "likes" section.
    g.add_edge("alice", you, "HAS_PREFERENCE", tea, fact="You like tea")
    g.add_edge("alice", you, "HAS_PREFERENCE", tea, fact="You like tea")
    g.add_edge("alice", you, "HAS_PREFERENCE", coffee, fact="You like coffee")
    g.add_edge("alice", you, "DISLIKES", onions, fact="You don't like onions")
    persona = g.get_persona("alice")
    assert "likes" in persona and "dislikes" in persona
    assert persona["likes"][0]["object"] == "tea"  # strongest like first


def test_persona_summary_text_lists_facts():
    g = _graph()
    you = g.upsert_node("alice", "You", "person")
    rome = g.upsert_node("alice", "Rome", "place")
    tea = g.upsert_node("alice", "tea", "preference")
    g.add_edge("alice", you, "LIVES_IN", rome, fact="You live in Rome")
    g.add_edge("alice", you, "HAS_PREFERENCE", tea, fact="You like tea")
    txt = g.persona_summary_text("alice")
    assert "tea" in txt.lower()
    assert "rome" in txt.lower()
