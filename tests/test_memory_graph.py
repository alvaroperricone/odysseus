"""Tests for the persistent memory graph store (Phase 3a).

Uses real SQLAlchemy + in-memory SQLite (no LLM, no Chroma). Covers entity
resolution, bi-temporal supersession, owner-scoping, k-hop traversal, and
as-of time travel.
"""

import types as _types
from datetime import datetime, timedelta

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")
if not isinstance(sqlalchemy, _types.ModuleType):
    pytest.skip("sqlalchemy is stubbed in this environment", allow_module_level=True)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.database import Base, GraphNode, GraphEdge
from src.memory_graph import MemoryGraph, normalize_predicate, normalize_type

if type(Base).__name__ == "MagicMock":
    pytest.skip("core.database is stubbed — run this file in isolation", allow_module_level=True)


def _graph():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    return MemoryGraph(session_factory=Session)


# ── ontology normalization ──

def test_normalize_predicate_and_type():
    assert normalize_predicate("lives_in") == "LIVES_IN"
    assert normalize_predicate("Has Preference") == "HAS_PREFERENCE"
    assert normalize_predicate("bogus") is None
    assert normalize_type("Person") == "person"
    assert normalize_type("nonsense") == "topic"


# ── nodes / entity resolution ──

def test_upsert_node_dedupes_by_name_and_collects_aliases():
    g = _graph()
    a = g.upsert_node("alice", "Rome", "place")
    b = g.upsert_node("alice", "rome", "place")   # case-insensitive same node
    assert a == b

    db = g._session()
    try:
        node = db.query(GraphNode).filter(GraphNode.id == a).first()
        assert node.name == "Rome"
    finally:
        db.close()


def test_upsert_node_owner_scoped():
    g = _graph()
    a = g.upsert_node("alice", "Rome", "place")
    b = g.upsert_node("bob", "Rome", "place")
    assert a != b  # different owners -> different nodes


def test_resolve_node_alias_match():
    g = _graph()
    db = g._session()
    try:
        nid = g.upsert_node("alice", "Robert", "person", aliases=["Bob", "Bobby"], db=db)
        assert g.resolve_node(db, "alice", "bobby") == nid
        assert g.resolve_node(db, "alice", "Bob", "person") == nid
        assert g.resolve_node(db, "alice", "Charlie") is None
    finally:
        db.close()


# ── edges + temporal ──

def test_add_edge_rejects_unknown_predicate():
    g = _graph()
    s = g.upsert_node("alice", "Alice", "person")
    o = g.upsert_node("alice", "Rome", "place")
    assert g.add_edge("alice", s, "TELEPORTS_TO", o) is None


def test_functional_predicate_supersedes_old_edge_nonlossy():
    g = _graph()
    s = g.upsert_node("alice", "Alice", "person")
    rome = g.upsert_node("alice", "Rome", "place")
    milan = g.upsert_node("alice", "Milan", "place")

    t0 = datetime(2026, 1, 1)
    t1 = datetime(2026, 6, 1)
    e1 = g.add_edge("alice", s, "LIVES_IN", rome, valid_at=t0, now=t0)
    e2 = g.add_edge("alice", s, "LIVES_IN", milan, valid_at=t1, now=t1)

    db = g._session()
    try:
        old = db.query(GraphEdge).filter(GraphEdge.id == e1).first()
        new = db.query(GraphEdge).filter(GraphEdge.id == e2).first()
        # Old edge superseded, not deleted (non-lossy).
        assert old is not None
        assert old.invalid_at == t1
        assert old.expired_at is not None
        # New edge is current.
        assert new.invalid_at is None and new.expired_at is None
    finally:
        db.close()

    current = g.current_edges("alice", subject_id=s, predicate="LIVES_IN")
    assert len(current) == 1
    assert current[0].object_id == milan


def test_non_functional_predicate_keeps_multiple():
    g = _graph()
    s = g.upsert_node("alice", "Alice", "person")
    tea = g.upsert_node("alice", "tea", "preference")
    coffee = g.upsert_node("alice", "coffee", "preference")
    g.add_edge("alice", s, "HAS_PREFERENCE", tea)
    g.add_edge("alice", s, "HAS_PREFERENCE", coffee)
    current = g.current_edges("alice", subject_id=s, predicate="HAS_PREFERENCE")
    assert len(current) == 2  # HAS_PREFERENCE is not functional — both stand


# ── traversal ──

def test_neighbors_khop_traversal():
    g = _graph()
    alice = g.upsert_node("alice", "Alice", "person")
    bob = g.upsert_node("alice", "Bob", "person")
    acme = g.upsert_node("alice", "Acme", "organization")
    g.add_edge("alice", alice, "RELATED_TO", bob)
    g.add_edge("alice", bob, "WORKS_FOR", acme)

    one_hop = g.neighbors("alice", [alice], hops=1)
    ids1 = {n.id for n in one_hop["nodes"]}
    assert ids1 == {alice, bob}  # Acme is 2 hops away

    two_hop = g.neighbors("alice", [alice], hops=2)
    ids2 = {n.id for n in two_hop["nodes"]}
    assert ids2 == {alice, bob, acme}


def test_neighbors_owner_scoped():
    g = _graph()
    a_alice = g.upsert_node("alice", "Shared", "topic")
    a_bob = g.upsert_node("bob", "Shared", "topic")
    other = g.upsert_node("bob", "Secret", "topic")
    g.add_edge("bob", a_bob, "RELATED_TO", other)
    # Alice traversing her own seed sees nothing of bob's graph.
    res = g.neighbors("alice", [a_alice], hops=2)
    assert {n.id for n in res["nodes"]} == {a_alice}


def test_graph_retrieve_seeds_and_expands():
    g = _graph()
    alice = g.upsert_node("alice", "Alice", "person")
    bob = g.upsert_node("alice", "Bob", "person")
    acme = g.upsert_node("alice", "Acme Corp", "organization")
    g.add_edge("alice", alice, "RELATED_TO", bob, fact="Alice knows Bob")
    g.add_edge("alice", bob, "WORKS_FOR", acme, fact="Bob works for Acme Corp")

    hits = g.graph_retrieve("tell me about Bob", "alice", hops=2)
    texts = " ".join(h["text"] for h in hits)
    assert hits
    assert all(h["source"] == "graph" for h in hits)
    assert all(h["score"] >= 0.35 for h in hits)  # above the reasoner keep-floor
    assert "Bob" in texts


def test_graph_retrieve_no_seed_returns_empty():
    g = _graph()
    g.upsert_node("alice", "Rome", "place")
    assert g.graph_retrieve("quantum chromodynamics", "alice") == []


def test_graph_retrieve_owner_scoped():
    g = _graph()
    g.upsert_node("alice", "Alice", "person")
    b = g.upsert_node("bob", "Bob", "person")
    org = g.upsert_node("bob", "Acme", "organization")
    g.add_edge("bob", b, "WORKS_FOR", org, fact="Bob works for Acme")
    # Alice querying for Bob's data gets nothing.
    assert g.graph_retrieve("Bob Acme", "alice") == []


# ── get_name (identity) ──

def test_get_name_distinct_nodes_returns_both():
    g = _graph()
    s = g.upsert_node("alice", "Alice", "person")
    full = g.upsert_node("alice", "Alvaro", "person")
    short = g.upsert_node("alice", "alv", "person")
    g.add_edge("alice", s, "HAS_NAME", full)
    g.add_edge("alice", s, "PREFERS_NAME", short)

    out = g.get_name("alice")
    assert out.get("name") == "Alvaro"
    assert out.get("preferred") == "alv"


def test_get_name_shared_node_returns_both():
    # "mi chiamo Alvaro" + "chiamami Alvaro" -> entity resolution collapses
    # both surface forms to ONE node id, but the predicates differ. get_name
    # must still surface BOTH name and preferred from that shared node.
    g = _graph()
    s = g.upsert_node("alice", "Alice", "person")
    alvaro = g.upsert_node("alice", "Alvaro", "person")
    g.add_edge("alice", s, "HAS_NAME", alvaro)
    g.add_edge("alice", s, "PREFERS_NAME", alvaro)

    out = g.get_name("alice")
    assert out.get("name") == "Alvaro"
    assert out.get("preferred") == "Alvaro"


def test_neighbors_as_of_time_travel():
    g = _graph()
    s = g.upsert_node("alice", "Alice", "person")
    rome = g.upsert_node("alice", "Rome", "place")
    milan = g.upsert_node("alice", "Milan", "place")
    t0 = datetime(2026, 1, 1)
    t1 = datetime(2026, 6, 1)
    g.add_edge("alice", s, "LIVES_IN", rome, valid_at=t0, now=t0)
    g.add_edge("alice", s, "LIVES_IN", milan, valid_at=t1, now=t1)

    # Now: lives in Milan.
    now_res = g.neighbors("alice", [s], hops=1)
    assert milan in {n.id for n in now_res["nodes"]}
    assert rome not in {n.id for n in now_res["nodes"]}

    # As of March 2026: still lived in Rome.
    past_res = g.neighbors("alice", [s], hops=1, as_of=datetime(2026, 3, 1))
    past_ids = {n.id for n in past_res["nodes"]}
    assert rome in past_ids
    assert milan not in past_ids
