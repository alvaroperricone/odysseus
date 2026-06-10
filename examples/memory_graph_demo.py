#!/usr/bin/env python3
"""Standalone demo of the owner-scoped, bi-temporal memory-graph store.

No API keys, no external services: an in-memory SQLite graph seeded with a few
facts, exercising the four behaviours that make the store useful and that a
small local model cannot corrupt:

  1. Non-lossy bi-temporal supersession + as-of time travel
  2. Code-first entity resolution (no model judgment)
  3. Bounded k-hop retrieval vs a flat keyword store (no LLM at query time)
  4. Persona: salience reinforcement + time-decay + non-lossy consolidation

Run from the repo root:

    python examples/memory_graph_demo.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

# Keep the demo hermetic: never touch the app DB. core.database reads
# DATABASE_URL at import time (default ./data/app.db); point it at :memory:.
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine                       # noqa: E402
from sqlalchemy.orm import sessionmaker                     # noqa: E402

from core.database import Base, GraphNode, GraphEdge        # noqa: E402
from src.memory_graph import MemoryGraph                    # noqa: E402
from src.text_tokens import content_tokens                  # noqa: E402

OWNER = "demo-user"


def make_graph() -> MemoryGraph:
    """A fresh, isolated in-memory graph (each demo gets its own)."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return MemoryGraph(session_factory=sessionmaker(bind=engine))


def rule(title: str) -> None:
    print("\n" + "=" * 70 + "\n" + title + "\n" + "=" * 70)


def demo_supersession_and_time_travel() -> None:
    rule("1) Non-lossy supersession + as-of time travel   LIVES_IN: Rome -> Milan")
    g = make_graph()
    me = g.upsert_node(OWNER, "Alvaro", "person")
    rome = g.upsert_node(OWNER, "Rome", "place")
    milan = g.upsert_node(OWNER, "Milan", "place")

    jan, jun = datetime(2026, 1, 1), datetime(2026, 6, 1)
    g.add_edge(OWNER, me, "LIVES_IN", rome, fact="Alvaro lives in Rome", valid_at=jan, now=jan)
    g.add_edge(OWNER, me, "LIVES_IN", milan, fact="Alvaro lives in Milan", valid_at=jun, now=jun)

    def lives_in(as_of=None):
        sub = g.neighbors(OWNER, [me], hops=1, as_of=as_of)
        names = {n.id: n.name for n in sub["nodes"]}
        return [names[e.object_id] for e in sub["edges"] if e.predicate == "LIVES_IN"]

    print("  now             ->", lives_in())
    print("  as-of 2026-03   ->", lives_in(datetime(2026, 3, 1)))

    db = g._session()
    try:
        rows = db.query(GraphEdge).filter(
            GraphEdge.owner == OWNER, GraphEdge.predicate == "LIVES_IN").all()
        print(f"  edges on disk   -> {len(rows)} (nothing deleted; history auditable)")
        for e in rows:
            state = ("current" if e.invalid_at is None and e.expired_at is None
                     else f"superseded at {e.invalid_at.date()}")
            print(f"      {e.fact + ' ':.<34} {state}")
    finally:
        db.close()


def demo_entity_resolution() -> None:
    rule("2) Code-first entity resolution   no model can mint a duplicate")
    g = make_graph()
    nid = g.upsert_node(OWNER, "Robert", "person", aliases=["Bob", "Bobby"])
    for surface in ["robert", "BOB", "  bobby ", "Robert"]:
        same = g.upsert_node(OWNER, surface, "person")
        print(f"  upsert {surface!r:>10} -> {'same node' if same == nid else 'NEW NODE'}")

    db = g._session()
    try:
        people = db.query(GraphNode).filter(
            GraphNode.owner == OWNER, GraphNode.type == "person").all()
        print("  person nodes    ->", [n.name for n in people], "(surface forms collapsed to one)")
    finally:
        db.close()


def demo_khop_vs_flat() -> None:
    rule("3) Bounded k-hop retrieval vs a flat keyword store   no LLM at query")
    g = make_graph()
    me = g.upsert_node(OWNER, "Alvaro", "person")
    acme = g.upsert_node(OWNER, "Acme Corp", "organization")
    milan = g.upsert_node(OWNER, "Milan", "place")
    g.add_edge(OWNER, me, "WORKS_FOR", acme, fact="Alvaro works for Acme Corp")
    g.add_edge(OWNER, acme, "LOCATED_IN", milan, fact="Acme Corp is based in Milan")

    query = "where is Alvaro's office"
    facts = ["Alvaro works for Acme Corp", "Acme Corp is based in Milan"]
    qtoks = set(content_tokens(query))
    flat = [f for f in facts if qtoks & set(content_tokens(f))]
    graph = [h["text"] for h in g.graph_retrieve(query, OWNER, hops=2)]

    print("  query           ->", query)
    print("  flat keyword    ->", flat, "(misses the location: no shared word)")
    print("  graph k-hop     ->", graph, "(Alvaro -> Acme -> Milan surfaces it)")


def demo_persona() -> None:
    rule("4) Persona: reinforcement + time-decay + non-lossy consolidation")
    g = make_graph()
    me = g.upsert_node(OWNER, "Alvaro", "person")
    tea = g.upsert_node(OWNER, "tea", "preference")
    kombucha = g.upsert_node(OWNER, "kombucha", "preference")

    now = datetime(2026, 6, 1)
    long_ago = now - timedelta(days=400)
    g.add_edge(OWNER, me, "HAS_PREFERENCE", tea, fact="prefers tea", now=now)
    g.add_edge(OWNER, me, "HAS_PREFERENCE", tea, now=now)                 # repeat -> reinforced
    g.add_edge(OWNER, me, "HAS_PREFERENCE", kombucha, fact="liked kombucha (once)", now=long_ago)

    print("  persona before  ->", g.persona_summary_text(OWNER, now=now).replace("\n", " | "))
    retired = g.consolidate_persona(OWNER, now=now)
    print(f"  consolidate     -> retired {retired} stale low-evidence fact(s), non-lossy")
    print("  persona after   ->", g.persona_summary_text(OWNER, now=now).replace("\n", " | "))


def main() -> None:
    demo_supersession_and_time_travel()
    demo_entity_resolution()
    demo_khop_vs_flat()
    demo_persona()
    print("\nAll four behaviours run on plain in-memory SQLite: no LLM, no keys, no infra.\n")


if __name__ == "__main__":
    main()
