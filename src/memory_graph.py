"""memory_graph.py — the persistent, owner-scoped memory graph store.

Backed by SQLite (GraphNode / GraphEdge tables) + optionally ChromaDB for
semantic node/edge search. This is the storage + traversal + temporal layer;
extraction (building the graph from conversation) lives in graph_extractor.py,
and hybrid retrieval wiring lands in Phase 3c.

Reliability principles (4-6B models):
- Closed ontology: entity types and predicates are fixed enums, validated here.
- Entity resolution is CODE (exact/alias/fuzzy), never a model judgment.
- Bi-temporal supersession is RULE-based for functional predicates: a new fact
  with the same (subject, predicate) but a different object deterministically
  invalidates the prior one. Nothing is deleted (non-lossy / auditable).
- k-hop traversal is a bounded BFS in code — no graph DB, no LLM at query time.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ── closed ontology ──

ENTITY_TYPES = frozenset({
    "person", "preference", "place", "organization", "event", "project",
    "goal", "trait", "skill", "topic", "possession", "datetime",
})

PREDICATES = frozenset({
    "HAS_PREFERENCE", "DISLIKES", "RELATED_TO", "WORKS_FOR", "WORKS_ON",
    "LIVES_IN", "LOCATED_IN", "ATTENDED", "HAS_GOAL", "HAS_TRAIT", "OWNS",
    "INTERESTED_IN", "OCCURRED_ON",
    # Identity: the user's name and how they prefer to be addressed.
    "HAS_NAME", "PREFERS_NAME",
})

# Single-valued ("functional") predicates: a subject has at most one current
# object. A new value supersedes the old deterministically — no LLM judgment.
FUNCTIONAL_PREDICATES = frozenset({
    "LIVES_IN", "WORKS_FOR", "OCCURRED_ON", "HAS_NAME", "PREFERS_NAME",
})

# Persona layer: evidence (salience) decays with a half-life; edges whose
# effective salience drops below the floor are retired (non-lossy).
PERSONA_HALF_LIFE_DAYS = 45.0
PERSONA_RETIRE_FLOOR = 0.25

# Map predicates to user-model sections for the persona summary.
_PERSONA_SECTIONS = {
    "HAS_NAME": "identity",
    "PREFERS_NAME": "identity",
    "HAS_PREFERENCE": "likes",
    "DISLIKES": "dislikes",
    "HAS_TRAIT": "traits",
    "HAS_GOAL": "goals",
    "INTERESTED_IN": "interests",
    "RELATED_TO": "relationships",
    "WORKS_FOR": "work",
    "WORKS_ON": "work",
    "LIVES_IN": "location",
}


def normalize_type(t: str) -> str:
    t = (t or "").strip().lower()
    return t if t in ENTITY_TYPES else "topic"


def normalize_predicate(p: str) -> Optional[str]:
    p = (p or "").strip().upper().replace(" ", "_")
    return p if p in PREDICATES else None


def _norm_name(name: str) -> str:
    return " ".join((name or "").strip().split()).lower()


class MemoryGraph:
    """Owner-scoped graph store. Pass a `session_factory` (defaults to the app's
    SessionLocal) and an optional `vector` index (semantic search; degrades to
    SQL/BM25 when absent)."""

    def __init__(self, session_factory=None, vector=None):
        self._session_factory = session_factory
        self.vector = vector

    def _session(self):
        if self._session_factory is not None:
            return self._session_factory()
        from core.database import SessionLocal
        return SessionLocal()

    # ── nodes ──

    def resolve_node(self, db, owner, name: str, type: Optional[str] = None) -> Optional[str]:
        """Find an existing node id for `name` (code-first entity resolution):
        exact canonical match, then alias match, then optional fuzzy match.
        Returns None if no confident match — caller creates a new node."""
        from core.database import GraphNode

        norm = _norm_name(name)
        if not norm:
            return None
        q = db.query(GraphNode).filter(GraphNode.owner == owner)
        if type:
            q = q.filter(GraphNode.type == normalize_type(type))
        candidates = q.all()

        # 1. exact canonical name
        for n in candidates:
            if _norm_name(n.name) == norm:
                return n.id
        # 2. alias match
        for n in candidates:
            for a in (n.aliases or []):
                if _norm_name(a) == norm:
                    return n.id
        # 3. optional fuzzy match (rapidfuzz), conservative threshold
        try:
            from rapidfuzz import fuzz
            best_id, best_score = None, 0.0
            for n in candidates:
                score = fuzz.WRatio(norm, _norm_name(n.name))
                if score > best_score:
                    best_id, best_score = n.id, score
            if best_id is not None and best_score >= 92.0:
                return best_id
        except Exception:  # noqa: BLE001 — rapidfuzz optional; absence is fine
            pass
        return None

    def upsert_node(
        self, owner, name: str, type: str = "topic", *,
        summary: str = "", aliases: Optional[list] = None,
        source_ref: Optional[str] = None, db=None,
    ) -> Optional[str]:
        """Resolve-or-create a node. Merges a new surface form into the existing
        node's aliases when it resolves. Returns the node id."""
        name = (name or "").strip()
        if not name:
            return None
        own_db = db is None
        db = db or self._session()
        try:
            from core.database import GraphNode

            ntype = normalize_type(type)
            existing_id = self.resolve_node(db, owner, name, ntype)
            if existing_id:
                node = db.query(GraphNode).filter(GraphNode.id == existing_id).first()
                if node is not None:
                    # Record a new surface form as an alias (keep canonical name).
                    if _norm_name(node.name) != _norm_name(name):
                        al = list(node.aliases or [])
                        if name not in al and node.name != name:
                            al.append(name)
                            node.aliases = al
                    if summary and not node.summary:
                        node.summary = summary
                    db.commit()
                return existing_id

            node_id = uuid.uuid4().hex[:16]
            node = GraphNode(
                id=node_id, owner=owner, type=ntype, name=name,
                aliases=list(aliases or []), summary=summary or "",
                confidence=1.0, salience=0.0, source_ref=source_ref,
            )
            db.add(node)
            db.commit()
            self._index_node(node_id, name, summary)
            return node_id
        finally:
            if own_db:
                db.close()

    def _index_node(self, node_id, name, summary):
        if self.vector is None:
            return
        try:
            self.vector.add(node_id, (name + " " + (summary or "")).strip())
        except Exception:  # noqa: BLE001
            logger.debug("node vector index failed for %s", node_id)

    # ── edges (bi-temporal) ──

    def add_edge(
        self, owner, subject_id: str, predicate: str, object_id: str, *,
        fact: str = "", confidence: float = 1.0,
        valid_at: Optional[datetime] = None, source_ref: Optional[str] = None,
        supersede: bool = True, db=None, now: Optional[datetime] = None,
    ) -> Optional[str]:
        """Add a fact edge. For functional predicates, deterministically
        supersedes any conflicting current edge (different object) instead of
        deleting it. Returns the new edge id, or None on invalid predicate."""
        pred = normalize_predicate(predicate)
        if pred is None:
            logger.debug("rejected edge with unknown predicate %r", predicate)
            return None
        own_db = db is None
        db = db or self._session()
        try:
            from core.database import GraphEdge

            now = now or datetime.utcnow()
            valid_at = valid_at or now

            # Reinforce an identical CURRENT fact instead of duplicating it —
            # repeated observation strengthens the persona signal (salience).
            existing = db.query(GraphEdge).filter(
                GraphEdge.owner == owner,
                GraphEdge.subject_id == subject_id,
                GraphEdge.predicate == pred,
                GraphEdge.object_id == object_id,
                GraphEdge.invalid_at.is_(None),
                GraphEdge.expired_at.is_(None),
            ).first()
            if existing is not None:
                existing.salience = (existing.salience or 1.0) + 1.0
                existing.confidence = max(existing.confidence or 0.0, confidence)
                existing.last_seen = now
                if fact and not existing.fact:
                    existing.fact = fact
                db.commit()
                return existing.id

            if supersede and pred in FUNCTIONAL_PREDICATES:
                conflicts = db.query(GraphEdge).filter(
                    GraphEdge.owner == owner,
                    GraphEdge.subject_id == subject_id,
                    GraphEdge.predicate == pred,
                    GraphEdge.object_id != object_id,
                    GraphEdge.invalid_at.is_(None),
                    GraphEdge.expired_at.is_(None),
                ).all()
                for c in conflicts:
                    c.invalid_at = valid_at      # event-time: stopped being true now
                    c.expired_at = now           # transaction-time: we retracted it now

            edge_id = uuid.uuid4().hex[:16]
            edge = GraphEdge(
                id=edge_id, owner=owner, subject_id=subject_id, predicate=pred,
                object_id=object_id, fact=fact or "", confidence=confidence,
                salience=1.0, last_seen=now,
                valid_at=valid_at, invalid_at=None, expired_at=None,
                source_ref=source_ref,
            )
            db.add(edge)
            db.commit()
            self._index_edge(edge_id, fact)
            return edge_id
        finally:
            if own_db:
                db.close()

    def _index_edge(self, edge_id, fact):
        if self.vector is None or not fact:
            return
        try:
            self.vector.add("edge:" + edge_id, fact)
        except Exception:  # noqa: BLE001
            logger.debug("edge vector index failed for %s", edge_id)

    # ── traversal ──

    @staticmethod
    def _edge_current(edge, as_of: Optional[datetime]) -> bool:
        if as_of is None:
            return edge.invalid_at is None and edge.expired_at is None
        # As-of event time: edge was valid at `as_of`.
        if edge.valid_at is not None and edge.valid_at > as_of:
            return False
        if edge.invalid_at is not None and edge.invalid_at <= as_of:
            return False
        return True

    def _edges_for(self, db, owner, node_ids, as_of):
        from core.database import GraphEdge
        if not node_ids:
            return []
        rows = db.query(GraphEdge).filter(
            GraphEdge.owner == owner,
            (GraphEdge.subject_id.in_(node_ids)) | (GraphEdge.object_id.in_(node_ids)),
        ).all()
        return [e for e in rows if self._edge_current(e, as_of)]

    def neighbors(self, owner, seed_ids, *, hops: int = 1, as_of: Optional[datetime] = None, db=None) -> dict:
        """Bounded BFS expansion from seed node ids. Returns
        {"nodes": [GraphNode...], "edges": [GraphEdge...]} for the reachable
        subgraph at the given (optional) point in event time."""
        own_db = db is None
        db = db or self._session()
        try:
            from core.database import GraphNode

            seen_nodes = set(seed_ids or [])
            frontier = set(seed_ids or [])
            edges = {}
            for _ in range(max(hops, 0)):
                if not frontier:
                    break
                hop_edges = self._edges_for(db, owner, list(frontier), as_of)
                next_frontier = set()
                for e in hop_edges:
                    edges[e.id] = e
                    for nid in (e.subject_id, e.object_id):
                        if nid not in seen_nodes:
                            seen_nodes.add(nid)
                            next_frontier.add(nid)
                frontier = next_frontier
            nodes = []
            if seen_nodes:
                nodes = db.query(GraphNode).filter(
                    GraphNode.owner == owner, GraphNode.id.in_(list(seen_nodes))
                ).all()
            return {"nodes": nodes, "edges": list(edges.values())}
        finally:
            if own_db:
                db.close()

    def graph_retrieve(self, query: str, owner, *, hops: int = 2, as_of=None,
                       limit: int = 12, db=None) -> list:
        """Hybrid graph retrieval for the reasoner (NO LLM at query time).

        Seeds nodes by content-token overlap with the query (degrades fine
        without embeddings), expands the k-hop current-truth subgraph, and
        returns reasoner-compatible hit dicts (id/text/score/source='graph')
        where text is each fact. Closer facts (touching a seed) score higher.
        """
        from src.text_tokens import content_tokens
        from core.database import GraphNode

        own_db = db is None
        db = db or self._session()
        try:
            qtoks = set(content_tokens(query))
            if not qtoks:
                return []
            nodes = db.query(GraphNode).filter(GraphNode.owner == owner).all()
            scored_seeds = []
            for n in nodes:
                ntoks = set(content_tokens(n.name))
                for a in (n.aliases or []):
                    ntoks |= set(content_tokens(a))
                overlap = len(qtoks & ntoks)
                if overlap > 0:
                    scored_seeds.append((overlap, n.id))
            if not scored_seeds:
                return []
            scored_seeds.sort(reverse=True)
            seed_ids = [nid for _, nid in scored_seeds[:8]]
            seed_set = set(seed_ids)

            sub = self.neighbors(owner, seed_ids, hops=hops, as_of=as_of, db=db)
            name_by_id = {n.id: n.name for n in sub["nodes"]}
            out = []
            for e in sub["edges"]:
                subj = name_by_id.get(e.subject_id, e.subject_id)
                obj = name_by_id.get(e.object_id, e.object_id)
                text = e.fact or f"{subj} {e.predicate.replace('_', ' ').lower()} {obj}"
                close = 0.75 if (e.subject_id in seed_set or e.object_id in seed_set) else 0.5
                out.append({"id": "graph:" + e.id, "text": text, "score": close, "source": "graph"})
            out.sort(key=lambda h: h["score"], reverse=True)
            return out[:limit]
        finally:
            if own_db:
                db.close()

    def current_edges(self, owner, *, subject_id=None, predicate=None, db=None) -> list:
        """All currently-true edges for an owner, optionally filtered."""
        own_db = db is None
        db = db or self._session()
        try:
            from core.database import GraphEdge
            q = db.query(GraphEdge).filter(
                GraphEdge.owner == owner,
                GraphEdge.invalid_at.is_(None),
                GraphEdge.expired_at.is_(None),
            )
            if subject_id:
                q = q.filter(GraphEdge.subject_id == subject_id)
            if predicate:
                p = normalize_predicate(predicate)
                if p:
                    q = q.filter(GraphEdge.predicate == p)
            return q.all()
        finally:
            if own_db:
                db.close()

    # ── persona layer (user model: confidence/decay) ──

    @staticmethod
    def _effective_salience(edge, now: datetime, half_life: float = PERSONA_HALF_LIFE_DAYS) -> float:
        """Stored evidence weight decayed by time since it was last reinforced."""
        base = edge.salience if edge.salience is not None else 1.0
        anchor = edge.last_seen or edge.valid_at or edge.created_at
        if anchor is None:
            return base
        elapsed_days = max((now - anchor).total_seconds() / 86400.0, 0.0)
        return base * (0.5 ** (elapsed_days / max(half_life, 0.1)))

    def get_persona(self, owner, *, now: Optional[datetime] = None,
                    limit_per_section: int = 8, db=None) -> dict:
        """The user model derived from the current graph: facts grouped into
        persona sections (likes/dislikes/traits/goals/...), each sorted by
        time-decayed salience (strongest, most-recent first)."""
        own_db = db is None
        db = db or self._session()
        try:
            from core.database import GraphNode
            now = now or datetime.utcnow()
            edges = self.current_edges(owner, db=db)
            node_ids = {e.object_id for e in edges} | {e.subject_id for e in edges}
            names = {}
            if node_ids:
                for n in db.query(GraphNode).filter(
                    GraphNode.owner == owner, GraphNode.id.in_(list(node_ids))
                ).all():
                    names[n.id] = n.name
            sections: dict = {}
            for e in edges:
                sec = _PERSONA_SECTIONS.get(e.predicate)
                if not sec:
                    continue
                sections.setdefault(sec, []).append({
                    "object": names.get(e.object_id, e.object_id),
                    "fact": e.fact or "",
                    "salience": self._effective_salience(e, now),
                    "predicate": e.predicate,
                })
            for sec in sections:
                sections[sec].sort(key=lambda x: x["salience"], reverse=True)
                sections[sec] = sections[sec][:limit_per_section]
            return sections
        finally:
            if own_db:
                db.close()

    def get_name(self, owner, *, db=None) -> dict:
        """The user's learned identity from the graph: {"name", "preferred"}
        (either may be absent). '' / missing when never stated."""
        own_db = db is None
        db = db or self._session()
        try:
            from core.database import GraphNode
            # Resolve PER PREDICATE, not per object_id: entity resolution can
            # collapse identical surface forms ("mi chiamo Alvaro" /
            # "chiamami Alvaro") to ONE node, so HAS_NAME and PREFERS_NAME may
            # share an object_id. Keying by object_id would let one clobber the
            # other; key by predicate and keep the highest-salience current edge.
            best = {}  # predicate -> chosen edge
            for e in self.current_edges(owner, db=db):
                if e.predicate not in ("HAS_NAME", "PREFERS_NAME"):
                    continue
                cur = best.get(e.predicate)
                if cur is None or (e.salience or 0.0) > (cur.salience or 0.0):
                    best[e.predicate] = e
            if not best:
                return {}
            node_ids = {e.object_id for e in best.values()}
            names = {
                n.id: n.name for n in db.query(GraphNode).filter(
                    GraphNode.owner == owner, GraphNode.id.in_(list(node_ids))
                ).all()
            }
            out = {}
            key = {"HAS_NAME": "name", "PREFERS_NAME": "preferred"}
            for pred, e in best.items():
                if e.object_id in names:
                    out[key[pred]] = names[e.object_id]
            return out
        finally:
            if own_db:
                db.close()

    def persona_summary_text(self, owner, *, now: Optional[datetime] = None,
                             max_items: int = 12, exclude_sections=(), db=None) -> str:
        """Compact, salience-ranked persona block to ground the assistant's
        replies in the user's stable preferences/traits. `exclude_sections` drops
        whole sections (e.g. "identity") — the bot's reply path surfaces the
        user's name via a structured prompt block, NOT these 2nd-person facts,
        which a small model otherwise mistakes for its own identity."""
        persona = self.get_persona(owner, now=now, db=db)
        if not persona:
            return ""
        for sec in exclude_sections:
            persona.pop(sec, None)
        if not persona:
            return ""
        items = []
        for sec, rows in persona.items():
            for r in rows:
                fact = r["fact"] or f"{sec}: {r['object']}"
                items.append((r["salience"], fact))
        items.sort(key=lambda x: x[0], reverse=True)
        seen, lines = set(), []
        for _, fact in items:
            if fact and fact.lower() not in seen:
                seen.add(fact.lower())
                lines.append(fact)
            if len(lines) >= max_items:
                break
        return "\n".join(lines)

    def consolidate_persona(self, owner, *, now: Optional[datetime] = None,
                            half_life: float = PERSONA_HALF_LIFE_DAYS,
                            floor: float = PERSONA_RETIRE_FLOOR, db=None) -> int:
        """Retire (non-lossy) stale, low-evidence preference/trait edges whose
        decayed salience has fallen below the floor. Functional state edges
        (location/employer) are never auto-retired — they persist until
        superseded. Returns the number retired."""
        own_db = db is None
        db = db or self._session()
        try:
            from core.database import GraphEdge
            now = now or datetime.utcnow()
            edges = db.query(GraphEdge).filter(
                GraphEdge.owner == owner,
                GraphEdge.invalid_at.is_(None),
                GraphEdge.expired_at.is_(None),
            ).all()
            retired = 0
            for e in edges:
                if e.predicate in FUNCTIONAL_PREDICATES:
                    continue
                if self._effective_salience(e, now, half_life) < floor:
                    e.invalid_at = now
                    e.expired_at = now
                    retired += 1
            if retired:
                db.commit()
            return retired
        finally:
            if own_db:
                db.close()
