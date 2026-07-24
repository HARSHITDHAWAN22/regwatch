"""
Amendment graph: circulars often supersede or amend earlier ones. This
module builds a directed graph (NetworkX) of those relationships so the
system can answer "is there a newer circular that changes this?" instead
of treating every circular as if it exists in isolation.

Kept as a pure in-memory graph rebuilt from the DB (Repository-style) -
cheap to rebuild at this project's scale, and means the graph can never
drift out of sync with the source-of-truth Circular rows.
"""
import networkx as nx
from sqlalchemy.orm import Session
from app.models.circular import Circular


def build_amendment_graph(db: Session) -> nx.DiGraph:
    """Nodes = circular IDs. Edge A -> B means 'A supersedes/amends B'."""
    graph = nx.DiGraph()
    circulars = db.query(Circular).all()

    for c in circulars:
        graph.add_node(c.id, title=c.title, circular_number=c.circular_number, status=c.status)

    for c in circulars:
        if c.supersedes_id:
            graph.add_edge(c.id, c.supersedes_id, relation="supersedes")

    return graph


def find_superseding_circulars(db: Session, circular_id: str) -> list[dict]:
    """Given a circular, find any newer circular(s) that supersede it -
    used to warn 'this clause may be outdated' during impact assessment."""
    graph = build_amendment_graph(db)
    if circular_id not in graph:
        return []
    # predecessors of circular_id are nodes with an edge -> circular_id,
    # i.e. circulars that supersede this one
    superseding_ids = list(graph.predecessors(circular_id))
    return [
        {"id": nid, **graph.nodes[nid]}
        for nid in superseding_ids
    ]


def find_amendment_chain(db: Session, circular_id: str) -> list[dict]:
    """Full chain of circulars this one supersedes, oldest last -
    useful for showing a compliance officer the full history of a rule."""
    graph = build_amendment_graph(db)
    if circular_id not in graph:
        return []
    chain = []
    current = circular_id
    visited = set()
    while True:
        successors = list(graph.successors(current))  # what current supersedes
        if not successors or successors[0] in visited:
            break
        next_id = successors[0]
        visited.add(next_id)
        chain.append({"id": next_id, **graph.nodes[next_id]})
        current = next_id
    return chain
