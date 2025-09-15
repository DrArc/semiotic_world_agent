# rag_semio.py
from __future__ import annotations
import os, re
from pathlib import Path
from typing import List

# Minimal, dependency-light RAG.
# Prefers scikit-learn TF-IDF; falls back to simple keyword scoring.

def _load_docs(root: str) -> List[str]:
    rootp = Path(root)
    docs = []
    if rootp.exists():
        for p in rootp.rglob("*.md"):
            try:
                docs.append(p.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                pass
    # Built-in seed corpus (semiotics + architecture)
    if not docs:
        docs = [DEFAULT_SEMIO, DEFAULT_ARCH, DEFAULT_PROMPTS]
    return docs

def _tfidf_retrieve(query: str, docs: List[str], k: int) -> str:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return _keyword_retrieve(query, docs, k)

    vec = TfidfVectorizer(stop_words="english", max_features=8192)
    X = vec.fit_transform(docs + [query])
    qv = X[-1]
    D = X[:-1]
    sim = cosine_similarity(qv, D).ravel()
    idx = sim.argsort()[::-1][:k]
    return "\n\n---\n\n".join([docs[i] for i in idx])

def _keyword_retrieve(query: str, docs: List[str], k: int) -> str:
    # crude: rank by count of query tokens
    toks = [t for t in re.split(r"\W+", query.lower()) if t]
    scores = []
    for i, d in enumerate(docs):
        s = 0
        dl = d.lower()
        for t in toks:
            s += dl.count(t)
        scores.append((s, i))
    idx = [i for _, i in sorted(scores, reverse=True)[:k]]
    return "\n\n---\n\n".join([docs[i] for i in idx])

def retrieve_context(query: str, top_k: int = 5, root: str = "rag") -> str:
    docs = _load_docs(root)
    if not docs:
        return ""
    return _tfidf_retrieve(query, docs, top_k)

# ---- seed corpus ----
DEFAULT_SEMIO = """
# Semiotic Lens (Barthes, architecture-adapted)
- **Denotation**: literal description of forms (plan, volume, façade ordering, circulation).
- **Connotation**: cultural associations (domesticity, monumentality, transparency, surveillance).
- **Myth**: governing narrative (progress, sustainability theater, techno-utopia, vernacular return).
- **Signifiers**: materials, joints, apertures, massing, rhythm, ornament, lighting.
- **Codes**: tectonic honesty, machine aesthetic, parametric expressiveness, postmodern quotation.
"""

DEFAULT_ARCH = """
# Architecture Vocab Cheatsheet
- **Materials**: cast-in-place concrete, board-formed concrete, weathering steel, TGI-glass fin curtain wall, brick soldier course, rammed earth.
- **Styles**: brutalist, high-tech, neo-rationalist, critical regionalist, parametricism, metabolic, supertall corporate modern.
- **Typologies**: courtyard housing, rowhouse, slab tower, big box retrofit, transit hub, cultural center.
- **Spatial devices**: enfilade, void atrium, double-skin façade, pilotis, megastructure, podium + tower.
- **Metrics**: FAR, porosity, daylight factor, thermal mass, embodied carbon, surface-to-volume ratio.
"""

DEFAULT_PROMPTS = """
# Prompt patterns (architectural, semiotic-forward)
- "axono at 45°, {material palette}, {light mood}, denotation/connotation split: {pair}, myth: {story}"
- "elevational study, {style}, tectonic emphasis on {joint/connection}, legibility over spectacle"
- "urban fragment, {typology}, interfaces coded as {code words}, foreground the signifiers"
"""