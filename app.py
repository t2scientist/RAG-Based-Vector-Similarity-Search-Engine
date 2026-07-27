

import heapq
import math
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import requests
from flask import Flask, Response, jsonify, request, send_from_directory

DIMS = 16  # dimensionality of the demo vectors
# Document embeddings dimension is discovered at runtime from Ollama's output


# =====================================================================
#  DATA TYPES
# =====================================================================

@dataclass
class VectorItem:
    id: int
    metadata: str
    category: str
    emb: List[float]


DistFn = Callable[[List[float], List[float]], float]


# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a)
    nb = sum(y * y for y in b)
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(name: str) -> DistFn:
    if name == "cosine":
        return cosine
    if name == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    """The baseline: compare the query against every single vector."""

    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem) -> None:
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        results = [(dist(q, v.emb), v.id) for v in self.items]
        results.sort(key=lambda r: r[0])
        return results[:k]

    def remove(self, id_: int) -> None:
        self.items = [v for v in self.items if v.id != id_]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    """Binary space partitioning tree. Splits on one axis per level,
    cycling through dimensions. Works well for low dimensions, degrades
    towards brute force as dimensionality grows (curse of dimensionality)."""

    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def insert(self, v: VectorItem) -> None:
        self.root = self._insert(self.root, v, 0)

    def _insert(self, node: Optional[KDNode], v: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(v)
        axis = depth % self.dims
        if v.emb[axis] < node.item.emb[axis]:
            node.left = self._insert(node.left, v, depth + 1)
        else:
            node.right = self._insert(node.right, v, depth + 1)
        return node

    def knn(self, q: List[float], k: int, dist: DistFn) -> List[Tuple[float, int]]:
        # Max-heap of size k, implemented with negated distances.
        heap: List[Tuple[float, int]] = []

        def visit(node: Optional[KDNode], depth: int) -> None:
            if node is None:
                return
            dn = dist(q, node.item.emb)
            if len(heap) < k:
                heapq.heappush(heap, (-dn, node.item.id))
            elif dn < -heap[0][0]:
                heapq.heapreplace(heap, (-dn, node.item.id))

            axis = depth % self.dims
            diff = q[axis] - node.item.emb[axis]
            closer, farther = (node.left, node.right) if diff < 0 else (node.right, node.left)
            visit(closer, depth + 1)
            if len(heap) < k or abs(diff) < -heap[0][0]:
                visit(farther, depth + 1)

        visit(self.root, 0)
        result = [(-d, i) for d, i in heap]
        result.sort(key=lambda r: r[0])
        return result

    def rebuild(self, items: List[VectorItem]) -> None:
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW — Hierarchical Navigable Small World
# =====================================================================

class _HNode:
    __slots__ = ("item", "max_layer", "nbrs")

    def __init__(self, item: VectorItem, max_layer: int):
        self.item = item
        self.max_layer = max_layer
        self.nbrs: List[List[int]] = [[] for _ in range(max_layer + 1)]


class HNSW:
    """Multilayer graph. Upper layers are a sparse 'highway' that gets a
    search close to the right neighborhood fast; layer 0 holds every node
    and does the fine-grained work. Same principle used by Pinecone,
    Weaviate, Chroma and Milvus."""

    def __init__(self, m: int = 16, ef_build: int = 200):
        self.G: Dict[int, _HNode] = {}
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        self.rng = random.Random(42)

    def _rand_level(self) -> int:
        u = self.rng.random()
        while u <= 0.0:
            u = self.rng.random()
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, layer: int,
                       dist: DistFn) -> List[Tuple[float, int]]:
        visited = {ep}
        d0 = dist(q, self.G[ep].item.emb)
        candidates = [(d0, ep)]              # min-heap
        found = [(-d0, ep)]                  # max-heap (negated)

        while candidates:
            cd, cid = heapq.heappop(candidates)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = self.G.get(cid)
            if node is None or layer >= len(node.nbrs):
                continue
            for nid in node.nbrs[layer]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist(q, self.G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(candidates, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        result = [(-d, i) for d, i in found]
        result.sort(key=lambda r: r[0])
        return result

    @staticmethod
    def _select_nbrs(cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [c[1] for c in cands[:max_m]]

    def insert(self, item: VectorItem, dist: DistFn) -> None:
        id_ = item.id
        lvl = self._rand_level()
        self.G[id_] = _HNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(item.emb, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            w = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            selected = self._select_nbrs(w, max_m)
            self.G[id_].nbrs[lc] = selected

            for nid in selected:
                neighbor = self.G.get(nid)
                if neighbor is None:
                    continue
                if len(neighbor.nbrs) <= lc:
                    neighbor.nbrs.extend([] for _ in range(lc + 1 - len(neighbor.nbrs)))
                conn = neighbor.nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ranked = sorted(
                        ((dist(neighbor.item.emb, self.G[c].item.emb), c)
                         for c in conn if c in self.G)
                    )
                    neighbor.nbrs[lc] = [c for _, c in ranked[:max_m]]

            if w:
                ep = w[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q: List[float], k: int, ef: int, dist: DistFn) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                w = self._search_layer(q, ep, 1, lc, dist)
                if w:
                    ep = w[0][1]
        w = self._search_layer(q, ep, max(ef, k), 0, dist)
        return w[:k]

    def remove(self, id_: int) -> None:
        if id_ not in self.G:
            return
        for node in self.G.values():
            for layer in node.nbrs:
                if id_ in layer:
                    layer.remove(id_)
        if self.entry_pt == id_:
            self.entry_pt = -1
            for other in self.G:
                if other != id_:
                    self.entry_pt = other
                    break
        del self.G[id_]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes, edges = [], []

        for id_, node in self.G.items():
            nodes.append({
                "id": id_, "metadata": node.item.metadata,
                "category": node.item.category, "maxLyr": node.max_layer,
            })
            for lc in range(min(node.max_layer, max_l - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(node.nbrs):
                    for nid in node.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})

        return {
            "topLayer": self.top_layer, "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer, "edgesPerLayer": edges_per_layer,
            "nodes": nodes, "edges": edges,
        }

    def size(self) -> int:
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE (demo 16D index)
# =====================================================================

class VectorDB:
    """Runs all three algorithms side by side over the same 16D demo data."""

    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.lock = threading.Lock()
        self.next_id = 1

    def insert(self, meta: str, cat: str, emb: List[float], dist: DistFn) -> int:
        with self.lock:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str, algo: str) -> dict:
        with self.lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()

            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)

            us = int((time.perf_counter() - t0) * 1_000_000)

            hits = []
            for d, id_ in raw:
                v = self.store.get(id_)
                if v:
                    hits.append({"id": v.id, "meta": v.metadata, "cat": v.category,
                                 "emb": v.emb, "dist": d})
            return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self.lock:
            dfn = get_dist_fn(metric)

            def timed(fn) -> int:
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1_000_000)

            return {
                "bfUs": timed(lambda: self.bf.knn(q, k, dfn)),
                "kdUs": timed(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": timed(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "n": len(self.store),
            }

    def all(self) -> List[VectorItem]:
        with self.lock:
            return list(self.store.values())

    def hnsw_info(self) -> dict:
        with self.lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self.lock:
            return len(self.store)


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


# =====================================================================
#  OLLAMA CLIENT — wraps the local Ollama REST API
#  Install:  https://ollama.com
#  Models:   ollama pull nomic-embed-text
#            ollama pull llama3.2
# =====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except requests.RequestException:
            return False
    def embed(self, text: str):
        endpoints = [
            (
                "/api/embed",
                {"model": self.embed_model, "input": text},
                lambda d: d.get("embeddings", [None])[0],
            ),
            (
                "/api/embeddings",
                {"model": self.embed_model, "prompt": text},
                lambda d: d.get("embedding"),
            ),
        ]

        for endpoint, payload, parser in endpoints:
            try:
                r = requests.post(
                    self.base + endpoint,
                    json=payload,
                    timeout=30,
                )

                print("Endpoint:", endpoint)
                print("Status:", r.status_code)
                print("Response:", r.text)

                if r.status_code == 200:
                    return parser(r.json())

            except Exception as e:
                print("Embed error:", e)

        return None

    def generate(self, prompt: str) -> str:
        try:
            r = requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180,
            )
            if r.status_code != 200:
                return "ERROR: Ollama unavailable. Run: ollama serve"
            return r.json().get("response", "")
        except requests.RequestException:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE — HNSW over real Ollama embeddings
# =====================================================================

@dataclass
class DocItem:
    id: int
    title: str
    text: str
    emb: List[float]


class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()   # brute-force fallback while the set is small
        self.lock = threading.Lock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self.lock:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int, max_dist: float = 0.7) -> List[Tuple[float, DocItem]]:
        with self.lock:
            if not self.store:
                return []
            raw = self.bf.knn(q, k, cosine) if len(self.store) < 10 else self.hnsw.knn(q, k, 50, cosine)
            out = []
            for d, id_ in raw:
                item = self.store.get(id_)
                if item and d <= max_dist:
                    out.append((d, item))
            return out

    def remove(self, id_: int) -> bool:
        with self.lock:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self) -> List[DocItem]:
        with self.lock:
            return list(self.store.values())

    def size(self) -> int:
        with self.lock:
            return len(self.store)


# =====================================================================
#  DEMO DATA (16D categorical vectors)
#  Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
# =====================================================================

def load_demo(db: VectorDB) -> None:
    dist = get_dist_fn("cosine")
    demo = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
    ]
    for meta, cat, emb in demo:
        db.insert(meta, cat, emb, dist)


# =====================================================================
#  HTTP SERVER (Flask)
# =====================================================================

app = Flask(__name__, static_folder=None)
db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


@app.after_request
def add_cors(res: Response) -> Response:
    res.headers["Access-Control-Allow-Origin"] = "*"
    res.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    res.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return res


@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def options_handler(_any=None):
    return ("", 204)


def parse_vec(s: str) -> List[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(float(part))
        except ValueError:
            pass
    return out


# ── DEMO VECTOR ENDPOINTS ──────────────────────────────────────────

@app.route("/search", methods=["GET"])
def route_search():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})
    k = int(request.args.get("k", 5) or 5)
    metric = request.args.get("metric") or "cosine"
    algo = request.args.get("algo") or "hnsw"

    out = db.search(q, k, metric, algo)
    results = [{
        "id": h["id"], "metadata": h["meta"], "category": h["cat"],
        "distance": h["dist"], "embedding": h["emb"],
    } for h in out["hits"]]
    return jsonify({
        "results": results, "latencyUs": out["us"],
        "algo": out["algo"], "metric": out["metric"],
    })


@app.route("/insert", methods=["POST"])
def route_insert():
    body = request.get_json(silent=True) or {}
    meta = body.get("metadata", "")
    cat = body.get("category", "")
    emb = body.get("embedding", [])
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"})
    id_ = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": id_})


@app.route("/delete/<int:id_>", methods=["DELETE"])
def route_delete(id_):
    return jsonify({"ok": db.remove(id_)})


@app.route("/items", methods=["GET"])
def route_items():
    return jsonify([{
        "id": v.id, "metadata": v.metadata, "category": v.category, "embedding": v.emb,
    } for v in db.all()])


@app.route("/benchmark", methods=["GET"])
def route_benchmark():
    q = parse_vec(request.args.get("v", ""))
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"})
    k = int(request.args.get("k", 5) or 5)
    metric = request.args.get("metric") or "cosine"
    b = db.benchmark(q, k, metric)
    return jsonify({
        "bruteforceUs": b["bfUs"], "kdtreeUs": b["kdUs"],
        "hnswUs": b["hnswUs"], "itemCount": b["n"],
    })


@app.route("/hnsw-info", methods=["GET"])
def route_hnsw_info():
    return jsonify(db.hnsw_info())


# ── DOCUMENT + RAG ENDPOINTS ───────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def route_doc_insert():
    body = request.get_json(silent=True) or {}
    title = body.get("title", "")
    text = body.get("text", "")
    if not title or not text:
        return jsonify({"error": "need title and text"})

    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            })
        chunk_title = f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.dims})


@app.route("/doc/delete/<int:id_>", methods=["DELETE"])
def route_doc_delete(id_):
    return jsonify({"ok": doc_db.remove(id_)})


@app.route("/doc/list", methods=["GET"])
def route_doc_list():
    docs = []
    for d in doc_db.all():
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        docs.append({
            "id": d.id, "title": d.title, "preview": preview,
            "words": d.text.count(" ") + 1,
        })
    return jsonify(docs)


@app.route("/doc/search", methods=["POST"])
def route_doc_search():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    hits = doc_db.search(q_emb, k)
    return jsonify({"contexts": [
        {"id": item.id, "title": item.title, "distance": round(d, 4)} for d, item in hits
    ]})


@app.route("/doc/ask", methods=["POST"])
def route_doc_ask():
    body = request.get_json(silent=True) or {}
    question = body.get("question", "")
    k = int(body.get("k", 3))
    if not question:
        return jsonify({"error": "need question"})

    # Step 1: embed the question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"})

    # Step 2: retrieve top-k relevant chunks
    hits = doc_db.search(q_emb, k)

    # Step 3: build the prompt
    ctx = "".join(f"[{i + 1}] {item.title}:\n{item.text}\n\n" for i, (_, item) in enumerate(hits))
    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx}"
        f"Question: {question}\n\n"
        "Answer:"
    )

    # Step 4: generate the answer
    answer = ollama.generate(prompt)

    # Step 5: return everything
    return jsonify({
        "answer": answer, "model": ollama.gen_model,
        "contexts": [{
            "id": item.id, "title": item.title, "text": item.text, "distance": round(d, 4),
        } for d, item in hits],
        "docCount": doc_db.size(),
    })


@app.route("/status", methods=["GET"])
def route_status():
    return jsonify({
        "ollamaAvailable": ollama.is_available(),
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "docCount": doc_db.size(),
        "docDims": doc_db.dims,
        "demoDims": DIMS,
        "demoCount": db.size(),
    })


@app.route("/stats", methods=["GET"])
def route_stats():
    return jsonify({
        "count": db.size(), "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"],
    })


# ── FRONTEND ────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def route_index():
    return send_from_directory(".", "index.html")


# =====================================================================
#  ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    load_demo(db)

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    app.run(host="0.0.0.0", port=8080, threaded=True)
