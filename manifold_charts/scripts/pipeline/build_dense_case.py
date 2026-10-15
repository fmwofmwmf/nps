"""Build a denser rotate dataset from the oracle (exact poses, exact tangent frames, kNN tangent-filtered edges).
usage: build_dense.py FACTOR  ->  work/rotate_dense_x{FACTOR}/new_dataset.npz  (q, frames, local_edges, cover_edges, connectivity_only)"""
import os, sys, time, numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
ROOT = Path(__file__).resolve().parents[2]                          # manifold_charts
sys.path.insert(0, str(ROOT))
from atlas_rom.cases import load_spec
from atlas_rom.simulate import oracle_eval

FACTOR = float(sys.argv[1]); t0 = time.time()
spec = load_spec(os.environ.get("CASE", "rotate")); oracle = oracle_eval._build_oracle(spec)
import os
BASE = os.environ.get("BASE", "rotate"); PER = int(os.environ.get("PER", "0")); OUTN = os.environ.get("OUTN", "")
d = np.load(ROOT / "work" / BASE / "new_dataset.npz"); q0, F0 = d["q"], d["frames"]; n0 = len(q0)
rng = np.random.default_rng(123)
nn0 = np.median(cKDTree(q0).query(q0, k=2)[0][:, 1]); r = nn0 / np.sqrt(FACTOR)     # target spacing for density x FACTOR
print(f"original {n0} poses, NN median {nn0:.3f}; target min spacing {r:.3f}", flush=True)

def tangent_basis(x, eps=1e-6):
    cols = []
    for i in range(len(x)):
        e = np.zeros_like(x); e[i] = eps
        cols.append((oracle.residual(x + e) - oracle.residual(x - e)) / (2 * eps))
    U, S, Vt = np.linalg.svd(np.stack(cols, 1), full_matrices=True)
    rank = int((S > 1e-6 * S[0]).sum()); return Vt[rank:].T                       # (48, 2) orthonormal

# candidates: tangent jitter around each original pose, projected exactly onto the manifold
cand = []
per = PER if PER else int(np.ceil(60 * FACTOR))
for i in range(n0):
    Fi = F0[i]
    for _ in range(per):
        a = rng.standard_normal(F0.shape[2]) * (0.7 * nn0)
        try: p = np.asarray(oracle.project(q0[i] + Fi @ a), float)
        except Exception: continue
        # converged onto the constraint set AND inside every declared joint limit / workspace bound
        # (the residual check alone admitted poses past the slot ends: 28% of stephenson's dense set)
        if oracle.response_validity(p)[0]: cand.append(p)
cand = np.array(cand); print(f"{len(cand)} projected candidates [{time.time()-t0:.0f}s]", flush=True)
rng.shuffle(cand)
accn = q0.copy()
for s in range(0, len(cand), 400):
    chunk = cand[s:s + 400]
    dmin = np.sqrt(((chunk[:, None, :] - accn[None]) ** 2).sum(-1)).min(1)
    new = []
    for p, dm in zip(chunk, dmin):
        if dm <= r: continue
        if new and np.sqrt(((np.array(new) - p) ** 2).sum(-1)).min() <= r: continue
        new.append(p)
    if new: accn = np.vstack([accn, np.array(new)])
q = accn; n = len(q)
print(f"accepted {n} poses (x{n/n0:.2f}) [{time.time()-t0:.0f}s]", flush=True)

frames = np.stack([tangent_basis(x) for x in q]); print(f"frames computed [{time.time()-t0:.0f}s]", flush=True)
# edges: 6 nearest pose-space neighbours kept if the chord is within 0.5 of the tangent (as the original builder's tangent_tol)
tree = cKDTree(q); dist, nbr = tree.query(q, k=9); edges = set()
for i in range(n):
    kept = 0
    for j in nbr[i, 1:]:
        v = q[j] - q[i]; nv = np.linalg.norm(v)
        offt = np.linalg.norm(v - frames[i] @ (frames[i].T @ v)) / nv
        if offt <= 0.5 and kept < 6: edges.add((min(i, j), max(i, j))); kept += 1
edges = np.array(sorted(edges), dtype=np.int64)
G = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n))
ncomp = connected_components(G, directed=False)[0]
print(f"edges {len(edges)}, mean degree {2*len(edges)/n:.2f}, components {ncomp}", flush=True)
res = max(np.linalg.norm(oracle.residual(x), np.inf) for x in q)
nn = cKDTree(q).query(q, k=2)[0][:, 1]
print(f"max constraint residual {res:.1e}; NN spacing median {np.median(nn):.3f} (orig {nn0:.3f}); min {nn.min():.3f}", flush=True)
out = ROOT / "work" / (OUTN or f"rotate_dense_x{int(FACTOR)}"); out.mkdir(exist_ok=True)
np.savez(out / "new_dataset.npz", q=q, frames=frames, local_edges=edges, cover_edges=edges, connectivity_only=np.zeros(len(edges), bool))
print("saved", out, f"[{time.time()-t0:.0f}s]")
