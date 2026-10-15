"""Error-ratio ("advantage") labels and training for the chart router (``adv_router``).

Quality of a chart at a state is its ERROR against the true pose -- one number per chart, in L2 pose units:
  state (i, u):   claim q_c = D_i(u);   true pose q* = oracle.project(q_c)   (nearest on-manifold configuration)
  e_k(q*)         = || D_k(P_k(q*)) - q* ||   with P_k the exact decoder projection into chart k
  e_i (staying)   = sqrt( ||q_c - q*||^2 + e_i(q*)^2 )
  e_j (switching) = e_j(q*)
  A(i,u,j)        = log((e_i + eps) / (e_j + eps))    > 0  <=>  chart j is better than staying.
eps is the atlas's own noise floor (median best-chart projection error over the dataset).  The oracle is a
BUILD-TIME label source only; the runtime router never sees it.

State sets: (a) territory -- anchors / jitter / interpolation inside each chart; (b) overlap -- dataset poses seen
from each of their 3 best charts, where decisions are made; (c) wide jitter beyond each chart's anchors.

The in-memory entry point is ``train_router``; ``router_build.attach`` calls it on a build's ``patch_atlas.pt`` and
stores the result as ``physics_fields["router"]``.  Run from testing/manifold_charts.
"""
from __future__ import annotations

import copy
import datetime
import json
import time

import numpy as np
import torch

from atlas_rom.rom.adv_router import Net, featurize

A_CLIP = 8.0
MARGIN = float(np.log(3.0))


def jac_b(m, U):
    f = lambda z: m.decode(z.unsqueeze(0)).squeeze(0)
    return torch.func.vmap(torch.func.jacfwd(f))(U).detach()                         # (N, D, d)


def proj_batched(m, Q, U0, *, iters, lam0, tries, tol, lam_floor, damp_floor, shrink=0.3, step_stop=None):
    """Batched LM  argmin_u ||decode(u) - q||^2 .  Returns (U, cost) with cost = sum of squares per sample."""
    U = U0.clone()
    with torch.no_grad():
        R = m.decode(U) - Q
    c = (R * R).sum(-1)
    lam = torch.full_like(c, lam0)
    active = torch.ones(len(U), dtype=torch.bool, device=U.device)
    d = U.shape[1]; eye = torch.eye(d, dtype=U.dtype, device=U.device)
    for _ in range(iters):
        if not bool(active.any()):
            break
        idx = active.nonzero().flatten()
        Ua, Qa, Ra, ca, la = U[idx], Q[idx], R[idx], c[idx], lam[idx].clone()
        with torch.no_grad():
            J = jac_b(m, Ua)
            A = J.transpose(1, 2) @ J
            g = (J.transpose(1, 2) @ Ra.unsqueeze(-1)).squeeze(-1)
            tr = torch.diagonal(A, dim1=1, dim2=2).sum(-1) / d
            accepted = torch.zeros(len(idx), dtype=torch.bool, device=U.device)
            newU, newR, newc, newl, stepn = Ua.clone(), Ra.clone(), ca.clone(), la.clone(), torch.zeros_like(ca)
            for _t in range(tries):
                todo = (~accepted).nonzero().flatten()
                if len(todo) == 0:
                    break
                damp = la[todo] * tr[todo].clamp_min(damp_floor)
                st = torch.linalg.solve(A[todo] + damp[:, None, None] * eye, -g[todo].unsqueeze(-1)).squeeze(-1)
                ut = Ua[todo] + st
                rt = m.decode(ut) - Qa[todo]
                ct = (rt * rt).sum(-1)
                ok = ct < ca[todo]
                acc = todo[ok]
                newU[acc], newR[acc], newc[acc] = ut[ok], rt[ok], ct[ok]
                newl[acc] = (la[acc] * shrink).clamp_min(lam_floor)
                stepn[acc] = st[ok].norm(dim=-1)
                accepted[acc] = True
                la[todo[~ok]] *= 10.0
            done = ~accepted
            if step_stop is not None:
                done = done | (accepted & (stepn < step_stop))
            if tol is not None:
                done = done | (accepted & ((ca - newc) < tol * ca.clamp_min(1.0)))
            U[idx], R[idx], c[idx], lam[idx] = newU, newR, newc, newl
            active[idx[done]] = False
    return U, c


def proj_source(m, Q, U0):
    """Tight decoder projection: 60 LM iterations, stop at |step| < 1e-13."""
    return proj_batched(m, Q, U0, iters=60, lam0=1e-6, tries=12, tol=None, lam_floor=1e-12,
                        damp_floor=1e-14, step_stop=1e-13)


def projection_errors(M, Qstar):
    """r[:, k] = exact decoder-projection error (L2) of each target pose into chart k."""
    K = len(M)
    r = torch.zeros(len(Qstar), K, device=Qstar.device, dtype=Qstar.dtype)
    for k in range(K):
        with torch.no_grad():
            E0 = M[k].encode(Qstar)
        _U, c = proj_source(M[k], Qstar, E0)
        r[:, k] = torch.sqrt(c.detach())
    return r


def territory_states(anchors, sc, n_per, rng):
    """(a) anchors / jitter / interpolation inside each chart's own territory."""
    states = []
    for i, A in enumerate(anchors):
        for n in range(n_per):
            mode = n % 3
            b = A[rng.integers(len(A))]
            if mode == 0:
                u = b
            elif mode == 1:
                u = b + rng.standard_normal(len(b)) * sc[i] * np.exp(rng.uniform(np.log(1e-2), np.log(0.3)))
            else:
                t = rng.random(); u = (1 - t) * b + t * A[rng.integers(len(A))]
            states.append((i, np.asarray(u, float)))
    return states


def label_states(M, oracle, states, eps, dev):
    """M: per-chart models on ``dev``; states: list of (i, u).
    Returns (keep_idx, A[len(keep), K-1] ordered by j != i, e_stay)."""
    K = len(M)
    by_chart = {}
    for n, (i, u) in enumerate(states):
        by_chart.setdefault(i, []).append(n)
    A = np.full((len(states), K), np.nan)
    e_stay = np.full(len(states), np.nan)
    for i, ids in by_chart.items():
        U = torch.as_tensor(np.stack([states[n][1] for n in ids]), device=dev)
        with torch.no_grad():
            Qc = M[i].decode(U)
        Qc_np = Qc.cpu().numpy()
        qs = np.full_like(Qc_np, np.nan)
        for r_, q in enumerate(Qc_np):
            try:
                p = np.asarray(oracle.project(q), float)
                if np.linalg.norm(oracle.residual(p), np.inf) < 1e-6:
                    qs[r_] = p
            except Exception:
                pass
        ok = np.isfinite(qs).all(1)
        if not ok.any():
            continue
        Qs = torch.as_tensor(qs[ok], device=dev)
        r = projection_errors(M, Qs)
        off = (Qc[torch.as_tensor(ok, device=dev)] - Qs).norm(dim=1)
        e_i = torch.sqrt(off ** 2 + r[:, i] ** 2)
        Aij = torch.log((e_i[:, None] + eps) / (r + eps)).clamp(-A_CLIP, A_CLIP).cpu().numpy()
        idx = np.asarray(ids)[ok]
        A[idx] = Aij
        e_stay[idx] = e_i.cpu().numpy()
    keep = np.flatnonzero(np.isfinite(A).any(1))
    cols = [[j for j in range(K) if j != i] for i, _u in states]
    out = np.stack([A[n, cols[n]] for n in keep]) if len(keep) else np.zeros((0, K - 1))
    return keep, out, e_stay[keep]


def noise_floor(q_all, dev, M, n=3000, seed=0):
    q = q_all[np.random.default_rng(seed).choice(len(q_all), min(n, len(q_all)), replace=False)]
    r = projection_errors(M, torch.as_tensor(q, device=dev))
    return float(r.min(1).values.median())


def train_router(models, anchors, q_all, oracle, *, n_per=300, epochs=4000, seed=11, device="cuda", log=print):
    """Train the router net from in-memory chart models.  Returns (payload, report).

    payload = {"state", "dmax", "K", "sc", "eps"} -- everything ``adv_router.AdvRouter`` needs.  ``oracle`` is a
    build-time label source only (``check_data_validity._build_oracle(spec)``)."""
    t_start = time.time(); started_at = datetime.datetime.now().isoformat(timespec="seconds")
    timing = {}
    K = len(models)
    dev = torch.device(device)
    M = [copy.deepcopy(m).to(dev) for m in models]
    q_all = np.asarray(q_all, float)
    t0 = time.time()
    eps = noise_floor(q_all, dev, M)
    timing["noise_floor_seconds"] = time.time() - t0
    log(f"router: eps (median best-chart projection error over the dataset): {eps:.4g}")
    A_anchor = [np.asarray(a, float) for a in anchors]
    sc = [np.maximum(a.std(0), 1e-4) for a in A_anchor]
    dmax = max(len(s) for s in sc)
    rng = np.random.default_rng(seed)

    states = territory_states(A_anchor, sc, n_per, rng)
    n_a = len(states)
    qs = q_all[rng.choice(len(q_all), min(4000, len(q_all)), replace=False)]
    r_all = projection_errors(M, torch.as_tensor(qs, device=dev)).cpu().numpy()
    for q, r in zip(qs, r_all):                      # (b) overlap: each pose seen from its 3 best charts
        for i in np.argsort(r)[:3]:
            with torch.no_grad():
                u = M[i].encode(torch.as_tensor(q, device=dev).unsqueeze(0)).squeeze(0).cpu().numpy()
            states.append((int(i), u))
    n_b = len(states) - n_a
    for i in range(K):                               # (c) wide jitter: where a chart is on its way out
        for _ in range(600):
            b = A_anchor[i][rng.integers(len(A_anchor[i]))]
            states.append((i, b + rng.standard_normal(len(b)) * sc[i] * np.exp(rng.uniform(np.log(0.1), np.log(3.0)))))
    n_c = len(states) - n_a - n_b

    t0 = time.time()
    keep, Aij, _e_stay = label_states(M, oracle, states, eps, dev)
    timing["labels_seconds"] = time.time() - t0
    log(f"router: labelled {len(keep)} of {len(states)} states ({n_a} territory, {n_b} overlap, {n_c} wide) [{timing['labels_seconds']:.0f}s]")
    kind = np.array([0] * n_a + [1] * n_b + [2] * n_c)[keep]
    st = [states[k] for k in keep]

    rows, y, sid, rk = [], [], [], []
    for n, (i, u) in enumerate(st):
        for jj, j in enumerate([j for j in range(K) if j != i]):
            rows.append((i, u, j)); y.append(Aij[n, jj]); sid.append(n); rk.append(kind[n])
    y, sid, rk = np.array(y), np.array(sid), np.array(rk)
    X = np.stack([featurize(i, u, j, sc, dmax, K) for i, u, j in rows])
    held = set(rng.choice(len(st), int(0.2 * len(st)), replace=False).tolist())
    te = np.array([s in held for s in sid])
    Xt, yt = torch.as_tensor(X, device=dev), torch.as_tensor(y, device=dev)
    tr = torch.as_tensor(~te, device=dev)
    net = Net(dmax, K).double().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.time()
    for _ in range(epochs):
        opt.zero_grad()
        ((net(Xt[tr]) - yt[tr]) ** 2).mean().backward()
        opt.step(); sched.step()
    timing["train_seconds"] = time.time() - t0
    with torch.no_grad():
        pred = net(Xt).cpu().numpy()
    log(f"router: trained on {int((~te).sum())} rows [{timing['train_seconds']:.0f}s]")

    report = {"eps": eps, "n_charts": K,
              "n_states": {"territory": n_a, "overlap": n_b, "wide": n_c, "labelled": int(len(keep))},
              "n_per": n_per, "epochs": epochs, "held_out": {}}
    for lab, kk in (("territory", 0), ("overlap", 1), ("wide", 2)):
        m = te & (rk == kk)
        if m.any():
            report["held_out"][lab] = {"rows": int(m.sum()),
                                       "rmse_A": float(np.sqrt(np.mean((pred[m] - y[m]) ** 2))),
                                       "corr": float(np.corrcoef(pred[m], y[m])[0, 1]),
                                       "median_abs_A": float(np.median(np.abs(y[m])))}
    yy, pp = y.reshape(len(st), K - 1), pred.reshape(len(st), K - 1)
    h = np.array(sorted(held))
    best_ex, best_pr = yy[h].max(1), pp[h].max(1)
    pick_true = yy[h][np.arange(len(h)), pp[h].argmax(1)]
    want, switch = best_ex > MARGIN, best_pr > MARGIN
    report["held_out_decisions"] = {
        "states": int(len(h)), "margin": MARGIN,
        "exact_switch_fraction": float(want.mean()), "net_switch_fraction": float(switch.mean()),
        "switches_when_it_should": float(np.mean(switch[want])) if want.any() else None,
        "false_switch_when_exact_best_A_below_0": float(np.mean(switch[best_ex < 0])) if (best_ex < 0).any() else None,
        "pick_true_A_above_0": float(np.mean(pick_true[switch] > 0)) if switch.any() else None,
        "pick_true_A_above_half_margin": float(np.mean(pick_true[switch] > MARGIN / 2)) if switch.any() else None}
    log("router: " + json.dumps(report["held_out"])); log("router: " + json.dumps(report["held_out_decisions"]))
    timing.update(started_at=started_at, finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
                  wall_seconds=time.time() - t_start)
    report["timing"] = timing
    payload = {"state": {k: v.cpu() for k, v in net.state_dict().items()}, "dmax": dmax, "K": K, "sc": sc, "eps": eps}
    return payload, report
