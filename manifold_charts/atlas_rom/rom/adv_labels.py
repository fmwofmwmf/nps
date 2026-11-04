"""Error-ratio ("advantage") labels and training for the chart router (``adv_router``).

Quality of a chart at a state is its ERROR against the true pose -- one number per chart, in L2 pose units:
  state (i, u):   claim q_c = D_i(u);   true pose q* = oracle.project(q_c)   (nearest on-manifold configuration)
  e_k             = || D_k(P_k(q_c)) - q* ||   with P_k the exact decoder projection into chart k.  Every chart,
                    the source included, is scored on the CLAIM a handoff actually passes (not on q*, which would
                    hand switching the oracle's correction for free and make it look better in both directions)
  score target    s_k = -log(e_k + eps);   A(i,u,j) = s_j - s_i = log((e_i + eps) / (e_j + eps))
                    > 0  <=>  chart j is better than staying.
eps is the atlas's own noise floor (median best-chart projection error over the dataset).  The oracle is a
BUILD-TIME label source only; the runtime router never sees it.

State sets: (a) territory -- anchors / jitter / interpolation inside each chart; (b) overlap -- dataset poses seen
from each of their 3 best charts, where decisions are made; (c) wide jitter beyond each chart's anchors; (d) exit --
states each chart's own learned dynamics reach past its anchors (``exit_states``).  Off-territory (c + d) and
on-territory (a + b) states carry equal total weight in the fit.

The in-memory entry point is ``train_router``; ``router_build.attach`` calls it on a build's ``patch_atlas.pt`` and
stores the result as ``physics_fields["router"]``.  Run from testing/manifold_charts.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import json
import os
import time

import numpy as np
import torch

from atlas_rom.rom.adv_router import ScoreNet, featurize_state

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


def exit_states(models, mass_fields, potential_fields, anchors, sc, case, body_order, rng, *, n_roll=200,
                speed=(0.5, 5.0), record_every=20, max_steps=2000, near=1.0, far=10.0):
    """(d) states each chart's OWN learned dynamics reach past the edge of its anchors.

    Batched copy of ``PatchAtlasRuntime._integrate_chart`` (semi-implicit Euler on the learned mass / potential, joint
    torques through the decoder; no routing, no boundary field): ``n_roll`` rollouts per chart from random anchors with
    a random latent velocity (anchor-std x U(speed) per second) and a constant control drawn from each actuator's
    [lo, hi].  A state is recorded every ``record_every`` steps once it is more than ``near`` anchor spacings from the
    chart's anchors; a rollout ends at ``far`` spacings, a non-finite state, or ``max_steps``.  Build time only."""
    from scipy.spatial import cKDTree
    from atlas_rom.cases import load_spec
    from atlas_rom.rom.ops import joint_angle_torch
    spec = load_spec(case)
    scen = spec.scenarios.get("default") or (next(iter(spec.scenarios.values())) if spec.scenarios else {})
    dt = float(scen.get("dt", 1e-3))
    acts = [(spec.mechanism.joint(a.target), float(a.gain), float(a.lo), float(a.hi))
            for a in spec.actuators if a.kind == "joint_torque"]
    states = []
    for i, (model, mass, pot) in enumerate(zip(models, mass_fields, potential_fields)):
        A = np.asarray(anchors[i], float); d = A.shape[1]
        tree = cKDTree(A); spacing = float(np.median(tree.query(A, k=2)[0][:, 1]))
        u = torch.as_tensor(A[rng.integers(len(A), size=n_roll)])
        dirs = rng.standard_normal((n_roll, d)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        udot = torch.as_tensor(dirs * sc[i] * rng.uniform(*speed, size=(n_roll, 1)))
        tau = torch.as_tensor(np.stack([rng.uniform(lo, hi, n_roll) * g for _j, g, lo, hi in acts], 1)
                              if acts else np.zeros((n_roll, 0)))
        with torch.no_grad():
            p = (mass(u) @ udot.unsqueeze(-1)).squeeze(-1)
        alive = torch.ones(n_roll, dtype=torch.bool)
        u_start = u.clone(); eye = torch.eye(d, dtype=u.dtype)
        for step in range(1, max_steps + 1):
            x = u.clone().requires_grad_(True)
            Mx = mass(x)
            # A learned mass that has lost positive definiteness is exactly a state the router must leave:
            # record it as an exit state, end that rollout, and keep the batch solvable.
            Md = Mx.detach(); finite = torch.isfinite(Md).all((-2, -1))
            ev = torch.linalg.eigvalsh(torch.where(finite[:, None, None], Md, eye.expand_as(Md)))
            singular = ~finite | (ev[:, 0] <= 1e-9 * ev[:, -1].abs())
            if singular.any():
                for r in torch.nonzero(singular & alive).flatten().tolist():
                    states.append((i, u[r].numpy().copy()))
                alive &= ~singular
                Mx = torch.where(singular[:, None, None], eye.expand_as(Mx), Mx)
            Hk = 0.5 * (p.unsqueeze(1) @ torch.linalg.solve(Mx, p.unsqueeze(-1))).sum()
            total = Hk + pot(x).sum()
            if acts:
                T = model.transforms(x)
                total = total - (tau * torch.stack([joint_angle_torch(T, j, body_order) for j, *_ in acts], 1)).sum()
            (g,) = torch.autograd.grad(total, x)
            with torch.no_grad():
                p = p - dt * g
                u = u + dt * torch.linalg.solve(Mx.detach(), p.unsqueeze(-1)).squeeze(-1)
                alive &= torch.isfinite(u).all(1) & torch.isfinite(p).all(1)
            if step % record_every == 0:
                un = u.numpy(); dist = np.full(n_roll, np.inf)
                ok = alive.numpy()
                dist[ok] = tree.query(un[ok])[0] / spacing
                for r in np.flatnonzero(ok & (dist > near) & (dist <= far)):
                    states.append((i, un[r].copy()))
                alive &= torch.as_tensor(dist <= far)
                if not alive.any():
                    break
            u = torch.where(alive.unsqueeze(1), u, u_start); p = torch.where(alive.unsqueeze(1), p, torch.zeros_like(p))
    return states


def claim_errors(M, Qc, Qs):
    """e[:, k] = || D_k(P_k(q_c)) - q* ||: chart k's exact projection of the CLAIM q_c (what a handoff actually
    passes), scored against the true pose q*.  For the source chart P_i(q_c) ~ u, so e_i is its own drift.
    Also returns each chart's projected latent U[k] (the same configuration in chart k's coordinates)."""
    e = torch.zeros(len(Qc), len(M), device=Qc.device, dtype=Qc.dtype)
    U = []
    for k, m in enumerate(M):
        with torch.no_grad():
            E0 = m.encode(Qc)
        Uk, _c = proj_source(m, Qc, E0)
        with torch.no_grad():
            e[:, k] = (m.decode(Uk) - Qs).norm(dim=1)
        U.append(Uk.detach())
    return e, U


_DYN_CHUNK = 8192


def dynamics_mismatch(models, mass_fields, potential_fields, bodies, latents, *, device=None):
    """Per chart k at latent u (each row of ``latents[k]``): how far chart k's LEARNED dynamics fields are from the
    teacher they were distilled from (``learned.fit_dynamics``), evaluated where the runtime would actually be.

      delta_M      = max |log lambda|, lambda the eigenvalues of M_teacher^-1 M_learned   (kinetic-energy error factor)
      f_err, f_ref = || grad V_learned - grad V_teacher ||_{M^-1},  || grad V_teacher ||_{M^-1}   (accelerations)

    The fields are only fit at anchors; off them they extrapolate, and a chart whose poses still look right can
    carry badly wrong dynamics there.  Build-time only: the teacher is never part of the runtime.

    Batched over each chart's rows (``ops.mass_matrix_batch`` / ``ops.potential_and_grad_batch``, chunks of
    ``_DYN_CHUNK``) on ``device`` (default CPU; ``nn.Module`` models / fields are copied there, plain callables are
    used as given); same quantities as the per-row loop it replaces."""
    from atlas_rom.rom import ops
    masses, inertia, com = bodies["masses"], bodies["inertia"], bodies["com"]
    up, gravity = bodies["up"], bodies["gravity"]
    dev = torch.device("cpu" if device is None else device)
    _on = lambda f: copy.deepcopy(f).double().to(dev).eval() if isinstance(f, torch.nn.Module) else f
    out = []
    for k, U in enumerate(latents):
        U = np.asarray(U, float)
        if len(U) == 0:
            out.append(np.zeros((0, 3))); continue
        model, mass, pot = _on(models[k]), _on(mass_fields[k]), _on(potential_fields[k])
        rows = []
        for lo in range(0, len(U), _DYN_CHUNK):
            Ut = torch.as_tensor(U[lo:lo + _DYN_CHUNK], dtype=torch.float64, device=dev)
            Mt = ops.mass_matrix_batch(model, Ut, masses, inertia, com)                   # (n, d, d)
            with torch.no_grad():
                Ml = mass(Ut)
            lam = torch.linalg.eigvals(torch.linalg.solve(Mt, Ml)).real
            d_m = lam.clamp_min(1e-12).log().abs().amax(-1)
            _v, gt = ops.potential_and_grad_batch(model, Ut, masses, com, up, gravity)    # (n, d)
            x = Ut.detach().clone().requires_grad_(True)
            (gl,) = torch.autograd.grad(pot(x).sum(), x)
            gl = gl.detach()
            norm = lambda f: torch.sqrt((f * torch.linalg.solve(Mt, f.unsqueeze(-1)).squeeze(-1)).sum(-1).clamp_min(0.0))
            rows.append(torch.stack([d_m, norm(gl - gt), norm(gt)], -1).cpu().numpy())
        del model, mass, pot
        out.append(np.concatenate(rows, 0).reshape(-1, 3))
    return out


_WORKER_ORACLE = None


def _init_oracle_worker(case):
    """Process-pool initializer: the oracle is not picklable, but it builds from the case name in ~0 s."""
    global _WORKER_ORACLE
    from atlas_rom.cases import load_spec
    from atlas_rom.simulate import oracle_eval
    _WORKER_ORACLE = oracle_eval._build_oracle(load_spec(case))


def _project_rows(X, oracle, starts=None, max_steps=None):
    """q* = oracle.project(q) per row, NaN where it fails or does not converge onto the constraint set.
    The convergence tolerance is the oracle's own ``residual_tol`` (default 1e-6, exact-joint oracles): a
    penalty-joint backend declares the closure gap its joints actually hold.

    With ``starts`` (one feasible pose per row) an oracle that has ``project_from`` walks each claim from
    its own start, within ``max_steps``, instead of from wherever the previous row left the scene."""
    out = np.full_like(X, np.nan)
    tol = float(getattr(oracle, "residual_tol", 1e-6))
    warm = starts is not None and hasattr(oracle, "project_from")
    for r_, q in enumerate(X):
        try:
            p = np.asarray(oracle.project_from(starts[r_], q, max_steps=max_steps) if warm
                           else oracle.project(q), float)
            if np.linalg.norm(oracle.residual(p), np.inf) < tol:
                out[r_] = p
        except Exception:
            pass
    return out


def _project_chunk(task):
    X, starts, max_steps = task
    return _project_rows(X, _WORKER_ORACLE, starts, max_steps)


def project_claims(X, oracle, *, case=None, workers=None, starts=None, max_steps=None):
    """Project every claim row onto the true mechanism.  With ``case`` (and more than one worker) the rows are split
    across CPU processes, each building its own oracle; results are identical to the serial loop."""
    import os
    X = np.asarray(X, float)
    workers = (os.cpu_count() or 1) if workers is None else int(workers)
    if case is None or workers <= 1 or len(X) < 2 * workers:
        return _project_rows(X, oracle, starts, max_steps)
    from concurrent.futures import ProcessPoolExecutor
    n_split = workers * 4
    chunks = list(zip(np.array_split(X, n_split),
                      np.array_split(np.asarray(starts, float), n_split) if starts is not None else [None] * n_split,
                      [max_steps] * n_split))
    # one BLAS thread per worker: the oracle's dense solves would otherwise oversubscribe the cores
    keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: "1" for k in keys})
    try:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_oracle_worker, initargs=(case,)) as pool:
            parts = list(pool.map(_project_chunk, chunks))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return np.concatenate(parts, axis=0)


def _nearest_starts(claims, start_pool, start_weigh):
    """Each claim's nearest start pose, measured in ``start_weigh``'s coordinates."""
    from scipy.spatial import cKDTree
    weigh = start_weigh or (lambda rows: rows)
    _d, idx = cKDTree(np.asarray(weigh(start_pool), float)).query(np.asarray(weigh(claims), float))
    return np.asarray(start_pool, float)[idx]


def label_states(M, oracle, states, eps, dev, *, case=None, workers=None, start_pool=None, start_weigh=None,
                 max_steps=None):
    """M: per-chart models on ``dev``; states: list of (i, u).
    Returns (keep_idx, E[len(keep), K] per-chart claim errors, U_map[len(keep)][K] each chart's latent of the claim)."""
    K = len(M)
    by_chart = {}
    for n, (i, u) in enumerate(states):
        by_chart.setdefault(i, []).append(n)
    E = np.full((len(states), K), np.nan)
    U_map = [None] * len(states)
    claims = {}
    for i, ids in by_chart.items():
        U = torch.as_tensor(np.stack([states[n][1] for n in ids]), device=dev)
        with torch.no_grad():
            claims[i] = M[i].decode(U)
    # all claims projected in one (parallel) pass: the oracle is the cost
    order = list(by_chart)
    X = np.concatenate([claims[i].cpu().numpy() for i in order], axis=0)
    starts = None if start_pool is None else _nearest_starts(X, start_pool, start_weigh)
    q_all = project_claims(X, oracle, case=case, workers=workers, starts=starts, max_steps=max_steps)
    splits = np.cumsum([len(by_chart[i]) for i in order])[:-1]
    projected = dict(zip(order, np.split(q_all, splits)))
    for i, ids in by_chart.items():
        Qc = claims[i]
        qs = projected[i]
        ok = np.isfinite(qs).all(1)
        if not ok.any():
            continue
        Qs = torch.as_tensor(qs[ok], device=dev)
        e, U = claim_errors(M, Qc[torch.as_tensor(ok, device=dev)], Qs)
        idx = np.asarray(ids)[ok]
        E[idx] = e.cpu().numpy()
        for r_, n in enumerate(idx):
            U_map[n] = [U[k][r_].cpu().numpy() for k in range(K)]
    keep = np.flatnonzero(np.isfinite(E).all(1))
    return keep, E[keep], [U_map[n] for n in keep]


# ---- stage checkpoints -------------------------------------------------------------------------------------------
# A router build is ~1 h of oracle labels on a contact case.  Each stage is saved to ``checkpoint_dir`` when it
# finishes and loaded instead of recomputed on restart; ``meta.json`` fingerprints the inputs so a stale directory
# is refused, never silently reused.

def _ck_save(path, obj):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)                 # atomic: a kill mid-write never leaves a truncated checkpoint


def _ck_load(path):
    return torch.load(path, map_location="cpu", weights_only=False) if os.path.isfile(path) else None


def _fingerprint(models, q_all, settings):
    h = hashlib.sha256()
    for m in models:
        for k, v in sorted(m.state_dict().items()):
            h.update(k.encode()); h.update(v.detach().cpu().double().numpy().tobytes())
    h.update(np.ascontiguousarray(q_all, float).tobytes())
    h.update(json.dumps(settings, sort_keys=True).encode())
    return h.hexdigest()


def _ck_open(checkpoint_dir, models, q_all, settings, log):
    if checkpoint_dir is None:
        return None
    os.makedirs(checkpoint_dir, exist_ok=True)
    meta = {"fingerprint": _fingerprint(models, q_all, settings), "settings": settings}
    path = os.path.join(checkpoint_dir, "meta.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            old = json.load(f)
        if old.get("fingerprint") != meta["fingerprint"]:
            raise RuntimeError(f"{checkpoint_dir}: checkpoints belong to different charts/dataset/settings "
                               f"({old.get('settings')} vs {settings}); delete the directory to start fresh")
        log(f"router: resuming from checkpoints in {checkpoint_dir}")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    return checkpoint_dir


def label_states_chunked(M, oracle, states, eps, dev, *, case=None, workers=None, ck=None, chunk=4000, log=print,
                         start_pool=None, start_weigh=None, max_steps=None):
    """``label_states`` in chunks of ``chunk`` states, each saved to ``ck`` as it finishes.  Rows are labelled
    independently, so the result equals one ``label_states`` call."""
    keeps, Es, Us = [], [], []
    n_chunks = (len(states) + chunk - 1) // chunk
    for c in range(n_chunks):
        lo = c * chunk
        path = None if ck is None else os.path.join(ck, f"labels_{c:03d}.pt")
        got = None if path is None else _ck_load(path)
        if got is None:
            t0 = time.time()
            keep, E, U_map = label_states(M, oracle, states[lo:lo + chunk], eps, dev, case=case, workers=workers,
                                          start_pool=start_pool, start_weigh=start_weigh, max_steps=max_steps)
            got = {"keep": keep + lo, "E": E, "U_map": U_map}
            if path is not None:
                _ck_save(path, got)
            log(f"router: labels chunk {c + 1}/{n_chunks} [{time.time() - t0:.0f}s]")
        keeps.append(np.asarray(got["keep"], int)); Es.append(got["E"]); Us.extend(got["U_map"])
    K = len(M)
    return (np.concatenate(keeps) if keeps else np.zeros(0, int),
            np.concatenate(Es, axis=0) if Es else np.zeros((0, K)), Us)


def noise_floor(q_all, dev, M, n=3000, seed=0):
    q = q_all[np.random.default_rng(seed).choice(len(q_all), min(n, len(q_all)), replace=False)]
    r = projection_errors(M, torch.as_tensor(q, device=dev))
    return float(r.min(1).values.median())


def _sample_states(models, M, A_anchor, sc, q_all, n_per, rng, dev, K, timing, log,
                   mass_fields, potential_fields, bodies, case, exit_rollouts):
    """The four router state sets (territory, overlap, wide, exit); returns (states, (n_a, n_b, n_c, n_d))."""
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
        for _ in range(2000):
            b = A_anchor[i][rng.integers(len(A_anchor[i]))]
            states.append((i, b + rng.standard_normal(len(b)) * sc[i] * np.exp(rng.uniform(np.log(0.1), np.log(3.0)))))
    n_c = len(states) - n_a - n_b
    n_d = 0
    if mass_fields is not None and potential_fields is not None and bodies is not None and case is not None:
        t0 = time.time()                             # (d) exit states: where each chart's own dynamics leave its anchors
        cpu_models = [copy.deepcopy(m).cpu().double().eval() for m in models]
        ex = exit_states(cpu_models, mass_fields, potential_fields, A_anchor, sc, case, list(bodies["names"]), rng,
                         n_roll=int(exit_rollouts))
        states.extend(ex); n_d = len(ex)
        timing["exit_states_seconds"] = time.time() - t0
        log(f"router: {n_d} exit states from each chart's own learned dynamics [{timing['exit_states_seconds']:.0f}s]")

    return states, (n_a, n_b, n_c, n_d)


def train_router(models, anchors, q_all, oracle, *, n_per=300, epochs=4000, seed=11, device="cuda", log=print,
                 mass_fields=None, potential_fields=None, bodies=None, case=None, workers=None,
                 exit_rollouts=200, checkpoint_dir=None, label_chunk=4000, ckpt_every=250,
                 warm_start=False, start_weigh=None, label_max_steps=None):
    """Train the router net from in-memory chart models.  Returns (payload, report).

    payload = {"kind", "state", "dmax", "K", "sc", "eps"} -- everything ``adv_router.AdvRouter`` needs.  ``oracle`` is a
    build-time label source only (``check_data_validity._build_oracle(spec)``).  With the chart's learned
    ``mass_fields`` / ``potential_fields`` and the case ``bodies`` (masses, inertia, com, up, gravity), each chart's
    score also pays for its dynamics mismatch: s_k = -log(e_k + eps) - delta_M - log(1 + delta_V) (see
    ``dynamics_mismatch``); without them the labels are pose-only.

    ``checkpoint_dir``: save each stage (states, label chunks, dynamics labels, training every ``ckpt_every``
    epochs) there and resume from whatever is already saved."""
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
    use_dyn = mass_fields is not None and potential_fields is not None and bodies is not None
    ck = _ck_open(checkpoint_dir, models, q_all,
                  {"n_per": int(n_per), "epochs": int(epochs), "seed": int(seed), "exit_rollouts": int(exit_rollouts),
                   "label_chunk": int(label_chunk), "dynamics": bool(use_dyn), "case": case}, log)
    saved = None if ck is None else _ck_load(os.path.join(ck, "states.pt"))
    if saved is not None:
        states, (n_a, n_b, n_c, n_d) = saved["states"], saved["counts"]
        rng.bit_generator.state = saved["rng"]
        log(f"router: loaded {len(states)} states from checkpoint")
    else:
        states, (n_a, n_b, n_c, n_d) = _sample_states(models, M, A_anchor, sc, q_all, n_per, rng, dev, K, timing, log,
                                                      mass_fields, potential_fields, bodies, case, exit_rollouts)
        if ck is not None:
            _ck_save(os.path.join(ck, "states.pt"),
                     {"states": states, "counts": (n_a, n_b, n_c, n_d), "rng": rng.bit_generator.state})

    t0 = time.time()
    # Warm start: each claim is projected from its nearest dataset pose (in ``start_weigh``'s metric), so
    # the oracle walks only the claim's own distance from the data, within ``label_max_steps``.
    keep, E, U_map = label_states_chunked(M, oracle, states, eps, dev, case=case, workers=workers, ck=ck,
                                          chunk=int(label_chunk), log=log,
                                          start_pool=q_all if warm_start else None, start_weigh=start_weigh,
                                          max_steps=label_max_steps)
    timing["labels_seconds"] = time.time() - t0
    log(f"router: labelled {len(keep)} of {len(states)} states ({n_a} territory, {n_b} overlap, {n_c} wide, {n_d} exit) [{timing['labels_seconds']:.0f}s]")
    kind = np.array([0] * n_a + [1] * n_b + [2] * n_c + [3] * n_d)[keep]
    st = [states[k] for k in keep]
    S = len(st)

    # Dynamics mismatch of every chart's learned fields at the latent where it would hold this claim.
    dM, dV = np.zeros_like(E), np.zeros_like(E)
    g_ref = None
    saved = None if ck is None or not use_dyn else _ck_load(os.path.join(ck, "dynamics.pt"))
    if saved is not None:
        dM, dV, g_ref = saved["dM"], saved["dV"], saved["g_ref"]
        log("router: loaded dynamics labels from checkpoint")
    elif use_dyn:
        t0 = time.time()
        cpu = [copy.deepcopy(m).cpu().double().eval() for m in models]
        ref = dynamics_mismatch(cpu, mass_fields, potential_fields, bodies, A_anchor, device=dev)
        g_ref = float(np.median(np.concatenate([r[:, 2] for r in ref])))     # typical gravity acceleration
        per = dynamics_mismatch(cpu, mass_fields, potential_fields, bodies,
                                [np.stack([U_map[n][k] for n in range(S)]) for k in range(K)], device=dev)
        for k in range(K):
            dM[:, k] = per[k][:, 0]
            # A RELATIVE force error needs a nonzero teacher force.  With no
            # potential at all (gravity 0) every teacher force is exactly zero,
            # and dividing the learned field's residual gradient by the 1e-12
            # floor made delta_V ~1e8 at every chart -- every score clipped to
            # -A_CLIP, every advantage 0, and a router that never hands off.
            if g_ref > 0.0:
                dV[:, k] = per[k][:, 1] / (per[k][:, 2] + g_ref)
        timing["dynamics_labels_seconds"] = time.time() - t0
        log(f"router: dynamics labels (mass + force mismatch vs the fields' teacher) [{timing['dynamics_labels_seconds']:.0f}s]")
        if ck is not None:
            _ck_save(os.path.join(ck, "dynamics.pt"), {"dM": dM, "dV": dV, "g_ref": g_ref})

    # Score targets: every chart's claim error, s_k ~ -log(e_k + eps) - delta_M - log(1 + delta_V).  A(i->j) = s_j - s_i.
    Y = np.clip(-np.log(E + eps) - dM - np.log1p(dV), -A_CLIP, A_CLIP)
    E_eff = (E + eps) * np.exp(dM) * (1.0 + dV)          # the error the score ranks on
    X = np.stack([featurize_state(i, u, sc, dmax, K) for i, u in st])
    # Cross-chart consistency (build time only): the same configuration seen from chart j must score every
    # chart the same as seen from chart i, weighted by how well chart j reproduces it (eps / (e_j + eps)).
    c_src, c_feat, c_w = [], [], []
    for n, (i, _u) in enumerate(st):
        for j in range(K):
            if j != i:
                c_src.append(n); c_feat.append(featurize_state(j, U_map[n][j], sc, dmax, K)); c_w.append(eps / (E[n, j] + eps))
    c_src, c_w = np.array(c_src), np.array(c_w)
    held = set(rng.choice(S, int(0.2 * S), replace=False).tolist())
    te = np.array([n in held for n in range(S)])
    c_tr = ~te[c_src]
    Xt, Yt = torch.as_tensor(X, device=dev), torch.as_tensor(Y, device=dev)
    Xc = torch.as_tensor(np.stack(c_feat), device=dev)
    Wc = torch.as_tensor(c_w, device=dev)
    src_t = torch.as_tensor(c_src, device=dev)
    tr, ctr = torch.as_tensor(~te, device=dev), torch.as_tensor(c_tr, device=dev)
    # Off-territory states (wide + exit) carry the same total fit weight as territory + overlap: they are rare but
    # they are where the router must decide to leave.
    off = kind >= 2
    w = np.zeros(S)
    for m_ in (~te & ~off, ~te & off):
        if m_.any():
            w[m_] = 1.0 / m_.sum()
    w /= w.sum()
    Wt = torch.as_tensor(w[~te], device=dev)
    net = ScoreNet(dmax, K).double().to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    t0 = time.time()
    start = 0
    train_path = None if ck is None else os.path.join(ck, "train.pt")
    saved = None if train_path is None else _ck_load(train_path)
    if saved is not None:
        net.load_state_dict(saved["net"]); opt.load_state_dict(saved["opt"]); sched.load_state_dict(saved["sched"])
        start = int(saved["epoch"])
        log(f"router: resuming training at epoch {start}/{epochs}")
    for ep in range(start, epochs):
        opt.zero_grad()
        s_all = net(Xt)
        fit = (Wt * ((s_all[tr] - Yt[tr]) ** 2).mean(1)).sum()
        cons = (Wc[ctr] * ((s_all[src_t[ctr]] - net(Xc[ctr])) ** 2).mean(1)).mean()
        (fit + cons).backward()
        opt.step(); sched.step()
        done = ep + 1
        if train_path is not None and (done % int(ckpt_every) == 0 or done == epochs):
            _ck_save(train_path, {"net": net.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                                  "epoch": done})
            log(f"router: training epoch {done}/{epochs} [{time.time() - t0:.0f}s]")
    timing["train_seconds"] = time.time() - t0
    with torch.no_grad():
        P = net(Xt).cpu().numpy()
        Pc = net(Xc).cpu().numpy()
    log(f"router: trained on {int((~te).sum())} states x {K} scores [{timing['train_seconds']:.0f}s]")

    # Pairwise views of the same scores for the decision metrics: rows (n, j != i).
    cols = [[j for j in range(K) if j != i] for i, _u in st]
    yy = np.stack([Y[n, cols[n]] - Y[n, st[n][0]] for n in range(S)])       # exact A(i -> j)
    pp = np.stack([P[n, cols[n]] - P[n, st[n][0]] for n in range(S)])       # predicted A(i -> j)
    report = {"kind": "score", "eps": eps, "n_charts": K,
              "n_states": {"territory": n_a, "overlap": n_b, "wide": n_c, "exit": n_d, "labelled": int(S)},
              "n_per": n_per, "epochs": epochs, "held_out": {}}
    for lab, kk in (("territory", 0), ("overlap", 1), ("wide", 2), ("exit", 3)):
        m = te & (kind == kk)
        if m.any():
            a, b = pp[m].ravel(), yy[m].ravel()
            report["held_out"][lab] = {"rows": int(a.size),
                                       "rmse_A": float(np.sqrt(np.mean((a - b) ** 2))),
                                       "corr": float(np.corrcoef(a, b)[0, 1]),
                                       "median_abs_A": float(np.median(np.abs(b)))}
    # Is each score real where its chart cannot represent the pose (OOD), not just where it can?
    far = E > 10.0 * eps
    report["scores"] = {f"chart_{k}": {
        "rmse_in_territory": float(np.sqrt(np.mean((P[te & ~far[:, k], k] - Y[te & ~far[:, k], k]) ** 2))) if (te & ~far[:, k]).any() else None,
        "rmse_out_of_territory": float(np.sqrt(np.mean((P[te & far[:, k], k] - Y[te & far[:, k], k]) ** 2))) if (te & far[:, k]).any() else None}
        for k in range(K)}
    hp = np.flatnonzero(te)
    k_pred, k_true = P[hp].argmax(1), E_eff[hp].argmin(1)
    report["false_best_rate"] = float(np.mean(E_eff[hp, k_pred] > 3.0 * E_eff[hp, k_true])) if len(hp) else None
    q3 = lambda v: {"p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))} if len(v) else None
    report["dynamics"] = None if not use_dyn else {"g_ref": g_ref, **{f"chart_{k}": {
        "delta_M_in_territory": q3(dM[~far[:, k], k]), "delta_M_out_of_territory": q3(dM[far[:, k], k]),
        "delta_V_in_territory": q3(dV[~far[:, k], k]), "delta_V_out_of_territory": q3(dV[far[:, k], k])}
        for k in range(K)}}
    # Two-way consistency after a real handoff: A(i->j) at u_i plus A(j->i) at u_j = P_j(D_i(u_i)).
    pos = {(n, j): r for r, (n, j) in enumerate((n, j) for n, (i, _u) in enumerate(st) for j in range(K) if j != i)}
    two, two_rep = [], []
    for n in hp:
        i = st[n][0]
        j = cols[n][int(np.argmax(pp[n]))]
        sc_j = Pc[pos[(n, j)]]
        v = abs((P[n, j] - P[n, i]) + (sc_j[i] - sc_j[j]))
        two.append(v)
        if E[n, j] <= 10.0 * eps:
            two_rep.append(v)
    summ = lambda v: {"n": len(v), "p50": float(np.median(v)), "p95": float(np.percentile(v, 95)), "max": float(np.max(v))} if len(v) else None
    report["two_way_abs_sum"] = {"all": summ(two), "receiver_representable": summ(two_rep)}
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
    log("router: " + json.dumps({"false_best_rate": report["false_best_rate"], "two_way_abs_sum": report["two_way_abs_sum"]}))
    timing.update(started_at=started_at, finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
                  wall_seconds=time.time() - t_start)
    report["timing"] = timing
    payload = {"kind": "score", "state": {k: v.cpu() for k, v in net.state_dict().items()},
               "dmax": dmax, "K": K, "sc": sc, "eps": eps}
    return payload, report
