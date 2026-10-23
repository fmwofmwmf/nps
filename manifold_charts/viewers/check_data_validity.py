"""check_data_validity.py -- validate a trained chart atlas against FRESH oracle poses.

Generates ``--test-poses`` new configurations by perturbing the training samples
and projecting them back onto the mechanism's constraint manifold with the
ideal-joint oracle (none of these poses were seen in training), then opens a
Polyscope viewer:

* ground truth pose (green) and its best-chart round-trip ``D(E(q))`` (orange);
* a ``pose`` slider to step through the test set;
* an ``offset`` slider that slides the ground-truth bodies away along +x, so the
  GT and the reconstruction can be inspected side by side (0 = overlaid, to see
  exactly where the decoder tears the mechanism).

Per pose it also prints the winning chart, the reconstruction RMS and the worst
revolute-joint closure gap of the reconstruction.

    python -m viewers.check_data_validity --work work/five_bar --test-poses 200
    python -m viewers.check_data_validity --work work/stephenson_link --test-poses 150 --offset 2.0
    python -m viewers.check_data_validity --work work/rotate \
        --artifact patch_atlas_dev_12chart_3k_p1_cuda.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # manifold_charts/
_HERE = os.path.join(_ROOT, "atlas_rom")
for _path in (_ROOT, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from atlas_rom.simulate.oracle_eval import _build_oracle, _oracle_test_poses, _worst_joint_gap  # noqa: E402,F401
from atlas_rom.cases import load_spec, resolve_case_name  # noqa: E402
from viewers.generic_manifold_diagnostics import _atlas_view  # noqa: E402
from viewers.mesh_view import MeshAdapter  # noqa: E402
from viewers.view_sample_set import (  # noqa: E402
    _infer_case, _load_chart_model, _load_samples,
)




def _best_roundtrip(chart_atlas, q):
    """``(rt, best_chart, recon_rms)`` -- best chart per pose is the min-recon one."""
    import torch

    states = chart_atlas.get("states") or []
    if not states:
        raise ValueError("atlas artifact has no chart 'states'")
    encoder_kind = chart_atlas.get("encoder_kind", "shallow")
    tensor = torch.as_tensor(np.asarray(q, float)).double()
    preds = np.empty((len(states), len(q), q.shape[1]), float)
    for i, state in enumerate(states):
        model = _load_chart_model(state, encoder_kind)
        with torch.no_grad():
            preds[i] = model.decode(model.encode(tensor)).numpy()
    mse = np.square(preds - np.asarray(q, float)[None]).mean(-1)   # (K, N)
    best = mse.argmin(0)
    rt = preds[best, np.arange(len(q))]
    rms = np.sqrt(mse.min(0) * q.shape[1])
    return rt, best, rms



def _dist_to_training(q_test, q_train, n_bodies):
    from scipy.spatial import cKDTree
    def feat(Q):
        T = Q.reshape(len(Q), n_bodies, 3, 4)
        return np.concatenate((T[..., :3, :2].reshape(len(Q), -1), T[..., 3].reshape(len(Q), -1)), 1)
    tree = cKDTree(feat(q_train))
    spacing = float(np.mean(tree.query(feat(q_train), k=2)[0][:, 1]))
    return tree.query(feat(q_test), k=1)[0], spacing


def _load_chart_atlas(work: str, artifact: str | None):
    """Load an explicitly selected atlas artifact, or preserve legacy discovery."""
    if artifact is None:
        report, chart_atlas, source = _atlas_view(work)
        return report, chart_atlas, source

    path = artifact if os.path.isfile(artifact) else os.path.join(work, artifact)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"artifact not found: {artifact}")
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    chart_atlas = state.get("chart_atlas")
    if not isinstance(chart_atlas, dict) or not chart_atlas.get("states"):
        raise ValueError(f"{path}: artifact has no trained chart states")
    return None, chart_atlas, os.path.abspath(path)


def check_data_validity(case, work, *, n_test, offset, seed, sigma, artifact=None, use_dataset=False,
                        rank_by_error=False):
    spec = load_spec(case)
    n_bodies = len(spec.mechanism.driven_bodies)

    q_train, _ = _load_samples(work)
    if q_train.shape[1] != 12 * n_bodies:
        raise ValueError(f"{work}: q width {q_train.shape[1]} != 12*{n_bodies}")

    report, chart_atlas, source = _load_chart_atlas(work, artifact)
    if not chart_atlas or not chart_atlas.get("states"):
        raise FileNotFoundError(f"no trained chart atlas (competitive_*.pt) in {work}")

    if use_dataset:
        # Walk the actual training set instead of fresh oracle-perturbed
        # poses -- every pose here WAS seen in training, so this is a fit
        # check (does the shipped atlas reconstruct its own data), not a
        # generalization probe.  d_train is identically 0 by construction.
        rng = np.random.default_rng(seed)
        n_use = min(int(n_test), len(q_train)) if n_test else len(q_train)
        idx = (np.arange(len(q_train)) if n_use >= len(q_train)
              else rng.choice(len(q_train), size=n_use, replace=False))
        q_test = q_train[np.sort(idx)]
    else:
        oracle = _build_oracle(spec)
        q_test = _oracle_test_poses(q_train, oracle, n_test, seed=seed, sigma=sigma)
    rt, best, rms = _best_roundtrip(chart_atlas, q_test)
    gt_gap = _worst_joint_gap(spec, q_test, n_bodies)
    rt_gap = _worst_joint_gap(spec, rt, n_bodies)
    d_train, spacing = _dist_to_training(q_test, q_train, n_bodies)
    extrapolating = d_train > 3.0 * spacing

    Tg = q_test.reshape(-1, n_bodies, 3, 4)
    Tr = rt.reshape(-1, n_bodies, 3, 4)

    mode = "training-set poses (--use-dataset)" if use_dataset else "fresh oracle poses"
    print(f"[check_data_validity] {case}: {len(q_test)} {mode}, "
          f"atlas {source} (K={len(chart_atlas['states'])})")
    print(f"  recon rms   mean {rms.mean():.4f}  p95 {np.percentile(rms, 95):.4f}  max {rms.max():.4f}")
    print(f"  GT joint gap mean {gt_gap.mean():.2e}  |  RT joint gap mean {rt_gap.mean():.4f} "
          f"p95 {np.percentile(rt_gap, 95):.4f} max {rt_gap.max():.4f}")
    in_dist = ~extrapolating
    if in_dist.any() and extrapolating.any():
        print(f"  in-distribution ({int(in_dist.mean() * 100)}%): recon rms {rms[in_dist].mean():.4f}   "
              f"| extrapolating (>3x NN spacing): recon rms {rms[extrapolating].mean():.4f}   "
              f"corr(rms, dist-to-train) {np.corrcoef(rms, d_train)[0, 1]:.2f}")

    import polyscope as ps
    from polyscope import imgui

    gt_mesh = MeshAdapter(spec.mechanism)
    rt_mesh = MeshAdapter(spec.mechanism)
    ps.init()
    ps.set_up_dir(gt_mesh.up_dir)
    ps.set_ground_plane_mode("shadow_only")
    rt_mesh.register(ps, Tr[0], name_prefix="RT/", color_override=(0.92, 0.42, 0.18))
    gt_mesh.register(ps, Tg[0], name_prefix="GT/", include_static=False,
                     color_override=(0.30, 0.72, 0.38))

    # slider position -> pose index; "rank by error" walks the worst reconstructions first
    worst_first = np.argsort(-rms, kind="stable")
    state = {"slot": 0, "offset": float(offset), "ranked": bool(rank_by_error)}
    order = lambda: worst_first if state["ranked"] else np.arange(len(q_test))

    def _apply():
        i = state["i"] = int(order()[state["slot"]])
        shifted = Tg[i].copy()
        shifted[:, :3, 3] = shifted[:, :3, 3] + np.array([state["offset"], 0.0, 0.0])
        gt_mesh.update(shifted)
        rt_mesh.update(Tr[i])

    def _callback():
        changed, value = imgui.SliderInt("pose", state["slot"], 0, len(q_test) - 1)
        if changed:
            state["slot"] = int(value); _apply()
        changed, value = imgui.Checkbox("rank by error (worst first)", state["ranked"])
        if changed:
            state["ranked"] = bool(value); state["slot"] = 0; _apply()
        changed, value = imgui.SliderFloat("offset", state["offset"], 0.0, 8.0)
        if changed:
            state["offset"] = float(value); _apply()
        imgui.Separator()
        i = state["i"]
        imgui.Text(f"pose {i + 1}/{len(q_test)}"
                   + (f"   (error rank {state['slot'] + 1})" if state["ranked"] else ""))
        imgui.Text(f"best chart  {int(best[i])}")
        imgui.Text(f"recon rms   {rms[i]:.4f}")
        imgui.Text(f"RT joint gap {rt_gap[i]:.4f}   (GT {gt_gap[i]:.1e})")
        imgui.Text(f"dist to train {d_train[i]:.3f}  ({d_train[i] / spacing:.1f}x NN spacing)"
                   + ("   [EXTRAPOLATING]" if extrapolating[i] else ""))

    ps.set_user_callback(_callback)
    _apply()
    ps.show()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", required=True, help="dataset dir holding a competitive_*.pt atlas")
    parser.add_argument("--case", default=None, help="case name (inferred from --work if omitted)")
    parser.add_argument("--test-poses", type=int, default=200, dest="n_test",
                        help="number of fresh oracle poses to generate (default 200)")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="initial GT +x displacement (slider goes 0..8)")
    parser.add_argument("--sigma", type=float, default=0.05,
                        help="per-coordinate perturbation stddev before oracle projection "
                             "(default 0.05 stays near the training distribution; raise it "
                             "to ~0.15 to probe extrapolation into unsampled corners)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--artifact", default=None,
                        help="checkpoint filename inside --work (or an absolute path); "
                             "default: auto-discover the canonical atlas")
    parser.add_argument("--use-dataset", action="store_true",
                        help="slide over the actual training-set poses instead of fresh "
                             "oracle-perturbed ones (a fit check, not a generalization probe; "
                             "--test-poses caps how many, --seed picks the subsample)")
    parser.add_argument("--rank-by-error", action="store_true",
                        help="start with the pose slider ordered worst reconstruction first "
                             "(also a checkbox in the viewer)")
    args = parser.parse_args()

    work = args.work if os.path.isdir(args.work) else os.path.join(_HERE, "work", args.work)
    if not os.path.isdir(work):
        parser.error(f"work dir not found: {args.work}")
    case = resolve_case_name(args.case) if args.case else _infer_case(work, os.path.basename(work))
    check_data_validity(case, work, n_test=args.n_test, offset=args.offset,
                        seed=args.seed, sigma=args.sigma, artifact=args.artifact,
                        use_dataset=args.use_dataset, rank_by_error=args.rank_by_error)


if __name__ == "__main__":
    main()
