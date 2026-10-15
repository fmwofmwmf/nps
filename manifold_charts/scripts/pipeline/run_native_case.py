"""Rotate refit with repo-native 5-joint closure.  usage: run_phase2.py P2 LR_STEP GAMMA TAG [FT_EPOCHS]
Stage 1: competitive_cae fit (8 charts, width 96, CUDA, closure x10, lr decay via lr_step_epochs/gamma).
Stage 2: per-chart fine-tune on each chart's competent poses (recon + closure x10), then re-score.
Scores (200 fresh oracle poses): D(E(q)) best-chart rms and worst-joint gap, plus cross-chart disagreement
(rms between D_i(E_i q) and D_j(E_j q) over poses two charts both reconstruct acceptably)."""
import os, json, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]                          # manifold_charts
sys.path.insert(0, str(ROOT))
from atlas_rom.cases import load_spec
from atlas_rom.charts.chart_contract import ChartFitConfig, ChartFitInput
from atlas_rom.charts.competitive_cae import CompetitiveCAEFitter
from atlas_rom.charts.validity import validity_from_case
from atlas_rom.simulate import oracle_eval

P2, STEP, GAMMA, TAG = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
FT = int(sys.argv[5]) if len(sys.argv) > 5 else 6000
MULT = 10.0
case = os.environ.get("CASE", "rotate"); dev = torch.device("cuda")
OUT = ROOT / "work" / f"{case}_native_{TAG}"; OUT.mkdir(parents=True, exist_ok=True)
t0 = time.time()
import os
DATA = os.environ.get("DATASET_DIR", case)
data = np.load(ROOT / "work" / DATA / "new_dataset.npz", allow_pickle=False)
q_orig = np.asarray(np.load(ROOT / "work" / case / "new_dataset.npz")["q"], float)
q = np.asarray(data["q"], float); n = len(q)
spec = load_spec(case); nb = len(spec.mechanism.driven_bodies)
inner = validity_from_case(spec)                                   # repo-native: includes grounded joints
blk = q.reshape(n, nb, 3, 4)
tv = blk[..., 3].var(0).mean(-1); rv = blk[..., :3].reshape(n, nb, -1).var(0).mean(-1)
allv = np.concatenate([rv, tv]); floor = float(np.median(allv[allv > 1e-12])); tv = np.where(tv > 1e-12, tv, floor)
closure_scale = float(tv.mean())
rs = np.empty((nb, 3, 4)); rs[..., :3] = np.where(rv > 1e-12, rv, floor)[:, None, None]; rs[..., 3] = tv[:, None]; rs = rs.reshape(-1)


evidence = ChartFitInput(q, np.asarray(data["frames"], float),
                         np.asarray(data.get("local_edges", data["cover_edges"]), dtype=int), physical_validity=inner)
config = ChartFitConfig(fitter_id="competitive_cae", width=96, epochs=1, construction_only=False, seed=0,
                        options={"n_charts": 8, "device": "cuda"})
report, models, patches, anchors = CompetitiveCAEFitter().fit(
    evidence, config, work=str(OUT), case=case, device="cuda", n_charts=8, pretrain_steps=500,
    phase1_max_epochs=3000, phase2_epochs=P2, validity=inner, closure_standardize_mult=10.0, lbfgs_polish_iters=int(os.environ.get('POLISH', '12000')),   # test hack: validity is already x(10/cs), so mult=cs gives closure*10/cs in the polish too
    exploration_snapshot_every=(int(os.environ["SNAP_EVERY"]) if os.environ.get("SNAP_EVERY") else None),
    phase2_exploration_snapshot_every=(int(os.environ["SNAP2_EVERY"]) if os.environ.get("SNAP2_EVERY") else None),
    epoch_trace=bool(os.environ.get("EPOCH_TRACE")), snapshot_prune_steps=bool(os.environ.get("PRUNE_SNAP")),
    m_step_batch_size=int(os.environ.get("BATCH", "256")),
    lr_step_epochs=STEP, lr_step_gamma=GAMMA, log=lambda *a: None)
t_fit = time.time() - t0
models = [m.cpu().double().eval() for m in models]

oracle = oracle_eval._build_oracle(spec)
qt = oracle_eval._oracle_test_poses(q_orig, oracle, 200, seed=0, sigma=0.05)
from scipy.spatial import cKDTree
_d = cKDTree(q).query(qt, k=1)[0]; print(f'fresh test poses: distance to nearest TRAINING pose median {np.median(_d):.3f} min {_d.min():.3f} (train NN spacing {np.median(cKDTree(q).query(q, k=2)[0][:,1]):.3f})', flush=True)
rs_t = torch.as_tensor(rs, dtype=torch.float64)


def score(ms, label):
    t = torch.as_tensor(qt).double(); P = []
    for m in ms:
        with torch.no_grad(): P.append(m.decode(m.encode(t)).numpy())
    P = np.stack(P)                                              # (K, N, 12nb)
    rms_all = np.sqrt(((P - qt[None]) ** 2).mean(-1) * qt.shape[1])   # (K, N)
    best = rms_all.argmin(0); R = P[best, np.arange(len(qt))]
    gap = oracle_eval._worst_joint_gap(spec, R, nb)
    # cross-chart disagreement over poses where >=2 charts reconstruct within 2x the best-chart rms (and < 1.5)
    dis = []
    for i in range(len(qt)):
        ok = np.flatnonzero(rms_all[:, i] <= max(2 * rms_all[:, i].min(), 0.2))
        for a in range(len(ok)):
            for b in range(a + 1, len(ok)):
                dis.append(np.sqrt(((P[ok[a], i] - P[ok[b], i]) ** 2).mean() * qt.shape[1]))
    dis = np.array(dis) if dis else np.array([np.nan])
    r = dict(label=label, rms_mean=float(rms_all.min(0).mean()), rms_p95=float(np.percentile(rms_all.min(0), 95)),
             gap_mean=float(gap.mean()), gap_p95=float(np.percentile(gap, 95)), gap_max=float(gap.max()),
             disagree_mean=float(np.nanmean(dis)), disagree_p95=float(np.nanpercentile(dis, 95)), n_pairs=int(len(dis)))
    print(json.dumps(r), flush=True); return r


def save(ms, name):
    torch.save({"chart_atlas": {"states": [{k: v.detach().cpu() for k, v in m.state_dict().items()} for m in ms],
                                "encoder_kind": "shallow"}}, OUT / name)


tp = report["two_phase"]
summary = dict(tag=TAG, p2=P2, lr_step=STEP, gamma=GAMMA, n_patches=report["n_patches"], n_uncovered=report["n_uncovered"],
               p2_peak_uncovered=max(x["n_uncovered"] for x in tp["phase2"]["coverage_curve"]),
               handoff=report.get("handoff_over_sqrt2_recon"), overlap_ok=report.get("overlap_ok"),
               noise=[report.get("inverse_noise_floor"), report.get("off_domain_noise_floor")], fit_seconds=t_fit)
summary["fit"] = score(models, "after fitter")
save(models, "atlas_native_polish.pt")

summary["polish_report"] = {k: v for k, v in report["lbfgs_polish"].items()} if report.get("lbfgs_polish") else None
summary["finetuned"] = None
summary["total_seconds"] = time.time() - t0
(OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
print("done", round(time.time() - t0), flush=True)
