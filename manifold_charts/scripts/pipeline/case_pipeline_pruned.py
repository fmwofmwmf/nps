"""Pruned end-to-end for one case: [densify x4] -> native fit (cover-mask retirement) -> router -> stage ROM.
usage: case_pipeline_pruned.py CASE [TAG]     (run from anywhere; writes work/<CASE>_native_<TAG>, work/polished_pipeline_20261007/roms/<CASE>_<TAG>)
The router (advantage net) is trained from the finished chart models and stored in the package as
physics_fields["router"] BEFORE staging.  The support / closure label nets are not part of this pipeline."""
import os, sys, subprocess, time
from pathlib import Path
case = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "pruned"
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]          # sibling scripts / manifold_charts
RUN = ROOT / "work" / "polished_pipeline_20261007"; RUN.mkdir(parents=True, exist_ok=True)   # logs + staged roms (local)
PY = str(ROOT.parent / ".venv" / "Scripts" / "python.exe")
env = {**os.environ, "CASE": case, "ROMS_DIR": str(RUN / "roms"), "PYTHONUNBUFFERED": "1"}
rom_name = f"{case}_{tag}"; t_all = time.time()
def run(name, args, extra=None):
    t0 = time.time(); print(f"[{case}] {name} ...", flush=True)
    with open(RUN / f"{case}_{tag}_{name}.log", "w") as f:
        r = subprocess.run([PY] + args, env={**env, **(extra or {})}, stdout=f, stderr=subprocess.STDOUT, cwd=str(ROOT))
    print(f"[{case}] {name} exit {r.returncode} [{time.time()-t0:.0f}s]", flush=True)
    if r.returncode: sys.exit(r.returncode)
if not (ROOT / "work" / f"{case}_dense_x4" / "new_dataset.npz").exists():
    run("densify", [str(HERE / "build_dense_case.py"), "4"], {"BASE": case, "PER": "60", "OUTN": f"{case}_dense_x4"})
run("fit", [str(HERE / "run_native_case.py"), "3000", "300", "0.5", tag], {"DATASET_DIR": f"{case}_dense_x4", "POLISH": "12000"})
run("router", ["-m", "atlas_rom.rom.router_build", "--work", str(ROOT / "work" / f"{case}_native_{tag}"), "--case", case,
               "--dataset", str(ROOT / "work" / f"{case}_dense_x4")])
code = f"""
import sys; sys.path.insert(0, r'{ROOT}')
from atlas_rom.rom import package
m = package.stage(r'{ROOT / 'work' / (case + '_native_' + tag)}', '{case}', out_root=r'{RUN / 'roms'}', package_name='{rom_name}', require_pass=False)
print('rom_mode_ready', m['rom_mode_ready'], m['gates'])
"""
run("stage", ["-c", code])
print(f"[{case}] ALL DONE total {time.time()-t_all:.0f}s", flush=True)
