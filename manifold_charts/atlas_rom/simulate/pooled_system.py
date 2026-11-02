"""pooled_system.py — an ideal-joint `PhysicalSystem` that answers batches in parallel.

Every command an explorer sends from one baseline is independent of the
others: the oracle restores that baseline before each one.  The advancing
front sends a node's whole probe set at once (`command_many`), and a
`LeafSystem` sends a batch of commands from DIFFERENT baselines with their own
scales (`command_batch`).  This system farms such batches out to worker
processes, each holding its own `IdealJointOracle` built from the same spec,
and answers single commands in the main process.  The responses are the same
numbers the serial oracle produces: the same code runs in each worker.

Why processes and not threads or a GPU: one projection is a few small dense
least-squares solves (315 x 252 on `drivetrain_steer`), dominated by LAPACK
calls that are fastest single-threaded at this size; BLAS threads cost more
than they save, and a GPU only wins on large batches, in fp64, with a
rank-deficient minimum-norm solve torch's CUDA least squares does not offer.

Analytic (ideal-joint) oracles only.  libuipc is one world per process and is
not pooled here.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from typing import Sequence

import numpy as np

from atlas_rom.simulate.oracle_base import PhysicalObservation, PhysicalResponse

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
_worker_oracle = None


def _init_worker(spec):
    global _worker_oracle
    from atlas_rom.simulate import ideal_oracle
    _worker_oracle = ideal_oracle.build_ideal_oracle(spec, log=lambda *_: None)


def _run_job(item):
    """A whole function of a serial system, run in this worker."""
    from atlas_rom.simulate.oracle_base import LegacyPhysicalOracleAdapter
    fn, args = item
    return fn(LegacyPhysicalOracleAdapter(_worker_oracle), *args)


def _respond(item):
    baseline, direction, scale = item
    _settled, token = _worker_oracle.baseline(baseline)
    r = _worker_oracle.directional_response(token, direction, scale)
    return r.result, r.converged, r.valid, r.diagnostics


class PooledIdealSystem:
    """`PhysicalSystem` over an ideal-joint oracle with a process pool.

    Use as a context manager (or call `close()`), so the workers exit."""

    def __init__(self, spec, workers: int | None = None, *, chunk: int = 1):
        from atlas_rom.simulate import ideal_oracle
        # Workers inherit the environment at spawn: one LAPACK thread each.
        for var in _THREAD_VARS:
            os.environ.setdefault(var, "1")
        self.oracle = ideal_oracle.build_ideal_oracle(spec, log=lambda *_: None)
        self.workers = workers or max(1, (os.cpu_count() or 2) - 1)
        self.chunk = chunk
        self._pool = mp.get_context("spawn").Pool(self.workers, initializer=_init_worker,
                                                  initargs=(spec,))
        self.batched_commands = 0

    # -- PhysicalSystem ------------------------------------------------------
    def seed(self):
        return PhysicalObservation(np.asarray(self.oracle.seed(), float))

    def restore(self, observation):
        settled, _token = self.oracle.baseline(np.asarray(observation.pose, float).reshape(-1))
        return settled

    def command(self, session, direction, scale, *, parent=None):
        _settled, token = self.oracle.baseline(session)
        r = self.oracle.directional_response(token, np.asarray(direction, float), scale, parent=parent)
        return PhysicalResponse(PhysicalObservation(np.asarray(r.result, float)), r.converged, r.valid,
                                r.diagnostics, parent)

    def command_many(self, session, directions: Sequence[np.ndarray], scale, *, parent=None):
        return self.command_batch([(session, d, scale) for d in directions], parent=parent)

    # -- batches from many baselines -----------------------------------------
    def command_batch(self, items, *, parent=None):
        """`items`: (session, direction, scale) triples, any baselines."""
        items = [(np.asarray(s, float), np.asarray(d, float), float(t)) for s, d, t in items]
        if len(items) <= 1:
            return [self.command(*item, parent=parent) for item in items]
        self.batched_commands += len(items)
        out = self._pool.map(_respond, items, chunksize=self.chunk)
        return [PhysicalResponse(PhysicalObservation(np.asarray(res, float)), conv, val, diag, parent)
                for res, conv, val, diag in out]

    # -- whole independent jobs ----------------------------------------------
    def map_jobs(self, fn, args_list):
        """`fn(system, *args)` for each args, each in a worker on a serial
        system over that worker's oracle.  `fn` must be module-level and its
        arguments and result picklable."""
        return self._pool.map(_run_job, [(fn, tuple(a)) for a in args_list], chunksize=1)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
