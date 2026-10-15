"""Response-only, backend-independent physical advancing-front exploration."""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import json
from pathlib import Path
import time
from typing import Callable

import numpy as np

from atlas_rom.simulate.oracle_base import PhysicalObservation, PhysicalSystem


@dataclass(frozen=True)
class AdvancingFrontOptions:
    step: float = .08
    max_samples: int | None = None
    max_seconds: float | None = 1800.
    spacing_fraction: float = .6
    seed: int = 0
    cone_angle_deg: float = 22.5
    probe_scale: float | None = None
    rank_gap_ratio: float = 1e3
    checkpoint_seconds: float | None = None
    report_every: int | None = None  # log progress/throughput every N accepted poses
    max_probes_per_node: int | None = None  # give up on a node's cone after this many tries


def _unit(value: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(value))
    return None if norm <= 1e-12 else np.asarray(value, float) / norm


def _rank_by_gap(singular: np.ndarray, gap_ratio: float) -> int | None:
    """First decisive spectral gap, from the top of the spectrum."""
    values = np.asarray(singular, float)
    if not len(values) or values[0] <= 0.:
        return None
    floor = values[0] * np.finfo(float).eps
    ratios = np.maximum(values, floor)[:-1] / np.maximum(values, floor)[1:]
    hits = np.flatnonzero(ratios >= gap_ratio)
    return int(hits[0]) + 1 if len(hits) else None


def _maximin_direction(probed: list[np.ndarray], d: int,
                       cone_angle_deg: float | None = None) -> tuple[np.ndarray, float]:
    """The direction of the widest unprobed cone, and its angular radius.

    Maximising the smallest angle to ``probed`` is the same as minimising the
    largest dot product, so the objective is ``max_i (u_i . p)`` on the unit
    sphere.  Two halves, because neither alone is enough:

    Cheap candidates first.  On a circle the widest gap's midpoint is exactly
    the normalised sum of the two probes bounding it, so the +/- pairwise sums
    contain the d == 2 answer outright -- no search, no dimension branch.  Both
    signs are required: ``u_i + u_j`` bisects the SMALLER angle between them, so
    for a gap wider than pi the unnegated sum points into the occupied side.
    The candidate scan is fully vectorised (one ``(cands @ used.T).max(1)``),
    and ``d == 1`` is closed form -- S^0 is two points.

    Then an annealed descent, which is what d >= 3 needs: there the deepest hole
    is generically equidistant from up to d probes and lies on no pairwise
    bisector.  ``max`` is non-smooth, so it is softened by log-sum-exp, whose
    gradient is a softmax-weighted mean of the probes and which sharpens to the
    true max as the schedule runs.  At zero sharpness the weights are uniform
    and the minimiser is exactly ``-sum(u_i)`` normalised, so the walk starts
    from a closed-form optimum and tracks it as the objective sharpens; the
    broad early landscape is what removes the need to restart in each basin,
    and a smooth objective needs no line search (``1/sharpness`` is the
    Lipschitz step).

    ``cone_angle_deg`` is an optional early-out: the candidate scan is a lower
    bound on the true maximin separation (its points are a subset of the
    sphere), so once that bound already exceeds the cone the enqueue decision
    is settled and the descent is skipped -- at worst a marginally sub-optimal
    direction.  Per the contract below that only ever costs one probe that
    comes back ``filled``, never a missed hole.

    Every point is scored by the EXACT max dot and the best is kept, so the
    result is never worse than the candidates alone and the softening can only
    influence *where* the search looks.  That direction matters: log-sum-exp
    overestimates the max, which would understate the separation, and
    understating it retires a node while a real hole remains -- the failure
    that stalls exploration.  Overstating it merely costs one probe that comes
    back ``filled``.
    """
    used = np.asarray([unit for value in probed if (unit := _unit(value)) is not None], float)
    neutral = np.ones(d, float) / np.sqrt(d)
    if not len(used):
        return neutral, float(np.pi)

    if d == 1:
        # S^0 = {+1, -1}: the widest gap is whichever half-line is emptier.
        column = used[:, 0]
        plus, minus = float(column.max()), float((-column).max())
        if plus <= minus:
            return np.array([1.0]), float(np.arccos(np.clip(plus, -1.0, 1.0)))
        return np.array([-1.0]), float(np.arccos(np.clip(minus, -1.0, 1.0)))

    blocks = [-used]
    centroid = _unit(-used.sum(axis=0))
    if centroid is not None:
        blocks.append(centroid[None, :])
    left, right = np.triu_indices(len(used), k=1)
    if len(left):
        pair = np.vstack((used[left] + used[right], used[left] - used[right]))
        norms = np.linalg.norm(pair, axis=1)
        keep = norms > 1e-12
        pair = pair[keep] / norms[keep, None]
        blocks.append(np.vstack((pair, -pair)))
    candidates = np.vstack(blocks)
    scores = (candidates @ used.T).max(axis=1)
    winner = int(scores.argmin())
    best, best_score = candidates[winner], float(scores[winner])
    best_separation = float(np.arccos(np.clip(best_score, -1.0, 1.0)))

    if cone_angle_deg is not None and best_separation > np.deg2rad(cone_angle_deg):
        return best, best_separation

    point = centroid if centroid is not None else neutral
    for sharpness in (1., 2., 4., 8., 16., 32., 64., 128., 256.):
        for _ in range(40):
            scaled = sharpness * (used @ point)
            weights = np.exp(scaled - scaled.max())
            gradient = (weights / weights.sum()) @ used
            tangent = gradient - point * float(point @ gradient)
            if np.linalg.norm(tangent) <= 1e-12:
                break
            stepped = _unit(point - tangent / max(sharpness, 2.))
            if stepped is None:
                break
            point = stepped
            score = float(np.max(used @ point))
            if score < best_score:
                best, best_score = point, score
    return best, float(np.arccos(np.clip(best_score, -1., 1.)))


def explore(system: PhysicalSystem, *, options: AdvancingFrontOptions | None = None,
            seeds: list[PhysicalObservation] | None = None,
            distance: Callable[[np.ndarray, np.ndarray], float] | None = None,
            log: Callable[[str], None] = lambda _message: None,
            checkpoint_dir: str | Path | None = None,
            resume_from: str | None = None, repair_inference: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Grow a certified local sample graph from paired physical responses.

    Rank, local frames, confidence, and cone coverage are inferred here from
    simulator responses.  Backends never provide tangent frames, dimensions,
    memberships, or command schedules.
    """
    options = options or AdvancingFrontOptions()
    if options.step <= 0 or not 0 < options.spacing_fraction <= 1:
        raise ValueError("step must be positive and spacing_fraction must be in (0, 1]")
    distance = distance or (lambda left, right: float(np.linalg.norm(left - right)))
    checkpoint_dir = None if checkpoint_dir is None else Path(checkpoint_dir)
    observations = list(seeds or [system.seed()])
    poses = [np.asarray(item.pose, float).reshape(-1) for item in observations]
    edges: list[tuple[int, int]] = []
    neighbours: dict[int, set[int]] = {}
    frontier: list[tuple[int, np.ndarray]] = []
    probed: dict[int, list[np.ndarray]] = {}
    inference_probes: dict[int, list[np.ndarray]] = {}
    records: dict[int, dict] = {}
    dirty: set[int] = set()
    # Nodes whose cone last measured fully covered.  Purely a recompute cache:
    # cleared whenever a node is dirtied (new neighbour / re-measured frame), so
    # it never suppresses a real re-check.  Not persisted -- a resume rebuilds
    # its frontier from the saved queue, and retired nodes are simply absent.
    retired: set[int] = set()
    state_counts: Counter[str] = Counter()
    started = time.perf_counter()
    last_checkpoint = started
    last_report_count, last_report_time = len(poses), started

    def checkpoint(*, budget_exhausted: bool = False, complete: bool = False) -> None:
        if checkpoint_dir is None:
            return
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_dir / "frontier_state.npz", poses=np.asarray(poses),
                            edges=np.asarray(edges, int), probed=np.asarray(probed, dtype=object),
                            frontier=np.asarray(frontier, dtype=object),
                            inference_probes=np.asarray(inference_probes, dtype=object),
                            records=np.asarray(records, dtype=object), dirty=np.asarray(sorted(dirty), int))
        (checkpoint_dir / "frontier_state.json").write_text(json.dumps({"budget_exhausted": budget_exhausted,
            "frontier_complete": complete, "frontier_state_counts": dict(state_counts),
            "options": options.__dict__}, indent=2))

    def infer(index: int) -> tuple[object, np.ndarray, dict]:
        """Infer one frame from paired h/h2 response experiments and secants."""
        observation = observations[index]
        session = system.restore(observation)
        ambient = poses[index].size
        if index in records and index not in dirty:
            return session, np.asarray(records[index]["frame"], float), records[index]
        prior = inference_probes.setdefault(index, [])
        for other in neighbours.get(index, ()):
            secant = _unit(poses[other] - poses[index])
            if secant is not None and not any(np.allclose(secant, value) for value in prior):
                prior.append(secant)
        # Walked secants are better-aimed than a random draw -- they connect two
        # accepted samples, so they lie near the tangent by construction -- but
        # they must TOP UP the generic experiments, never stand in for them.  A
        # fresh child always has exactly one neighbour (its parent), so seeding
        # from secants alone leaves it measuring a rank-1 span from a single
        # direction, and a rank-1 node has only two cone directions and retires
        # after two walks.  That read SO(3) as rank 1 at 176 of 177 nodes and
        # stopped exploration outright.
        count = max(1, int(np.ceil(np.log2(ambient + 1))))
        if len(prior) < count:
            # A compact, deterministic set of generic physical experiments.
            # Its size grows with observation complexity but is not a declared
            # state/tangent dimension and does not become the learned frame.
            generator = np.random.default_rng(np.random.SeedSequence((options.seed, index, ambient)))
            proposals, _ = np.linalg.qr(generator.standard_normal((ambient, count)), mode="reduced")
            prior.extend(proposals[:, column] for column in range(count - len(prior)))
        h = options.probe_scale if options.probe_scale is not None else options.step * .25
        secants: list[np.ndarray] = []
        stable = 0
        valid = 0
        for direction in prior:
            unit = _unit(direction)
            if unit is None:
                continue
            plus, minus = system.command_many(session, (unit, -unit), h, parent=index)
            half_plus, half_minus = system.command_many(session, (unit, -unit), h / 2., parent=index)
            responses = (plus, minus, half_plus, half_minus)
            if not all(item.valid and item.converged and np.isfinite(item.observation.pose).all() for item in responses):
                continue
            odd_h = (np.asarray(plus.observation.pose, float).reshape(-1) -
                     np.asarray(minus.observation.pose, float).reshape(-1)) / (2. * h)
            odd_half = (np.asarray(half_plus.observation.pose, float).reshape(-1) -
                        np.asarray(half_minus.observation.pose, float).reshape(-1)) / h
            richardson = (4. * odd_half - odd_h) / 3.
            if not np.isfinite(richardson).all() or np.linalg.norm(richardson) <= 1e-12:
                continue
            valid += 1
            relative = float(np.linalg.norm(odd_half - odd_h) / max(np.linalg.norm(richardson), 1e-12))
            if relative <= .2:
                stable += 1
            secants.append(richardson)
        if not secants:
            frame = np.empty((ambient, 0))
            diagnostics = {"probe_count": len(prior), "valid_pairs": 0, "stable_pairs": 0,
                           "outcome": "uncertain"}
        else:
            left, singular, _right = np.linalg.svd(np.column_stack(secants), full_matrices=False)
            rank = _rank_by_gap(singular, options.rank_gap_ratio)
            gap_ratio = (float(singular[rank - 1] / max(singular[rank], np.finfo(float).tiny))
                         if rank is not None and 0 < rank < len(singular) else None)
            frame = left[:, :rank] if rank is not None else np.empty((ambient, 0))
            diagnostics = {"probe_count": len(prior), "valid_pairs": valid, "stable_pairs": stable,
                           "singular_values": singular.tolist(), "spectral_gap_ratio": gap_ratio,
                           "outcome": "certified_regular" if rank is not None and stable else "uncertain"}
        record = {"index": index, "frame": frame, "rank": int(frame.shape[1]),
                  "outcome": diagnostics["outcome"], "diagnostics": diagnostics}
        records[index] = record
        dirty.discard(index)
        return session, frame, record

    def enqueue(index: int, frame: np.ndarray) -> None:
        if not frame.shape[1]:
            return
        # A node whose cone last came back fully covered stays covered until it
        # gains a neighbour or is re-measured -- both clear it from ``retired``.
        # Skipping the recompute here is what stops a mature node paying an
        # O(k^2) maximin search on every redundant "filled" pop.
        if index in retired:
            return
        if (options.max_probes_per_node is not None and
                len(probed.get(index, ())) >= options.max_probes_per_node):
            # Every probe from this node has failed the same way regardless
            # of direction (a local-curvature pocket, not a coverage gap
            # that more tries would close) -- give up rather than grind.
            retired.add(index)
            state_counts["exhausted"] += 1
            return
        ambient = [np.asarray(probed[index], float)] if probed.get(index) else []
        others = list(neighbours.get(index, ()))
        if others:
            ambient.append(np.asarray([poses[other] - poses[index] for other in others], float))
        coordinates = np.empty((0, frame.shape[1]))
        if ambient:
            projected = np.vstack(ambient) @ frame
            norms = np.linalg.norm(projected, axis=1)
            coordinates = projected[norms > 1e-12] / norms[norms > 1e-12, None]
        direction, separation = _maximin_direction(
            list(coordinates), frame.shape[1], cone_angle_deg=options.cone_angle_deg)
        if separation > np.deg2rad(options.cone_angle_deg):
            unit_ambient = _unit(frame @ direction)
            if unit_ambient is not None:
                frontier.append((index, unit_ambient))
        else:
            retired.add(index)

    if resume_from is not None:
        if checkpoint_dir is None or not (checkpoint_dir / "frontier_state.npz").exists():
            raise ValueError("resume_from requires a saved frontier checkpoint")
        log(f"physical_advancing_front: resuming frontier from {resume_from}")
        saved = np.load(checkpoint_dir / "frontier_state.npz", allow_pickle=True)
        poses = [np.asarray(row, float) for row in saved["poses"]]
        observations = [PhysicalObservation(row, {}) for row in poses]
        edges = [tuple(map(int, row)) for row in saved["edges"]]
        neighbours = {index: set() for index in range(len(poses))}
        for left, right in edges:
            neighbours.setdefault(left, set()).add(right); neighbours.setdefault(right, set()).add(left)

        def _saved_map(key):
            if key not in saved.files:
                return {}
            return saved[key].item() or {}

        probed = {int(key): [np.asarray(item, float) for item in value]
                  for key, value in _saved_map("probed").items()}
        frontier = [(int(index), np.asarray(direction, float))
                    for index, direction in (saved["frontier"].tolist()
                                             if "frontier" in saved.files else [])]
        inference_probes = {int(key): [np.asarray(item, float) for item in value]
                            for key, value in _saved_map("inference_probes").items()}
        records = {int(k): v for k, v in _saved_map("records").items()}
        dirty = set(map(int, saved["dirty"].tolist())) if "dirty" in saved.files else set()
        if repair_inference:
            repair = [index for index, record in records.items()
                      if record.get("outcome") != "certified_regular"]
            for index in repair:
                records.pop(index, None)
                inference_probes.pop(index, None)
            log(f"physical_advancing_front: re-probing {len(repair)} non-regular saved poses")
    # Nodes without a stored frame record (every node on a bare re-open) are
    # (re-)measured and their open cone directions pushed onto the frontier.  On
    # a full-fidelity checkpoint ``records`` is already populated and this is a
    # no-op; on a bare re-open it rebuilds the frontier from the pose cloud.
    for index in range(len(poses)):
        if index in records and index not in dirty:
            continue
        _session, frame, record = infer(index)
        if record["outcome"] == "certified_regular":
            enqueue(index, frame)

    while frontier:
        if (options.checkpoint_seconds is not None and
                time.perf_counter() - last_checkpoint >= options.checkpoint_seconds):
            checkpoint()
            last_checkpoint = time.perf_counter()
        if options.max_samples is not None and len(poses) >= options.max_samples:
            break
        if options.max_seconds is not None and time.perf_counter() - started >= options.max_seconds:
            break
        parent, direction = frontier.pop(0)
        probed.setdefault(parent, []).append(direction)
        session, frame, record = infer(parent)
        if record["outcome"] != "certified_regular":
            state_counts["unknown"] += 1
            continue
        response = system.command(session, direction, options.step, parent=parent)
        if not response.valid or not response.converged:
            state_counts["boundary" if response.diagnostics.get("physical_boundary") else "solver_failure"] += 1
            enqueue(parent, frame)
            continue
        candidate = np.asarray(response.observation.pose, float).reshape(-1)
        if not np.isfinite(candidate).all():
            state_counts["solver_failure"] += 1
            enqueue(parent, frame)
            continue
        if distance(candidate, poses[parent]) < options.spacing_fraction * options.step:
            state_counts["undersized"] += 1
            enqueue(parent, frame)
            continue
        nearby = [index for index, value in enumerate(poses)
                  if distance(candidate, value) < options.spacing_fraction * options.step]
        if nearby:
            child = min(nearby, key=lambda index: distance(candidate, poses[index]))
            if child != parent and (parent, child) not in edges and (child, parent) not in edges:
                edges.append((parent, child))
                neighbours.setdefault(parent, set()).add(child); neighbours.setdefault(child, set()).add(parent)
                dirty.update((parent, child)); retired.difference_update((parent, child))
                if child in records and records[child]["outcome"] != "certified_regular":
                    _child_session, child_frame, child_record = infer(child)
                    if child_record["outcome"] == "certified_regular":
                        enqueue(child, child_frame)
            state_counts["filled"] += 1
            enqueue(parent, frame)
            continue
        child = len(poses)
        poses.append(candidate); observations.append(response.observation); edges.append((parent, child))
        neighbours.setdefault(parent, set()).add(child); neighbours.setdefault(child, set()).add(parent)
        dirty.add(parent); retired.discard(parent)
        state_counts["open"] += 1
        _child_session, child_frame, child_record = infer(child)
        if child_record["outcome"] == "certified_regular":
            enqueue(child, child_frame)
        enqueue(parent, frame)

        if options.report_every and len(poses) - last_report_count >= options.report_every:
            now = time.perf_counter()
            rate = (len(poses) - last_report_count) / max(now - last_report_time, 1e-9)
            log(f"physical_advancing_front: {len(poses)} samples, {len(frontier)} queued cones, "
                f"{rate:.2f} samples/s, {now - started:.0f}s elapsed, states={dict(state_counts)}")
            last_report_count, last_report_time = len(poses), now

    for index in range(len(poses)):
        if index not in records:
            infer(index)
    regular = [index for index, record in records.items()
               if record["outcome"] == "certified_regular" and record["rank"] > 0]
    ranks = {records[index]["rank"] for index in regular}
    modal_rank = max(ranks, key=lambda rank: sum(records[index]["rank"] == rank for index in regular)) if ranks else 0
    regular = [index for index in regular if records[index]["rank"] == modal_rank]
    # Keep a physically observed seed even when every paired experiment ends
    # at a boundary.  It remains explicitly uncertain (rank zero), rather than
    # being silently fabricated into a chart frame or erased from the evidence.
    if not regular and poses:
        regular = [0]
    remap = {old: new for new, old in enumerate(regular)}
    regular_edges = [(remap[left], remap[right]) for left, right in edges if left in remap and right in remap]
    counts = {state: state_counts[state]
              for state in ("filled", "open", "unknown", "solver_failure", "boundary",
                            "undersized", "exhausted")}
    budget_exhausted = bool(frontier) and (options.max_samples is not None and len(poses) >= options.max_samples or
                                           options.max_seconds is not None and time.perf_counter() - started >= options.max_seconds)
    complete = not frontier and not counts["unknown"] and not counts["solver_failure"] and not budget_exhausted
    audit = [records[index] for index in regular]
    audit.extend({**record, "excluded_from_regular_training": True}
                 for index, record in records.items() if index not in remap)
    audit.append({"frontier_state_counts": counts, "frontier_complete": complete,
                  "budget_exhausted": budget_exhausted, "queued_cones": len(frontier),
                  "modal_regular_rank": modal_rank, "excluded_from_regular_training": True})
    checkpoint(budget_exhausted=budget_exhausted, complete=complete)
    log(f"physical_advancing_front: {len(regular)} regular samples, {len(regular_edges)} witnessed edges, complete={complete}, states={counts}")
    return (np.asarray([poses[index] for index in regular]), np.asarray(regular_edges, dtype=int).reshape(-1, 2),
            np.asarray([records[index]["frame"] for index in regular]), audit)
