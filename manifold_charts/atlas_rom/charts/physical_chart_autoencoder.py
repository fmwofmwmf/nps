"""Chart-conditioned physical-pose autoencoder; deliberately has no ``z`` input.

Each chart learns ``q -> E(q) in R^d -> D(E(q))`` directly from packed body
transforms (``d`` defaults to 2 but is caller-supplied via ``_module(..., d=...)``).  The decoder emits translation plus
a rotation per body, always on SO(3), never an unconstrained affine matrix.  Two rotation heads:

* ``"exp"``: a rotation vector ``w`` per body, ``R = exp(hat(w))`` (Rodrigues).  Smooth and defined for EVERY ``w``,
  and ``|dR| <= sqrt(2) |dw|`` everywhere, so the decoder Jacobian is bounded by the MLP's own Lipschitz constant at
  any latent, on or off the chart's data.
* ``"r6"``: two rotation columns, Gram--Schmidt.  Undefined where the second column is parallel to the first -- a
  codimension-2 set, so generically isolated points in a 2-D latent -- and the decoder Jacobian diverges there
  (clamped at 1e-8).  Kept so artifacts trained with it load and decode unchanged.
"""
from __future__ import annotations
import numpy as np

ROTATION_OUTPUTS = {"exp": 3, "r6": 6}


def rotation_head_of(state_dict, n_bodies):
    """The rotation head a saved chart state was trained with, read off the decoder's output width."""
    layers = sorted((int(k.split(".")[1]), k) for k in state_dict
                    if k.startswith("decoder.") and k.endswith(".weight"))
    per_body = int(state_dict[layers[-1][1]].shape[0]) // int(n_bodies)
    heads = {3 + r: name for name, r in ROTATION_OUTPUTS.items()}
    if per_body not in heads:
        raise ValueError(f"decoder emits {per_body} values per body; expected one of {sorted(heads)}")
    return heads[per_body]


def _module(n_bodies, width=96, d=2, encoder_kind="current", rotation_head="r6"):
    import torch
    import torch.nn as nn

    if rotation_head not in ROTATION_OUTPUTS:
        raise ValueError(f"unknown rotation_head {rotation_head!r}")
    n_rot = ROTATION_OUTPUTS[rotation_head]

    def mlp(a, b):
        return nn.Sequential(nn.Linear(a, width), nn.SiLU(), nn.Linear(width, width),
                             nn.SiLU(), nn.Linear(width, b)).double()

    class PhysicalChartAE(nn.Module):
        def __init__(self, centre, scale):
            super().__init__()
            self.n_bodies = n_bodies
            self.d = d
            self.register_buffer("centre", torch.as_tensor(centre).double())
            self.register_buffer("scale", torch.as_tensor(scale).double())
            if encoder_kind == "linear":
                self.encoder = nn.Linear(9 * n_bodies, d).double()
            elif encoder_kind == "shallow":
                # A single, spectrally well behaved hidden layer keeps the
                # inverse map simple while the decoder retains its capacity.
                self.encoder = nn.Sequential(nn.Linear(9 * n_bodies, width), nn.Tanh(),
                                             nn.Linear(width, d)).double()
            elif encoder_kind == "current":
                self.encoder = mlp(9 * n_bodies, d)
            else:
                raise ValueError(f"unknown encoder_kind {encoder_kind!r}")
            self.rotation_head = rotation_head
            self.decoder = mlp(d, (n_rot + 3) * n_bodies)
            # Optional detached whitening gauge applied to the raw code.  When
            # ``gauge_use`` is false the encoder keeps its bounded sigmoid
            # domain; competitive_cae fills these buffers so the *same* latent
            # map is used during training, field fitting and at runtime (the
            # state dict carries them, so no separate gauge payload is needed).
            self.register_buffer("gauge_use", torch.zeros((), dtype=torch.bool))
            self.register_buffer("gauge_mean", torch.zeros(d, dtype=torch.float64))
            # Orthogonal gauge: fixes the latent rotation/shear degeneracy only.
            # Scale is left to the loss (the pinned log-stretch reference), so
            # this factor is norm-1 by construction and cannot amplify or
            # degenerate as the support shrinks.
            self.register_buffer("gauge_map", torch.eye(d, dtype=torch.float64))
            # Keep the hot encode path free of device-scalar-to-Python
            # conversions.  The buffer remains authoritative for serialization;
            # this flag mirrors it at model construction/load/gauge refresh.
            self._gauge_enabled = False

        def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
            super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
            self._gauge_enabled = bool(self.gauge_use)

        def set_gauge(self, mean, gauge_map):
            """Install a latent gauge as a true coordinate reparameterization.

            ``encode`` returns ``(raw - mean) @ gauge_map``, so replacing the
            pair renames the coordinate the decoder is read in.  Carrying the
            decoder along is not optional: without it every previously decoded
            pose moves, which silently invalidates a trained chart (a refresh
            after training collapsed certified coverage 322 -> 119 on the
            sphere).  Writing ``u`` for the old code and ``u'`` for the new,

                raw = u' @ inv(G') + m'
                u   = u' @ (inv(G') @ G) + (m' - m) @ G  ==  u' A + b

            is affine, and an affine change of latent coordinates is absorbed
            exactly by the decoder's first linear layer: ``x A + b`` pushed
            through ``x W^T + c`` gives ``W <- W A^T`` and ``c <- c + W b``.
            The map is inverted with a solve rather than a transpose so the
            identity does not depend on the caller supplying an orthogonal
            gauge.  The *first* installation is not a reparameterization -- it
            replaces the bounded ``sigmoid`` code -- so nothing is carried.
            """
            with torch.no_grad():
                mean = torch.as_tensor(mean, dtype=self.gauge_mean.dtype,
                                       device=self.gauge_mean.device)
                gauge_map = torch.as_tensor(gauge_map, dtype=self.gauge_map.dtype,
                                            device=self.gauge_map.device)
                if self._gauge_enabled:
                    A = torch.linalg.solve(gauge_map, self.gauge_map)
                    b = (mean - self.gauge_mean) @ self.gauge_map
                    linear = next(x for x in self.decoder if isinstance(x, nn.Linear))
                    weight = linear.weight.clone()
                    linear.weight.copy_(weight @ A.T)
                    linear.bias.add_(weight @ b)
                self.gauge_mean.copy_(mean)
                self.gauge_map.copy_(gauge_map)
                self.gauge_use.fill_(True)
                self._gauge_enabled = True

        @staticmethod
        def _rotation6(x):
            a, b = x[..., :3], x[..., 3:]
            e1 = a / torch.linalg.vector_norm(a, dim=-1, keepdim=True).clamp_min(1e-8)
            b = b - (e1 * b).sum(-1, keepdim=True) * e1
            e2 = b / torch.linalg.vector_norm(b, dim=-1, keepdim=True).clamp_min(1e-8)
            return torch.stack((e1, e2, torch.linalg.cross(e1, e2, dim=-1)), dim=-1)

        @staticmethod
        def _rotation_exp(w):
            """Rodrigues ``exp(hat(w)) = I + A hat(w) + B hat(w)^2``, Taylor series for A, B near ``w = 0`` so value
            and gradient are finite everywhere."""
            theta2 = (w * w).sum(-1, keepdim=True)[..., None]                   # [..., 1, 1]
            small = theta2 < 1e-4
            safe2 = torch.where(small, torch.ones_like(theta2), theta2)
            theta = torch.sqrt(safe2)
            A = torch.where(small, 1.0 - theta2 / 6.0 + theta2 * theta2 / 120.0, torch.sin(theta) / theta)
            B = torch.where(small, 0.5 - theta2 / 24.0 + theta2 * theta2 / 720.0, (1.0 - torch.cos(theta)) / safe2)
            x, y, z = w.unbind(-1)
            o = torch.zeros_like(x)
            K = torch.stack((torch.stack((o, -z, y), -1),
                             torch.stack((z, o, -x), -1),
                             torch.stack((-y, x, o), -1)), -2)
            eye = torch.eye(3, dtype=w.dtype, device=w.device)
            return eye + A * K + B * (K @ K)

        def _rotation(self, x):
            return self._rotation_exp(x) if self.rotation_head == "exp" else self._rotation6(x)

        def features(self, q):
            T = q.reshape(-1, self.n_bodies, 3, 4)
            # First two *columns* are the continuous 6D rotation representation.
            r6 = T[..., :3, :2].transpose(-1, -2).reshape(len(T), self.n_bodies, 6)
            return torch.cat((r6, (T[..., 3] - self.centre) / self.scale), -1).reshape(len(T), -1)

        def encode_raw(self, q): return self.encoder(self.features(q))
        # Default users retain a bounded coordinate domain (sigmoid).  When a
        # detached whitening gauge has been installed (competitive CAE) the
        # exact same responsibility-local affine map is applied here, so every
        # consumer that calls ``encode`` sees the trained latent coordinate.
        def encode(self, q):
            raw = self.encode_raw(q)
            if self._gauge_enabled:
                mean = self.gauge_mean.to(raw.dtype)
                gauge_map = self.gauge_map.to(raw.dtype)
                return (raw - mean) @ gauge_map
            return torch.sigmoid(raw)

        def transforms(self, u):
            y = self.decoder(u).reshape(-1, self.n_bodies, n_rot + 3)
            R = self._rotation(y[..., :n_rot])
            t = y[..., n_rot:] * self.scale + self.centre
            return torch.cat((R, t[..., None]), -1)

        def decode(self, u): return self.transforms(u).reshape(len(u), -1)

        def forward(self, q, *, decode_only=False, encode_only=False):
            """Joint encoder/decoder forward used by dense chart ensembles.

            ``decode_only`` keeps decode calls inside ``torch.func.functional_call``
            so a persistent vmapped parameter ensemble can own both halves of
            the chart-local objective.
            """
            if decode_only:
                return self.decode(q)
            if encode_only:
                return self.encode(q)
            u = self.encode(q)
            return u, self.transforms(u)
    return PhysicalChartAE


def fit(q, edges, spec, *, epochs=5000, width=96, seed=0,
        closure_weight=0.0, cycle_weight=.1, edge_weight=.0,
        interpolation_weight=.5, interpolation_samples=256,
        closure_warmup_fraction=.2, initial_state=None, normalization=None,
        report_groups=None):
    """Fit a physical chart with held-out vertices *and* held-out graph edges."""
    import torch
    from atlas_rom.charts import chart_metric
    from atlas_rom.charts.validity import closure_attachments, closure_residual
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    q = np.asarray(q, float); edges = np.asarray(edges, int)
    n, n_bodies = len(q), q.shape[1] // 12
    T = q.reshape(n, n_bodies, 3, 4)
    if normalization is None:
        centre, scale = T[..., 3].mean((0, 1)), T[..., 3].std((0, 1)).clip(1e-6)
    else:
        centre, scale = (np.asarray(normalization[0], float),
                         np.asarray(normalization[1], float).clip(1e-6))
    model = _module(n_bodies, width)(centre, scale)
    if initial_state is not None:
        # Preserve the new patch's physical normalization, but continue the
        # same inverse pair as graph rings are added.
        initial_state = {k: v for k, v in initial_state.items()
                         if k not in ("centre", "scale")}
        model.load_state_dict(initial_state, strict=False)
    Q = torch.as_tensor(q).double(); attachment = closure_attachments(spec)
    held = np.zeros(n, bool); held[rng.permutation(n)[:max(1, n // 5)]] = True
    held_edge = np.zeros(len(edges), bool); held_edge[rng.permutation(len(edges))[:max(1, len(edges) // 5)]] = True
    train_edges = torch.as_tensor(edges[~held_edge], dtype=torch.long)
    train_vertices = torch.as_tensor(~held)
    with torch.no_grad():
        feature = model.features(Q)
        physical_length = torch.linalg.vector_norm(feature[train_edges[:, 0]] - feature[train_edges[:, 1]], dim=1)
        physical_length = physical_length / physical_length.mean().clamp_min(1e-8)
    # Deliberately small, weight-decayed networks supply the bounded-smoothness
    # assumption behind dense-sample interpolation training.
    opt = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-4)
    warmup = max(1, int(float(closure_warmup_fraction) * epochs))
    for step in range(epochs):
        opt.zero_grad(); u = model.encode(Q); pred = model.transforms(u)
        target = Q.reshape(n, n_bodies, 3, 4)
        # SO(3) Frobenius + scale-normalised translation, only train vertices.
        rot = (pred[..., :3] - target[..., :3]).square().mean((-1, -2, -3))
        trn = ((pred[..., 3] - target[..., 3]) / torch.as_tensor(scale)).square().mean((-1, -2))
        loss = (rot[train_vertices] + trn[train_vertices]).mean()
        # First learn the physical branch.  A full closure penalty at random
        # initialization can otherwise converge to a closed but unrelated pose.
        closure_scale = float(closure_weight) * min(1.0, (step + 1) / warmup)
        if attachment is not None:
            loss = loss + closure_scale * closure_residual(pred[train_vertices], attachment).mean()
        cycle = (model.encode(pred.reshape(n, -1)) - u).square().mean()
        # Fill the current chart domain through interpolations of physical
        # neighbours, rather than cycling only the encoded sample vertices.
        take = min(int(interpolation_samples), len(train_edges))
        chosen = train_edges[torch.randperm(len(train_edges))[:take]]
        t = torch.rand(take, 1, dtype=Q.dtype)
        interior = (1 - t) * u[chosen[:, 0]] + t * u[chosen[:, 1]]
        interpolation = (model.encode(model.decode(interior)) - interior).square().mean()
        latent_length = torch.linalg.vector_norm(u[train_edges[:, 0]] - u[train_edges[:, 1]], dim=1)
        latent_length = latent_length / latent_length.mean().clamp_min(1e-8)
        loss = loss + cycle_weight * cycle + interpolation_weight * interpolation
        loss = loss + edge_weight * (latent_length - physical_length).square().mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        u = model.encode(Q); pred = model.transforms(u); target = Q.reshape(n, n_bodies, 3, 4)
        trn = torch.linalg.vector_norm(pred[..., 3] - target[..., 3], dim=-1)
        rel = target[..., :3].transpose(-1, -2) @ pred[..., :3]
        ang = torch.rad2deg(torch.acos(((torch.diagonal(rel, dim1=-2, dim2=-1).sum(-1)-1)/2).clamp(-1, 1)))
        closure = (closure_residual(pred, attachment).sqrt()
                  if attachment is not None else None)
        reference, medoid = chart_metric.medoid_reference(q, n_bodies)
        J = chart_metric.twist_jacobian(model.transforms, u, torch.as_tensor(reference).double())
        condition = chart_metric.condition_report(J)
        cycle = torch.linalg.vector_norm(model.encode(pred.reshape(n, -1)) - u, dim=1)
        e_all = torch.as_tensor(edges, dtype=torch.long)
        midpoint = .5 * (u[e_all[:, 0]] + u[e_all[:, 1]])
        interpolation_cycle = torch.linalg.vector_norm(model.encode(model.decode(midpoint)) - midpoint, dim=1)
        # A held-out edge reports latent-vs-physical length correlation, not a z metric.
        e = torch.as_tensor(edges[held_edge], dtype=torch.long)
        dl = torch.linalg.vector_norm(u[e[:, 0]] - u[e[:, 1]], dim=1)
        dp = torch.linalg.vector_norm(model.features(Q)[e[:, 0]] - model.features(Q)[e[:, 1]], dim=1)
        corr = torch.corrcoef(torch.stack((dl, dp)))[0, 1]
        # Mechanism-scale reference: median per-body translation change across
        # the retained local physical-neighbour graph.
        edge_translation = torch.linalg.vector_norm(
            target[e_all[:, 0], :, :, 3] - target[e_all[:, 1], :, :, 3], dim=-1).median()
    metrics = {"heldout_vertices": int(held.sum()), "heldout_edges": int(held_edge.sum()),
               "heldout_translation_rms": float(trn[held].square().mean().sqrt()),
               "heldout_translation_rms_over_median_edge": float(
                   trn[held].square().mean().sqrt() / edge_translation.clamp_min(1e-12)),
               "heldout_rotation_deg_mean": float(ang[held].mean()),
               "cycle_rms": float(cycle.square().mean().sqrt()),
               "edge_interpolation_inverse_rms": float(interpolation_cycle.square().mean().sqrt()),
               "heldout_edge_length_correlation": float(corr),
               "reference_medoid": int(medoid)}
    if closure is not None:
        metrics["max_joint_closure"] = float(closure.max())
    metrics.update({f"twist_{k}": v for k, v in condition.items()})
    if report_groups:
        for name, indices in report_groups.items():
            indices = np.asarray(indices, int)
            if len(indices) == 0: continue
            metrics[f"{name}_translation_rms"] = float(trn[indices].square().mean().sqrt())
            metrics[f"{name}_rotation_deg_mean"] = float(ang[indices].mean())
    return model.eval(), metrics
