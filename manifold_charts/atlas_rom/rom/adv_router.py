"""Net-only error-ratio ("advantage") chart router.

The learned field A(i, u, j) = log((e_i + eps) / (e_j + eps)) says how much better chart j reproduces the
true pose than staying in chart i (see ``adv_labels`` for the labels).  The default ("score") router outputs
one score per chart from the active chart's latent, s^(i)_k(u_i) ~ -log(e_k + eps), and A(i, u, j) =
s^(i)_j - s^(i)_i: the receivers form one ranking (no preference cycles) and A(i->j) = -A(j->i) up to the
build-time cross-chart consistency fit.  The legacy pairwise net (one A per (i, j) input) still loads.
The runtime (``PatchAtlasRuntime.step``)
switches to argmax_j A when A exceeds a dimensionless margin ``-log(switch_ratio)``; the step that crosses the
band is bisected so the handoff lands on the boundary.  Every decision comes from the net: no closure readout,
no tolerances, no joint geometry, no oracle at runtime.

The net is stored in the package as ``physics_fields["router"]`` (written by ``router_build.attach``) and loaded by
``PatchAtlasROM.load`` into ``rom.router``.  ``AdvRouterTrack`` is the viewer wrapper around the runtime.
"""
from __future__ import annotations

import numpy as np
import torch

from atlas_rom.rom import routing_test_tracks as T
from atlas_rom.rom.runtime import PatchAtlasRuntime


class Net(torch.nn.Module):
    def __init__(self, dmax, K, h=128):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(dmax + 2 * K, h), torch.nn.SiLU(),
                                     torch.nn.Linear(h, h), torch.nn.SiLU(),
                                     torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


class ScoreNet(torch.nn.Module):
    """(u_i / anchor-std, onehot(i)) -> K chart scores."""

    def __init__(self, dmax, K, h=128):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(dmax + K, h), torch.nn.SiLU(),
                                     torch.nn.Linear(h, h), torch.nn.SiLU(),
                                     torch.nn.Linear(h, h), torch.nn.SiLU(), torch.nn.Linear(h, K))

    def forward(self, x):
        return self.f(x)


def featurize(i, u, j, sc, dmax, K):
    """(u_i / anchor-std zero-padded to dmax, onehot(i), onehot(j))."""
    x = np.zeros(dmax + 2 * K)
    z = np.asarray(u, float) / sc[i]
    x[:len(z)] = z
    x[dmax + i] = 1.0
    x[dmax + K + j] = 1.0
    return x


def featurize_state(i, u, sc, dmax, K):
    """(u_i / anchor-std zero-padded to dmax, onehot(i)) -- the score net's input."""
    x = np.zeros(dmax + K)
    z = np.asarray(u, float) / sc[i]
    x[:len(z)] = z
    x[dmax + i] = 1.0
    return x


class AdvRouter:
    """The trained router net plus its feature scaling; ``rom.router``.  ``kind`` is "score" (default) or "pair"."""

    def __init__(self, net, dmax, K, sc, eps, kind="score"):
        self.net, self.dmax, self.K, self.sc, self.eps = net, int(dmax), int(K), sc, float(eps)
        self.kind = kind

    @classmethod
    def from_payload(cls, payload):
        kind = payload.get("kind", "pair")                 # payloads written before the score router carry no kind
        net = (ScoreNet if kind == "score" else Net)(payload["dmax"], payload["K"]).double()
        net.load_state_dict(payload["state"])
        net.eval()
        return cls(net, payload["dmax"], payload["K"], payload["sc"], payload["eps"], kind=kind)

    def scores(self, i, u):
        """Score router only: the K chart scores seen from chart i at latent u."""
        with torch.no_grad():
            return self.net(torch.as_tensor(featurize_state(i, u, self.sc, self.dmax, self.K)).unsqueeze(0))[0].numpy()

    def adv(self, i, u, j):
        if self.kind == "score":
            s = self.scores(i, u)
            return float(s[j] - s[i])
        with torch.no_grad():
            return float(self.net(torch.as_tensor(featurize(i, u, j, self.sc, self.dmax, self.K)).unsqueeze(0)))

    def best(self, c, u):
        """(best receiver j, its advantage A, {j: A} over all j != c)."""
        if self.kind == "score":
            s = self.scores(c, u)
            a = {j: float(s[j] - s[c]) for j in range(self.K) if j != c}
        else:
            a = {j: self.adv(c, u, j) for j in range(self.K) if j != c}
        j = max(a, key=a.get)
        return j, a[j], a


class AdvRouterTrack(T.Track):
    """Viewer track around ``PatchAtlasRuntime``: the runtime does the routing, the track exposes readouts."""
    label = "adv"
    switch_ratio = 1.0 / 1.1     # switch when the receiver's predicted error < switch_ratio x staying (own slider in the viewer)

    def _make_runtime(self):
        return PatchAtlasRuntime(self.rom, dt=self.settings.dt, stall_guard=self.settings.stall_guard)

    def reset(self):
        super().reset()
        rt = self.rt
        rt._since_handoff = 10 ** 9
        rt.router_info = None
        rt.router_ratio_hist.clear()

    # readouts the viewers use
    info = property(lambda self: self.rt.router_info)
    ratio_hist = property(lambda self: self.rt.router_ratio_hist)
    last_dH = property(lambda self: self.rt.last_dH)
    last_pop = property(lambda self: self.rt.last_pop)

    @property
    def last_scores(self):
        info = self.rt.router_info
        return {} if not info else {info["chart"]: 0.0, **info["scores"]}

    def diagnostics(self) -> dict:
        d = super().diagnostics()
        d["router_info"] = self.rt.router_info
        return d

    def _advance(self, controls) -> bool:
        rt = self.rt
        rt.switch_ratio, rt.dwell = float(self.switch_ratio), int(self.settings.dwell)
        before = self.state["handoffs"]
        ok = rt.step(self.state, controls)
        if self.state["handoffs"] != before and rt.last_handoff is not None:
            self._record_handoff(rt.last_handoff["q_pre"], rt.last_handoff["residual"])
        return ok
