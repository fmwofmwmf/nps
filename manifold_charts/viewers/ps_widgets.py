"""ps_widgets.py — reusable polyscope/imgui pieces shared by the ROM viewers.

Nothing here knows about ROMs, tracks or routers.  ``FrameRecorder`` keeps one
opaque snapshot per rendered frame (the only required key is ``"time"``) and
draws the replay controls; the caller decides what a frame contains and how to
draw one (``on_show``).  ``energy_panel`` plots ``frame["diag"][name][attr]``.
"""
from __future__ import annotations

import collections

import numpy as np


def translation(t):
    m = np.eye(4)
    m[:3, 3] = t
    return m


def fmt(x, spec="+.4f"):
    return "—" if x is None else format(x, spec)


class FrameRecorder:
    """Live replay buffer.  Replay only reads the buffer; the simulation state is never touched,
    so "Resume live" continues exactly where it stopped.  ``idx is None`` means live."""

    def __init__(self, maxlen=3000):
        self.frames = collections.deque(maxlen=max(int(maxlen), 2))
        self.idx, self.playing, self.speed, self.acc = None, False, 1.0, 0.0   # speed: frames per redraw, <0 = backwards

    def __len__(self):
        return len(self.frames)

    @property
    def live(self):
        return self.idx is None

    @property
    def maxlen(self):
        return self.frames.maxlen

    def push(self, frame):
        self.frames.append(frame)

    def clear(self):
        self.frames.clear()
        self.idx, self.playing = None, False

    def shown(self):
        """The frame currently on screen (the replay frame, else the newest)."""
        return self.frames[self.idx] if self.idx is not None else self.frames[-1]

    def panel(self, imgui, on_show, pause_live=None):
        """Replay controls.  ``on_show(frame)`` redraws a frame; ``pause_live()`` is called when entering replay."""
        n = len(self.frames)
        if self.idx is None:
            if imgui.Button("Replay") and n > 1:
                if pause_live is not None:
                    pause_live()
                self.idx, self.playing = n - 1, False
            imgui.SameLine()
            imgui.Text(f"live  (buffer {n}/{self.maxlen})")
            return
        i = self.idx = min(self.idx, n - 1)
        if imgui.Button("Resume live"):
            self.idx, self.playing = None, False
            on_show(self.frames[-1])
            return
        imgui.SameLine()
        if imgui.Button("Pause replay" if self.playing else "Play replay"):
            self.playing = not self.playing
        imgui.SameLine()
        if imgui.Button("<"):
            i, self.playing = max(i - 1, 0), False
        imgui.SameLine()
        if imgui.Button(">"):
            i, self.playing = min(i + 1, n - 1), False
        if self.playing:
            # Fractional speeds accumulate, so 0.1x advances one frame per 10 redraws.
            self.acc += self.speed
            whole = int(self.acc)
            self.acc -= whole
            i = min(max(i + whole, 0), n - 1)
            if i in (0, n - 1) and whole != 0:
                self.playing = False
        changed, v = imgui.SliderFloat("replay speed (frames/redraw)", self.speed, -8.0, 8.0)
        if changed:
            self.speed = float(v)
        changed, v = imgui.SliderInt("replay frame", i, 0, n - 1)
        if changed:
            i, self.playing = int(v), False
        self.idx = i
        on_show(self.frames[i])
        imgui.TextColored((0.4, 0.8, 1.0, 1.0),
                          f"REPLAY frame {i}/{n - 1}  t={self.frames[i]['time']:.4f}  "
                          f"(paused; Resume live continues the simulation)")


ENERGY_Q = (("total", "energy", "total energy (change over the window)"),
            ("kinetic", "kinetic", "kinetic energy"),
            ("potential", "potential", "potential energy (change over the window)"))


def energy_panel(imgui, rec, names, enabled, ui):
    """Energy of every enabled track over the recorded frames (follows the replay frame).  Total / potential are shown as a change from
    the first frame of the window: the absolute levels are huge (V ~ m g h) and a handoff teleport shows up as a jump.
    ``ui`` carries ``energy_q`` and ``energy_window``."""
    if not rec.frames:
        return
    imgui.Separator()
    imgui.Text("energy")
    for key, _attr, _label in ENERGY_Q:
        imgui.SameLine()
        if imgui.RadioButton(key, ui["energy_q"] == key):
            ui["energy_q"] = key
    attr, label = next((a, l) for k, a, l in ENERGY_Q if k == ui["energy_q"])
    end = rec.idx if rec.idx is not None else len(rec.frames) - 1
    end = min(end, len(rec.frames) - 1)
    start = max(0, end - ui["energy_window"] + 1)
    for n in names:
        if not enabled[n]:
            continue
        vals = np.array([rec.frames[k]["diag"][n][attr] for k in range(start, end + 1)], dtype=float)
        if attr != "kinetic":
            vals = vals - vals[0]
        handoffs = rec.frames[end]["diag"][n]["handoffs"]
        cur = vals[-1]
        try:
            imgui.PlotLines(f"{n}  {label}##energy_{n}", vals.astype(np.float32), graph_size=(0, 60),
                            overlay_text=f"{cur:+.3g}   (handoffs {handoffs})")
        except Exception:
            imgui.Text(f"{n}: {cur:+.3g}")
