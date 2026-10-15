## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

### State-only evidence policy

Graphify is an index of repository state, not a source of opinions or conclusions.

- Use Graphify to locate relevant code and artifacts, then verify every conclusion against the actual source or persisted data.
- Treat only structurally extracted facts such as symbols, calls, imports, inheritance, definitions, and artifact contents as state evidence.
- Do not use `INFERRED` edges, rationale nodes, community labels, suggested questions, surprising connections, generated reports, reflections, plans, notes, or saved Q&A as evidence about current behavior.
- Do not run `graphify save-result` or otherwise write interpretive answers, recommendations, or conversation memory back into the graph.
- Do not use `graphify reflect` or `graphify-out/reflections/LESSONS.md` to guide codebase conclusions.
- Keep `graphify-out/**` excluded from graph input so generated graph material cannot recursively become repository evidence.
- Prefer AST-only updates. Do not perform semantic extraction for codebase-state queries.

### Execution environment

For this project, use the repository-managed UV environment:
`.\.venv\Scripts\python.exe`.

Do not use global Python, bundled Codex Python, or unrelated Conda environments
for project runs unless the user explicitly authorizes it.

### Physical chart atlas: methods before gates

For the learned physical-chart pipeline, a failed diagnostic, certificate, or
runtime gate is evidence of a problem, never its fix.  Do not respond to an
AE/transition aliasing or folding failure by adding another routing model,
lookup, or rejection gate as the primary solution.  Fix the chart method:
train off-anchor decoder probes to remain on the local physical manifold,
train AE overlap replay across local source neighbourhoods, enforce physical
separation in latent coordinates, and automatically shrink/split charts that
cannot satisfy those properties.  Runtime must remain learned-field/AE based
with no topology declaration or training-data membership lookup.

### Physical chart atlas: strict no-priors / no-cheating policy

Physical chart CAE training is forbidden unless the input dataset audit has
`frontier_complete: true`.  A dataset produced under a sampler time budget,
with `budget_exhausted: true` or any open/unknown frontier states, is invalid
for training regardless of its sample count or other passing diagnostics.

Core pipeline procedures may receive only simulator-produced frames and
physical evidence.  System parameters may contain whatever information is
necessary to define or operate the physical system, but that information must
not be communicated to the learning pipeline as declared dimensions,
coordinates, topology, tangent frames, chart counts, memberships, routing
labels, or equivalent shortcuts.

In particular, do not manufacture identity tangent frames, set learned chart
coordinates equal to system state (`u = q`), copy a declared state dimension
into the latent dimension, force a single/global chart, initialize a chart to
the intended answer, or use case/system declarations to decide chart structure.
The pipeline must infer its chart structure from simulator-produced frames and
other permitted physical observations.

Only the user may explicitly authorize an experimental exception.  Agents may
not mark work, code, tests, artifacts, or execution paths as experimental,
development-only, or testing-only in order to bypass production requirements.
Unless the user explicitly says otherwise, all code and all paths are
production and must obey the strict no-priors/no-cheating rules.

A passing test, diagnostic, certificate, runtime gate, or benchmark does not
legitimize a result obtained with prohibited priors.  Such a result is invalid
and must be reported as invalid; fix the underlying method rather than changing
the label or gate.

### Explicit coding permission and action preview

Read-only inspection is pre-approved.  Before any workspace-changing action,
state the specific intended change and obtain the user's explicit confirmation.

Before making any code, configuration, artifact-generation, or other workspace
change, state exactly what will be done and obtain the user's explicit
permission to do it.  Do not infer permission from a diagnosis, discussion,
or request for explanation.  Before every non-trivial action, communicate the
intended action and its purpose first; wait for permission before coding or
running a change-producing command. 

### Summary, experiment, and tool-use preferences

When a request leaves any method open, summaries and previews must state the
specific method before any tool is used. This applies to presentation choices
(for example, text chart versus PNG versus inline visual, including aggregation,
axes, labels, and comparison method) and to experiments, benchmarks, tests,
data generation, environment changes, and other non-trivial executions.

For visualization requests that require new code, first provide a concrete plan
and obtain explicit permission before taking action. Do not load or use
visualization skills unless the user explicitly authorizes them.

For experiments and executions, state the exact inputs, parameters, comparison
method, expected outputs, and whether files or the environment will change. Do
not assume unspecified details from context. Use only tools and methods the user
has explicitly requested or approved; if a material choice is unspecified,
describe the proposed method and wait for approval before using a tool that
depends on it.

### Roles: analyst vs experiment runner

This project runs as analyst <-> user <-> experiment runner.

- **Analyst seat** (the default when the user pastes a handoff or results
  summary plus a goal): read existing artifacts (reports, checkpoints,
  datasets, logs) freely and work out what they mean.  Do not launch training
  runs, write runner scripts, or "just check" with a background job.  Write a
  handoff the runner can execute without shared context: exact config and
  flags, epochs and stopping criterion, what to save, what to measure, and
  what each outcome would mean.  State mechanism, evidence, exact change, and
  blast radius, then stop.
- **Runner seat** (an implementation or run instruction, including a bare
  imperative such as "try X" or "stage Y"): do exactly that and nothing
  adjacent.  "Fix X" means edit, run the test suite, at most one quick
  correctness smoke, then stop and report.  Do not launch validation matrices,
  sweeps, multi-seed comparisons, or re-run arms the user did not name.  Do not
  add diagnostic analysis, write memories, or create side artifacts unless
  told.  When unsure whether something is in scope, it is not.

### General pipeline: no per-case shortcuts

The ROM / chart pipeline must treat every case (puck, five_bar, gear, ...) as
generally as possible.  Per-case shortcuts (posing a body directly and
bypassing the decoder, hardcoded barrier geometry, hand-derived mass or
potential for one case) are allowed only as a clearly labelled ground-truth
reference, never as the main path.  When something looks wrong, fix the
general machinery.

After every experiment, report two separate buckets: (1) manual /
case-specific decisions that must be re-decided per case, and (2) general
pipeline changes to shared code that now apply to every case.

### Runtime: no gates, no full-space quantities

- Never hide a runtime failure behind a gate: no hold-instead-of-switch,
  acceptance caps, stall guards, display-side smoothing, energy rescaling at
  handoff, or pose-offset fading.  Fix the upstream cause (coverage, switch
  timing, transport metric, labels).  Counters and readouts that expose a
  failure are fine.
- The shipped ROM runs on its learned fields alone.  No masses, inertia, COM,
  analytic mass matrix, decoder-Jacobian mass, oracle, or full simulator in
  the runtime / router / handoff path.  Anything like that must become a
  build-time label distilled into a learned field.  Flag such a need before
  editing.

### Deliverables and visualization

- Do not create Artifacts, HTML pages, or other side deliverables unless
  asked.  Plans, reviews, and write-ups go in the terminal as plain text.
- On any vis / plot / render request with no script named, ask which existing
  script to use (`atlas_rom/view_sample_set.py`, `view_samples.py`,
  `mesh_view.py`, `generic_manifold_diagnostics.py`, per-case `VIEW_EMBED` /
  `VISUAL_UVS` hooks).  Never improvise matplotlib.  A new visualization script
  is a code change that needs a plan and approval.

### Run hygiene

- Wrap every long or background run, especially CUDA libuipc, in an OS-level
  timeout (`timeout --kill-after=10 <secs> ...`).  A stuck `world.advance()`
  cannot be interrupted from Python; only killing the process works.
- Every training / build / rollout entrypoint under `atlas_rom/` writes a
  `timing` block into its report JSON: per-stage seconds,
  `started_at` / `finished_at` ISO timestamps, and total `wall_seconds`
  (see `patch_atlas.py` `build()` / `main()` for the pattern).
- `competitive_cae` device choice: batched / parallel sweeps use
  `device="cuda"`; single one-off runs use `device="cpu"`.
