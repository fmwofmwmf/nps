# Graph Report - libuipc-samples  (2026-10-15)

## Corpus Check
- 333 files · ~917,957 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 115 file(s) not represented in the graph (top: .pt 44, .obj 36, .npz 13)

## Summary
- 3178 nodes · 5208 edges · 241 communities (153 shown, 81 thin omitted)
- Extraction: 98% EXTRACTED · 2% INFERRED · 0% AMBIGUOUS · INFERRED: 120 edges (avg confidence: 0.89)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `fbc13be4`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Expectations
- view_pantograph_drag.py
- SDFNet
- MeshSpec
- CompetitiveCAEFitter
- RomSim
- generic_manifold_diagnostics.py
- physical_pose.py
- generic_genus4.py
- competitive_cae.py
- AdvRouter
- PatchAtlasRuntime
- AssetDir
- selfcheck.py
- SamplingOptions
- run_patch_atlas.py
- sampling.py
- load_spec
- PhysicsFitter
- geometry.py
- jansen_patch_rom/manifest.json
- stephenson_link_sep9/manifest.json
- stephenson_link_single_chart/manifest.json
- ._fit_inner
- RomBackend
- 88_maltese_cross/main.py
- AdvRouterTrack
- quotient.py
- validity_from_case
- generic_sphere_cae/manifest.json
- generic_sphere/manifest.json
- jansen_analytic_full_experimental/manifest.json
- jansen_analytic_handoff_candidate/manifest.json
- jansen_analytic_nonfold_coverage_experimental/manifest.json
- jansen_analytic_nonfold_experimental/manifest.json
- jansen_analytic_replay_experimental/manifest.json
- sarrus_stack/manifest.json
- scissor_lift_stack/manifest.json
- ManifoldOracle
- new_dataset.py
- jansen_walker_trial10v3/manifest.json
- scissor_lift/manifest.json
- scissor_lift_stack_dense/manifest.json
- twin_independent_pistons/manifest.json
- ndarray
- view_chart_exploration.py
- _points
- local_support_radii
- learned.py
- build_mesh
- test_generic_so3.py
- view_sample_set.py
- main
- .jacobian
- PositionAtlasRuntime
- PatchAtlasROM
- _module
- AtlasCaseSpec
- stephenson_link.py
- chart_contract.py
- ProjectionLedger
- lazy_tongs_pantograph.py
- fit_support_and_boundary
- diag_utils.py
- _ToyAE
- encode
- fit_dynamics
- scissor_lift.py
- build_oracle
- IdealJointSim
- SDFDirectionalOracle
- mobius_kaleidocycle.py
- rom_target.py
- ops.py
- FullBackend
- dataset.py
- SDFPhysicalOracle
- _restrict_to_components
- explore
- SampleSetView
- physical_chart_autoencoder.py
- _m_step
- LegacyPhysicalOracleAdapter
- oracle_eval.py
- sample_graph.py
- LibuipcConfigOracle
- loader.py
- jansen.py
- MassField
- FullSpaceSim
- PhysicalRBFSelector
- CoordinateMap
- adv_labels.py
- test_obstacle_constrained_2r.py
- sarrus_stack.py
- pen_pd_controls
- stewart_platform.py
- CylindricalReducedOracle
- validity.py
- registry.py
- Competitive CAE Mechanism Test Suite
- ClosureValidity
- joint_travel
- Frame 1: cube tumbling forward on near-edge, boards shifting right
- test_support_sdf_derives_its_witness_radius_from_the_anchor_cloud
- joint_travel
- _stacked_ensemble
- load_sdf_system
- enforce_support_coverage
- rigid_ipc/README.md
- charts/__init__.py
- sample/__init__.py
- simulate/__init__.py
- spec/__init__.py
- Reference
- MechanismSpec
- IdealJointOracle
- Rack-and-pinion meshing via IPC contact (rotating gear driving a linear toothed rack)
- Rain
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- VIEW_EMBED
- State accessor: ABD transforms/velocities & FEM positions
- _Progress
- Deformable triangle-mesh soft body (bunny)
- jansen_walker.py
- Advanced Scene Configuration
- Subscene partitioning / interpenetrating regions
- solver_spec.py
- UID (Unique Identifier)
- Interactive user-controlled particle & rope gameplay
- test_competitive_cae.py
- Compute Mesh d_hat
- case_pipeline_pruned.py
- keyboard (dependency)
- trimesh (dependency)
- AssetDir
- Check Libuipc
- CUDA Backend Engine
- DiscreteShellBending
- ElasticModuli
- NeoHookeanShell
- SceneSnapshot
- SceneSnapshotCommit
- Extract Surface from Tet Mesh
- uipc.Future
- ConfigAttributes
- Subscene Tabular
- AffineBodyStateAccessorFeature
- FiniteElementStateAccessorFeature
- ExternalArticulationConstraint
- Prismatic Joint
- Revolute Joint
- User Scriptable Animator
- RotatingMotor
- SoftTransformConstraint
- Recovery and Dump
- Libuipc
- pyuipc
- uv (package manager)
- 84_abd_revolute_joint_local/main.py
- 85_abd_prismatic_joint_local/main.py
- AssetDir
- MyEngine
- 87_scissor_lift_joint/main.py
- 86_panda_hydro_traj_cubes/main.py
- AssetDir
- AssetDir
- AssetDir
- graphify
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- AssetDir
- Q: Fixed-support micro-atlas pipeline ran end-to-end and halted with 18 of 94 uncertified patches. Decide whether to investigate large patches, widen the network, relax LIMITS, or use another repair before continuing.
- Q: summarize the current direction; it feels like we may be overcomplicating things
- Q: for sdf boundary cases: make the clamp project along phi_M^-1 grad phi_s, so all three boundary paths agree on the metric. C we should do this for correctness right
- Q: read both docs in /mapper, thoughts?
- Q: any new ideas or ideas from literature we missed?
- Q: read B, what do you think
- Q: read C, lets see whats missing still
- Q: The ROM jumps because its charts are not yet valid local coordinate charts everywhere they claim support. At a bad 1 to 0 transition, is this an encoder collision caused by weak overlap replay? Thoughts for latest Jansen ROM with high transition deltas?
- Q: What should change in the training process to eliminate the Jansen chart encoder collision while preserving the existing two-chart split?
- Q: How can we construct the entire reachable directed handoff collar generically, without case-specific knowledge?
- Q: Should support be derived from map reconstruction or transition success?
- Install (uv recommended)
- State Accessor Feature
- External Articulation Constraint Example
- 26_bunny_cloth_async/README.md
- Wrecking Ball
- 0_check_libuipc/README.md
- 10_ramp_sliding/README.md
- 11_bunny_cloth/README.md
- 15_scene_commit/README.md
- 16_geometry_operation/README.md
- 18_pendulum_joint/README.md
- 27_compute_mesh_d_hat/README.md
- 28_advanced_scene_config/README.md
- 29_subscene/README.md
- 30_check_uid/README.md
- 4_tiny_video_game/README.md
- 5_walking_cube_moving_board/README.md
- 7_simple_gear/README.md
- 8_screw_and_nut/README.md
- 9_pendulum/README.md
- examples/README.md
- libuipc-samples

## God Nodes (most connected - your core abstractions)
1. `MeshSpec` - 81 edges
2. `BodyDef` - 79 edges
3. `MechanismSpec` - 60 edges
4. `AtlasCaseSpec` - 59 edges
5. `SamplingOptions` - 48 edges
6. `BuildOptions` - 47 edges
7. `CoordinateDef` - 45 edges
8. `PatchAtlasRuntime` - 43 edges
9. `load_spec()` - 41 edges
10. `JointDef` - 41 edges

## Surprising Connections (you probably didn't know these)
- `margins()` --indirect_call--> `supports()`  [INFERRED]
  manifold_charts/atlas_rom/rom/learned.py → manifold_charts/atlas_rom/simulate/ideal_oracle.py
- `create_ddf()` --uses--> `AssetDir`  [INFERRED]
  examples/40_distance_diagnoser/diag_utils.py → examples/0_check_libuipc/asset_dir.py
- `make_abd_cube()` --uses--> `AssetDir`  [INFERRED]
  examples/84_abd_revolute_joint_local/main.py → examples/0_check_libuipc/asset_dir.py
- `make_abd_cube()` --uses--> `AssetDir`  [INFERRED]
  examples/85_abd_prismatic_joint_local/main.py → examples/0_check_libuipc/asset_dir.py
- `make_abd_cube()` --uses--> `AssetDir`  [INFERRED]
  examples/86_panda_hydro_traj_cubes/main.py → examples/0_check_libuipc/asset_dir.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Continuous sequence: cube tumbles across rightward-translating boards** — examples_5_walking_cube_moving_board_0_frame, examples_5_walking_cube_moving_board_1_frame, examples_5_walking_cube_moving_board_2_frame, examples_5_walking_cube_moving_board_3_frame [INFERRED 0.75]
- **Simple gear rack-and-pinion meshing simulation frame sequence** — examples_7_simple_gear_0_frame, examples_7_simple_gear_1_frame, examples_7_simple_gear_2_frame, examples_7_simple_gear_3_frame [INFERRED 0.85]

## Communities (241 total, 81 thin omitted)

### Community 0 - "Expectations"
Cohesion: 0.06
Nodes (48): A Bennett linkage: the classical spatial overconstrained 4R loop. Grubler's…, _rod(), Compact four-DOF SCARA with three continuous bearings and a vertical slide., _bar(), cases/five_bar.py — a planar 5R five-bar linkage with ideal joints, no contact.…, Display-only 3-D unfolding of the five-bar C-space for ``view_sample_set``. The…, VIEW_EMBED(), _panel() (+40 more)

### Community 1 - "view_pantograph_drag.py"
Cohesion: 0.06
Nodes (55): _forward(), ndarray, All five named points from the two independent joint angles. `C` is the…, MeshAdapter, mesh_view.py — generic rigid-mesh rendering for any declared mechanism.…, Draw any rigid-mesh mechanism: static bodies once, driven bodies per frame., _anchor_positions_and_poses(), jacobian_point() (+47 more)

### Community 2 - "SDFNet"
Cohesion: 0.19
Nodes (7): fit_constraint_energy_sdf(), load_sdf(), Regress decoded physical constraint energy directly in latent space. Unlike a…, Small signed-distance network used by packaged physical-AE ROMs., Latent coordinate to signed distance, with in-model normalization., SDFNet, no_grad

### Community 3 - "MeshSpec"
Cohesion: 0.06
Nodes (47): _bar(), cases/double_pendulum.py — a planar 2R open chain under gravity: the classic…, _gear_body(), The ``gears_1`` planetary train, migrated to the generic Atlas ROM build. The…, _body(), Rigid IPC ``398-intermitten-motion``: a curved wheel intermittently drives a…, Display-only observed angles of the wheel and the end piece., VIEW_EMBED() (+39 more)

### Community 4 - "CompetitiveCAEFitter"
Cohesion: 0.14
Nodes (25): ChartFitConfig, ChartFitInput, Simulator-produced evidence visible to a chart fitter. Deliberately absent are…, Method-owned fitting options; no physical prior is encoded here., CompetitiveCAEFitter, encoder_ablation(), Run the approved encoder comparison on identical physical evidence. This is…, Two-phase competitive charts. Phase 1 (cover) grows a topological cover under… (+17 more)

### Community 5 - "RomSim"
Cohesion: 0.22
Nodes (3): Smooth periodic spline through the drawn points, resampled uniformly in…, Move the reference point along the curve at the constant trace speed., RomSim

### Community 6 - "generic_manifold_diagnostics.py"
Cohesion: 0.13
Nodes (33): _artifact(), _atlas_view(), _draw_samples(), _latent_embeddings(), _limits(), _load_json(), _plot_charts(), _plot_contacts() (+25 more)

### Community 7 - "physical_pose.py"
Cohesion: 0.16
Nodes (13): medoid_reference(), The chart's medoid sample, as a ``(n_bodies, 3, 4)`` reference pose. The medoid…, physical_distance_matrix(), pose_distance_matrix(), pose_twists(), Physical SE(3) coordinates and distances used before atlas learning. This…, Principal rotation vector; stable also for planar five-bar poses., SE3Lift(reference pose + body twists) with rotations valid by construction. (+5 more)

### Community 8 - "generic_genus4.py"
Cohesion: 0.09
Nodes (20): Planar disk: the contractible two-dimensional control., _genus2_sdf(), Closed genus-2 surface: one smooth union of two ordinary tori., _genus3_sdf(), Closed genus-3 surface: a smoothly rounded tetrahedral wireframe., _genus4_sdf(), Closed genus-4 surface: rounded cube frame with one square face filled., Pair of pants: a planar disk with two disjoint physical holes. (+12 more)

### Community 9 - "competitive_cae.py"
Cohesion: 0.09
Nodes (30): _bridge_filtered_neighbours(), _assign_from(), _exclusive_support(), _integer_multiplicity_histogram(), _jacobian_fold_ratio(), _load_resume_checkpoint(), _ownership_weight(), _patch_far_distance() (+22 more)

### Community 10 - "AdvRouter"
Cohesion: 0.18
Nodes (6): AdvRouter, The trained advantage net plus its feature scaling; ``rom.router``., (best receiver j, its advantage A, {j: A} over all j != c)., _PlaneOracle, The whole space is on-manifold: projection is the identity, residual zero., test_train_router_returns_a_loadable_payload_for_a_toy_atlas()

### Community 11 - "PatchAtlasRuntime"
Cohesion: 0.06
Nodes (24): fixture, PatchAtlasRuntime, _ke_v(), ndarray, Decoded (n_bodies, 3, 4) pose at the current state., Live learned-boundary and support values for viewer/debug output., Runtime M is the learned SPD field, never a decoder Jacobian., Runtime V and force come from the learned scalar field. (+16 more)

### Community 13 - "selfcheck.py"
Cohesion: 0.08
Nodes (31): u_j = E_j(D_i(u_i)) -- the physical handoff between overlapping charts., transition(), actuator_maps.py — per-actuator generalized-force maps for a staged ROM. Every…, Structural validation only (no trained model needed): can every declared…, validate(), __getattr__(), patch_rom — package format, loader, and runtime for a built ``patch_atlas``.…, Keep field-training imports independent of optional runtime helpers. (+23 more)

### Community 14 - "SamplingOptions"
Cohesion: 0.04
Nodes (41): _observed_pose(), SDF zero-level circle S^1 in R^2 with physical pose observations. The smallest…, Physical pose emitted by the surface-response simulator., SO(3) = RP^3: one body on a spherical joint, the ideal-joint (analytic) path.…, _observed_pose(), SDF zero-level 3-sphere S^3 in R^4 with physical pose observations. The first…, Physical pose emitted by the surface-response simulator., _observed_pose() (+33 more)

### Community 15 - "run_patch_atlas.py"
Cohesion: 0.20
Nodes (11): cmd_selfcheck(), cmd_viewer(), active(), advance(), callback(), CspaceInset, _default_roms_dir(), main() (+3 more)

### Community 16 - "sampling.py"
Cohesion: 0.07
Nodes (41): _coordinates(), _explore(), _explore_legacy(), coordinate_at(), coordinate_near(), physical_domain(), response_validity(), _explored_sample_field() (+33 more)

### Community 17 - "load_spec"
Cohesion: 0.16
Nodes (19): load_spec(), Import `cases/<name>.py` and return its `CASE`., build_ideal_oracle(), ideal_oracle.py — the projection oracle for mechanisms that are ALGEBRAIC.…, Is this case's manifold algebraic, so the analytic oracle is valid?, supports(), test_case_routes_to_the_analytic_oracle(), _differential_pose() (+11 more)

### Community 18 - "PhysicsFitter"
Cohesion: 0.33
Nodes (4): make_physics_fitter(), PhysicsFitter, Boundary between learned chart maps and learned physical fields., Method-independent entry point for the existing field teachers.

### Community 19 - "geometry.py"
Cohesion: 0.11
Nodes (21): apply_transform(), _bevel_gear(), _cylinder(), _from_file(), _link(), _load_scene(), _orient(), geometry.py — declarative meshes -> vertices/faces -> physical inertia. One… (+13 more)

### Community 20 - "jansen_patch_rom/manifest.json"
Cohesion: 0.06
Nodes (31): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+23 more)

### Community 21 - "stephenson_link_sep9/manifest.json"
Cohesion: 0.06
Nodes (31): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+23 more)

### Community 22 - "stephenson_link_single_chart/manifest.json"
Cohesion: 0.06
Nodes (31): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+23 more)

### Community 23 - "._fit_inner"
Cohesion: 0.12
Nodes (28): _apply_monotone_territory(), _bootstrap_inertial_claims(), _bootstrap_monotone_state(), _connected_admissible_components(), _e_step(), elapsed_since(), _eval_all(), _inertial_frontier_weight() (+20 more)

### Community 25 - "88_maltese_cross/main.py"
Cohesion: 0.08
Nodes (22): AssetDir, on_update(), write_to_disk(), AssetDir, _as_VF(), build_driver(), build_driver_axle(), build_wheel() (+14 more)

### Community 26 - "AdvRouterTrack"
Cohesion: 0.22
Nodes (4): AdvRouterTrack, Viewer track around ``PatchAtlasRuntime``: the runtime does the routing, the…, pen_target(), Per-case pen constants come from the case module's ``PEN_TARGET`` hook (pen…

### Community 27 - "quotient.py"
Cohesion: 0.13
Nodes (17): build_quotient(), _Class, effective_samples(), _knn_pairs(), load(), local_equivalence_candidates(), members_of(), _orthonormal() (+9 more)

### Community 28 - "validity_from_case"
Cohesion: 0.22
Nodes (6): PhysicalValidity, Protocol, Closure validity for a case. ``include_grounded`` adds the joints to the fixed…, Optional mechanism validity hook. ``residual`` returns one non-negative squared…, validity_from_case(), runtime.py — PatchAtlasRuntime: passive + scripted-actuator stepping. v1 scope,…

### Community 29 - "generic_sphere_cae/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 30 - "generic_sphere/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 31 - "jansen_analytic_full_experimental/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 32 - "jansen_analytic_handoff_candidate/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 33 - "jansen_analytic_nonfold_coverage_experimental/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 34 - "jansen_analytic_nonfold_experimental/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 35 - "jansen_analytic_replay_experimental/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 36 - "sarrus_stack/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 37 - "scissor_lift_stack/manifest.json"
Cohesion: 0.07
Nodes (29): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+21 more)

### Community 38 - "ManifoldOracle"
Cohesion: 0.10
Nodes (13): CallbackDirectionalOracle, ManifoldOracle, observe(), paired_rows(), responses(), A sampler-facing oracle: restore, push in one direction, report outcome.…, Capture enough state to make subsequent responses comparable., Restore exactly the baseline captured by :meth:`snapshot`. (+5 more)

### Community 39 - "new_dataset.py"
Cohesion: 0.11
Nodes (26): load_samples(), audit(), build(), build_cover_edges(), build_from_walked_edges(), build_local_edges(), check_frame_consistency(), _degree_capped_select() (+18 more)

### Community 40 - "jansen_walker_trial10v3/manifest.json"
Cohesion: 0.07
Nodes (28): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+20 more)

### Community 41 - "scissor_lift/manifest.json"
Cohesion: 0.07
Nodes (28): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+20 more)

### Community 42 - "scissor_lift_stack_dense/manifest.json"
Cohesion: 0.07
Nodes (28): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+20 more)

### Community 43 - "twin_independent_pistons/manifest.json"
Cohesion: 0.07
Nodes (28): actuator_maps, artifact, body_order, build_report, case, control_schema, format, gates (+20 more)

### Community 44 - "ndarray"
Cohesion: 0.16
Nodes (11): DirectionalResponse, PhysicalResponse, PhysicalSystem, Any, ndarray, Protocol, Solver-neutral directional-response interface for physical chart sampling., Whether a settled response remains in this system's physical domain. The… (+3 more)

### Community 45 - "view_chart_exploration.py"
Cohesion: 0.10
Nodes (21): test_exploration_snapshot_index_accepts_an_explicit_report_name(), DisplayEmbedding, ExplorationView, _limits(), load_exploration_snapshot(), load_exploration_snapshot_index(), load_exploration_snapshots(), main() (+13 more)

### Community 46 - "_points"
Cohesion: 0.13
Nodes (17): _chart_of_sample(), _load_prune_snapshots(), _points(), ndarray, Optional case-supplied projection into 2/3 dimensions (not VISUAL_UVS)., Place every sample in a 2- or 3-D display space. Returns ``(points, label,…, Filter+renumber index lists onto a kept-sample subset (inv: old -> new / -1)., Restrict samples to ``keep`` and remap edges / chart membership indices. (+9 more)

### Community 47 - "local_support_radii"
Cohesion: 0.26
Nodes (12): off_node_redundancy(), in_local_support(), local_support_radii(), locally_expanded(), _neighbour_count(), ndarray, Data-derived local domains for probing learned latent charts. The domain is a…, Local rank grows with latent dimension instead of becoming directional. (+4 more)

### Community 48 - "learned.py"
Cohesion: 0.22
Nodes (7): _limit_crossing_pairs(), load_mass(), load_scalar(), Learned per-chart physical fields for the shipped patch-atlas ROM. The pose…, Bisect physical joint-limit witnesses into local valid/invalid pairs., Artifacts written before the fields carried normalization buffers lack them:…, _with_defaults()

### Community 49 - "build_mesh"
Cohesion: 0.18
Nodes (11): body_inertia(), build_bodies(), build_mesh(), build_static(), _compound(), ndarray, Concatenate sub-meshes into one body: a welded triangle of bars, a link with a…, MeshSpec -> (V (n,3) float64, F (m,3) int32). (+3 more)

### Community 50 - "test_generic_so3.py"
Cohesion: 0.33
Nodes (9): Principal rotation vector; the antipodal pair at |v| = pi is RP^3's seam., (N, 12) packed poses -> (N, 3) axis-angle vectors, for view_sample_set., _rotvec(), VIEW_EMBED(), generic_so3: the ideal-joint SO(3) = RP^3 fixture is a clean 3-manifold., _rot(), test_manifold_is_rank_three_and_exactly_rotations_everywhere(), test_rotvec_handles_the_theta_equals_pi_seam() (+1 more)

### Community 51 - "view_sample_set.py"
Cohesion: 0.10
Nodes (28): available_specs(), cases/ — one declarative spec file per mechanism, and nothing else.…, Every case that has a spec file, whether or not it has been built., Resolve an exact case name or an unambiguous prefix. CLI users commonly know…, resolve_case_name(), _build_oracle(), _oracle_test_poses(), ``n`` fresh on-manifold configs: perturb a random training sample, project. (+20 more)

### Community 52 - "main"
Cohesion: 0.14
Nodes (22): load_references(), ``.npz`` with a pose array under ``poses``/``pose``/``q``/``T``/ ``transforms``…, Identical across every track in one viewer session., Settings, _fmt(), _layout(), main(), advance() (+14 more)

### Community 53 - ".jacobian"
Cohesion: 0.11
Nodes (7): The rest assembly, projected — a spec whose rest pose is already assembled (the…, Keep algebraic responses within declared physical travel/workspace. These are…, The two world directions ACROSS a slider's rail at the current pose. The rail…, Signed carrier-relative rolling residual for one bevel-gear mesh., Vectorised :meth:`_bevel_row` over a stack ``Tb`` of packed poses, shape ``(M,…, Closed form — every residual is quadratic in the packed poses., The closest configuration on the manifold to `q`, by Gauss-Newton. Every…

### Community 54 - "PositionAtlasRuntime"
Cohesion: 0.06
Nodes (33): decode_flat(), decoder_jacobian(), EncoderHandoff, project_pose(), ProjectedHandoff, ndarray, handoff.py -- decoder-space projection and handoff initialisation (shipped…, ``d decode / du`` as a (D, d) matrix. (+25 more)

### Community 55 - "PatchAtlasROM"
Cohesion: 0.09
Nodes (27): PatchAtlasROM, ndarray, Rank already competent receivers with the persisted physical RBF. The selector…, Decoded packed pose (n_bodies, 3, 4) at ``(chart, u)``., Signed learned membership score; zero is the classifier boundary., A loaded, runnable patch atlas: models + patches + topology + case., _sha256(), attach() (+19 more)

### Community 56 - "_module"
Cohesion: 0.15
Nodes (17): _module(), decode(), encode(), encode_raw(), features(), forward(), __init__(), _load_from_state_dict() (+9 more)

### Community 57 - "AtlasCaseSpec"
Cohesion: 0.08
Nodes (5): AtlasCaseSpec, _jsonable(), The canonical, ORDERED control schema: one entry per declared actuator, in…, Recursively make a spec value JSON-serializable (arrays -> lists)., Everything the pipeline needs to know about one mechanism.

### Community 58 - "stephenson_link.py"
Cohesion: 0.12
Nodes (18): _arc_angle(), _arc_bars(), _arc_centre(), _bar(), _boss(), _eccentric_phase(), _norm(), _pivot() (+10 more)

### Community 59 - "chart_contract.py"
Cohesion: 0.25
Nodes (7): ChartFitter, evidence_from_sample_set(), FittedChartAtlas, Any, Protocol, Stable, evidence-only contracts for learned physical chart fitters., Build the contract without forwarding training-only metadata.

### Community 60 - "ProjectionLedger"
Cohesion: 0.29
Nodes (4): ProjectionLedger, Path, Bounded response-reuse cache with an optional append-only JSONL audit., Cache a minimal response and optionally emit its complete audit row.

### Community 61 - "lazy_tongs_pantograph.py"
Cohesion: 0.15
Nodes (18): _bar(), _body_transform(), _boss(), _crank_pin_rest(), _forward(), _gear(), _gusset(), _pt() (+10 more)

### Community 62 - "fit_support_and_boundary"
Cohesion: 0.20
Nodes (9): _cloud_samples(), _fit_margin_sdf(), fit_support_and_boundary(), _lbfgs(), _neutral_boundary_sdf(), A learned field for a sampled manifold with no boundary witnesses. A chart's…, Regress a latent SDF onto the decoded poses' normalized physical margin:…, Fit chart-local support and a physical-boundary SDF. The support classifier… (+1 more)

### Community 63 - "diag_utils.py"
Cohesion: 0.13
Nodes (28): AssetDir, build_edge_voronoi(), build_pt_voronoi(), compute_newton_dir(), create_ddf(), flag_label(), poll_keyboard(), Shared utilities for the 40_distance_diagnoser examples. (+20 more)

### Community 64 - "_ToyAE"
Cohesion: 0.25
Nodes (4): _Linear, Linear chart autoencoder on R^6 with a 2-d latent (encode/decode are exact…, test_batched_lm_projection_recovers_an_exactly_representable_pose(), _ToyAE

### Community 65 - "encode"
Cohesion: 0.18
Nodes (15): bwd(), chart_loss(), _chart_state(), _decode(), encode(), _eval_all_composite(), fwd(), _gate_angles_folds() (+7 more)

### Community 66 - "fit_dynamics"
Cohesion: 0.32
Nodes (5): fit_dynamics(), pot_loss(), pot_terms(), Distill Jacobian-derived M and gravitational V into learned fields. Targets are…, ScalarField

### Community 67 - "scissor_lift.py"
Cohesion: 0.12
Nodes (11): _annulus_sdf(), _observed_pose(), Production annulus test case: a free puck around a fixed central peg. The…, The generic test system: valid exactly when this SDF is non-positive., Simulator observation for the locked carriage/puck scene, not a chart map., _wall(), _anchor(), A single-pair, roller-ended scissor lift. (+3 more)

### Community 68 - "build_oracle"
Cohesion: 0.14
Nodes (6): build_oracle(), _anim(), oracle.py — build the full-space libuipc simulator for ANY declared mechanism.…, Return the tangent basis used by the explorer for a declared mechanism. UIPC's…, MechanismSpec -> a ready `LibuipcConfigOracle` (world already initialized).…, tangent_frame_for_spec()

### Community 69 - "IdealJointSim"
Cohesion: 0.18
Nodes (9): IdealJointSim, _perpendicular(), ndarray, ideal_sim.py — full-space rigid-body dynamics on the analytic ideal-joint…, outer(f, [p; 1]) into body i's block, p = its local image of the point., Residual rows (travel - stop) and their Jacobian, by central difference over…, argmin |q - q_pred|_W s.t. C(q) = 0 (and pinned stops), by Gauss-Newton with…, Mbar = int rho [p;1][p;1]^T over the body (4x4, SPD). (+1 more)

### Community 70 - "SDFDirectionalOracle"
Cohesion: 0.21
Nodes (5): Retract onto ``phi == 0`` using only local SDF evaluations. Open surfaces have…, Trace a continuous zero-set branch using local trust-region retractions. A…, Segment-traced responses over a region defined solely by ``phi <= 0``., Measure a zero-level-surface tangent from local SDF probes. Random paired…, SDFDirectionalOracle

### Community 71 - "mobius_kaleidocycle.py"
Cohesion: 0.33
Nodes (5): _bar_mesh(), cases/mobius_kaleidocycle.py — a closed ring of M tetrahedra, hinged edge to…, A cylindrical edge p0->p1; unlike `MeshSpec("link", ...)` this survives leaving…, Wireframe tetrahedron: a compound of its 6 edges., _tetra_mesh()

### Community 72 - "rom_target.py"
Cohesion: 0.31
Nodes (8): main(), Interactive "click a target, the ROM's real dynamics drive the pen there" --…, _ray_plane_hit(), run(), callback(), post(), seg(), trail()

### Community 73 - "ops.py"
Cohesion: 0.14
Nodes (15): actuator_jacobian(), joint_angle_torch(), mass_matrix(), potential_and_grad(), ndarray, ops.py — reduced mass/potential/actuator operators for a patch-atlas chart.…, d(theta_joint)/du, for the generalized force F = J^T tau of a joint-torque…, Signed joint angle from a torch (..., n_bodies, 3, 4) transform stack.… (+7 more)

### Community 75 - "dataset.py"
Cohesion: 0.21
Nodes (14): _check(), exists(), load_atlas_data(), load_dynamics_data(), membership_pairs(), path_in(), dataset.py — the standardized intermediate dataset every case flows through.…, The chart decomposition of the sample cloud. chart_ids (n_charts,) int chart… (+6 more)

### Community 76 - "SDFPhysicalOracle"
Cohesion: 0.21
Nodes (3): Physical-oracle facade for SDF tracing used by the shared strategy., Keep the observation-to-state differential private to the backend., SDFPhysicalOracle

### Community 77 - "_restrict_to_components"
Cohesion: 0.50
Nodes (4): _largest_component(), ``mask`` restricted to its largest connected component in the kNN witness graph…, Apply :func:`_largest_component` to every chart column of a bool matrix., _restrict_to_components()

### Community 78 - "explore"
Cohesion: 0.16
Nodes (18): AdvancingFrontOptions, explore(), enqueue(), infer(), _maximin_direction(), ndarray, Path, _rank_by_gap() (+10 more)

### Community 79 - "SampleSetView"
Cohesion: 0.23
Nodes (8): _chart_pass(), Positions for learned overlays; samples remain in data space. When a chart is…, Stable chart id for a column position., Column positions ordered by patch (support) size, largest first., Column POSITIONS to draw. ``isolate`` is a SIZE RANK: 0 = biggest chart, 1 =…, SampleSetView, on_check(), on_slide()

### Community 80 - "physical_chart_autoencoder.py"
Cohesion: 0.13
Nodes (19): condition_report(), gauge_fixed(), physical_twists(), Reference-relative physical-pose metric and its Jacobian/gauge diagnostics. Any…, Principal ``SO(3) -> so(3)``, vectorised, valid for ``||omega|| < pi``. Same…, ``(..., n_bodies, 3, 4)`` torch transforms -> ``(..., 6*n_bodies)`` twists.…, ``d(physical_twists)/du``, shape ``(n, 6*n_bodies, d)``. Exact forward-mode…, Whiten the latent's affine gauge by the mean pullback metric ``G``. ``G =… (+11 more)

### Community 81 - "_m_step"
Cohesion: 0.14
Nodes (22): batched_tangent(), _coverage_margin_loss(), dense_chart_forward(), _double_cover_mask(), ensemble_decode(), ensemble_decoder_spectral(), ensemble_encode(), _eval_all_lean() (+14 more)

### Community 83 - "oracle_eval.py"
Cohesion: 0.25
Nodes (5): oracle_eval.py -- fresh on-manifold evaluation poses from the physical oracle.…, Max revolute-joint closure gap ``||T_a p_a - T_b p_b||`` per pose. Joint pivots…, _worst_joint_gap(), Build a denser rotate dataset from the oracle (exact poses, exact tangent…, score()

### Community 84 - "sample_graph.py"
Cohesion: 0.13
Nodes (15): adjacency(), certified_edges(), hop_ball(), largest_component(), load(), neighbour_lists(), sample_graph.py — the certified neighbourhood graph of a sampled manifold. The…, Deduplicated, self-loop-free, sorted (i<j) edge list. (+7 more)

### Community 85 - "LibuipcConfigOracle"
Cohesion: 0.19
Nodes (5): LibuipcConfigOracle, libuipc-backed projection oracle retained by the physical-AE pipeline., Project a packed configuration by drive, settle, then joint-only relax., Capture simulator-owned transforms and drive state for probe replay., Put the UIPC scene back before every directional command. This deliberately…

### Community 86 - "loader.py"
Cohesion: 0.12
Nodes (7): build_topology(), LearnedSupportTopology, Route only through learned support fields and physical AE transfers. Unlike the…, Topology for the advantage router: no support fields, no routing here. The…, Create runtime topology. With support fields: the legacy learned-support…, RouterTopology, loader.py — rebuild a runnable ``PatchAtlasROM`` from a staged directory.…

### Community 87 - "jansen.py"
Cohesion: 0.33
Nodes (6): _body(), _pose_transform(), cases/jansen.py — the crank-coupled Jansen leg, one PERIODIC coordinate. The…, LOCAL[body][letter] transformed by POSE0[body] -- the world-frame pivot a…, A body is the union of its bars, each a thin box between two joints., _world_point()

### Community 88 - "MassField"
Cohesion: 0.38
Nodes (3): MassField, _mlp(), SPD matrix field, parameterized by a Cholesky factor.

### Community 89 - "FullSpaceSim"
Cohesion: 0.48
Nodes (3): FullSpaceSim, ndarray, fullspace_sim.py — production full-space libuipc simulator, control-schema…

### Community 90 - "PhysicalRBFSelector"
Cohesion: 0.33
Nodes (3): PhysicalRBFSelector, Serialized RBF receiving-chart ranker with a small physical correction., test_physical_rbf_selector_ranks_without_training_memberships()

### Community 91 - "CoordinateMap"
Cohesion: 0.21
Nodes (6): CoordinateMap, Evaluate a case's declared coordinates on recorded configurations. Built once…, Capture cloud-dependent state (angle references, the isomap embedding)., Nonlinear intrinsic coordinates of the config cloud (geodesic MDS). Used when…, (N, 12*n) -> (N, n, 3, 4)., unpack()

### Community 92 - "adv_labels.py"
Cohesion: 0.14
Nodes (20): jac_b(), label_states(), noise_floor(), proj_batched(), proj_source(), projection_errors(), Error-ratio ("advantage") labels and training for the chart router…, (a) anchors / jitter / interpolation inside each chart's own territory. (+12 more)

### Community 93 - "test_obstacle_constrained_2r.py"
Cohesion: 0.20
Nodes (11): characterize_collision_complement(), _periodic_components(), ndarray, Evaluator-only forward kinematics for the physical tool centre., 4-connected component sizes on the evaluator's periodic angle grid., Dense evaluator-only characterization of this physical design. This…, tool_position(), Evaluator-only topology smoke tests for the obstacle-constrained 2R arm. (+3 more)

### Community 94 - "sarrus_stack.py"
Cohesion: 0.21
Nodes (11): _bar_mesh(), _elbow(), _plate_mesh(), cases/sarrus_stack.py — two stacked Sarrus linkages, all-revolute, no contact.…, Display-only 2-D coordinate for ``view_sample_set``: (stage-1 plate travel…, A cylindrical link p0->p1; unlike `MeshSpec("link", ...)` this does not flatten…, A plate: a compound bar joining its two chain anchor points (plus a stub…, Elbow point of a symmetric 2-link chain rooted at `anchor`, bulging along… (+3 more)

### Community 95 - "pen_pd_controls"
Cohesion: 0.33
Nodes (4): pen_pd_controls(), pen(), Operational-space (task-space) computed torque on the PEN, using only the ROM's…, with_wind()

### Community 96 - "stewart_platform.py"
Cohesion: 0.40
Nodes (4): _file_body(), cases/stewart_platform.py — a rotary-actuated Stewart-Gough platform. Ported…, Display-only: place each sample by the moving plate's translation. The 6-D…, VIEW_EMBED()

### Community 97 - "CylindricalReducedOracle"
Cohesion: 0.30
Nodes (3): CylindricalReducedOracle, Integrate gravity in Y while clamping the other four body modes. IPC remains…, Hard two-coordinate cylindrical kinematics around a contact oracle. The wrapped…

### Community 98 - "validity.py"
Cohesion: 0.40
Nodes (4): validity_loss_raw(), _json_safe(), validity.py -- the chart contract shared by fitting, packaging and runtime.…, _validity_residual()

### Community 99 - "registry.py"
Cohesion: 0.67
Nodes (3): Central registry for chart fitter implementations., register(), resolve()

### Community 100 - "Competitive CAE Mechanism Test Suite"
Cohesion: 0.17
Nodes (11): Case requirements, Case-specific correctness conditions, Competitive CAE Mechanism Test Suite, Feature coverage matrix, Four-dimensional limit, Implementation sequence, Information boundary, Primary roster (+3 more)

### Community 102 - "joint_travel"
Cohesion: 0.27
Nodes (8): _axis_angle(), joint_travel(), Physical travel and workspace limits, independent of chart coordinates., Physical displacement of a revolute or sliding joint from its rest pose., Return physical-bound diagnostics for a pose, else ``None``. Only mechanism…, Normalized signed distance to the nearest declared physical bound, ``> 0``…, signed_margin(), violation()

### Community 103 - "Frame 1: cube tumbling forward on near-edge, boards shifting right"
Cohesion: 0.38
Nodes (7): Frame 0: cube resting on left board, tilted onto a corner, Frame 1: cube tumbling forward on near-edge, boards shifting right, Frame 2: cube upright at far end of upper board, near its edge, Frame 3: cube tipped off board edge onto lower board, boards moved further, Frictional contact & edge tipping between cube and moving supports, Two kinematically driven plate boards translating rightward under the cube, Walking/tumbling rigid cube driven across translating boards via frictional contact

### Community 104 - "test_support_sdf_derives_its_witness_radius_from_the_anchor_cloud"
Cohesion: 0.20
Nodes (9): fit_constraint_radius_sdf(), fit_local_support_sdf(), _fit_witness_sdf(), Fit a signed field from physical inside/outside witnesses. The labels establish…, Fit an SDF from local decoded-constraint witnesses only. Every supplied anchor…, Fit a conservative learned support SDF around locally valid samples. This is…, The runtime-support builder must not require a global radius argument., test_support_sdf_derives_its_witness_radius_from_the_anchor_cloud() (+1 more)

### Community 105 - "joint_travel"
Cohesion: 0.16
Nodes (12): inside_domains(), _axis_angle(), joint_angle_torch(), joint_travel(), outside_joint_limits(), coordinates.py — read the declared intrinsic coordinates off a configuration.…, Signed joint angle from a torch (..., n_bodies, 3, 4) transform stack.…, Signed rotation angle (rad) of R about `axis`, relative to R0. The full 3x3… (+4 more)

### Community 106 - "_stacked_ensemble"
Cohesion: 0.25
Nodes (9): One jvp(ensemble_decode, .) per latent axis == d jvps per chart. Mirrors the…, Grouped [K,M] masked reduction == sum of per-(pair,direction) means. The…, _stacked_ensemble(), ensemble_decode(), ensemble_encode(), test_batched_inverse_cycle_matches_per_chart_values_and_gradients(), test_batched_seam_replay_matches_the_scalar_pair_loop(), test_batched_tangent_jvp_matches_the_per_chart_loop() (+1 more)

### Community 107 - "load_sdf_system"
Cohesion: 0.39
Nodes (7): load_sdf_system(), Return an optional first-class SDF system without reading ``CASE``., Smoke tests for the additional topology fixtures' physical SDFs., test_handle_surfaces_have_surface_seeds_and_bounded_exteriors(), test_new_topology_cases_load_and_accept_their_seeds(), test_planar_topology_fixtures_have_the_intended_holes_and_cut(), test_punctured_torus_masks_the_removed_surface_disk()

### Community 108 - "enforce_support_coverage"
Cohesion: 0.67
Nodes (3): enforce_support_coverage(), margins(), Jointly enforce the learned-domain contract on valid sampler samples. Every…

### Community 116 - "MechanismSpec"
Cohesion: 0.08
Nodes (9): MechanismSpec, ndarray, Case-declared translational workspace boundary for a driven body., Bodies + joints + world conventions. Nothing about charts or training., Config-packing order: exactly the bodies the decoder poses., Index into the driven/config packing., Lower to `atlas_rom.spec.solver_spec.MechanismSpec` (baked vertices). The…, WorkspaceBound (+1 more)

### Community 121 - "IdealJointOracle"
Cohesion: 0.17
Nodes (7): IdealJointOracle, Bodies whose block of `tangent` is (numerically) zero. A stepping direction…, Exact analytic tangent — the override the FD paired probe would do.…, (basis (ambient, dim), singular values, dim) — exact, by SVD. `eps` and…, Cross-check every rank against the declared dimension. `MIN_GAP_RATIO` is a…, Rank from the FIRST large ratio gap, scanning down from the top. The genuine…, Analytic `project()` / `tangent_frame()` over packed (n_bodies, 3, 4) poses.…

### Community 122 - "Rack-and-pinion meshing via IPC contact (rotating gear driving a linear toothed rack)"
Cohesion: 0.70
Nodes (5): Gear-on-rack frame 0: pinion meshed with toothed rack, wide oblique view, Gear-on-rack frame 1: pinion advanced along rack, closer camera, Gear-on-rack frame 2: pinion rotated further, rack teeth engaged, Gear-on-rack frame 3: pinion near-top view, rack meshing close-up, Rack-and-pinion meshing via IPC contact (rotating gear driving a linear toothed rack)

### Community 123 - "Rain"
Cohesion: 0.29
Nodes (3): Rain, Closest boundary point of the CCW convex polygon to x: (point, outward normal,…, Falling balls (2D discs in the mechanism's xy plane) that hit EVERY link. Each…

### Community 127 - "AssetDir"
Cohesion: 0.14
Nodes (6): AssetDir, flatten_json(), flatten(), :param pattern: the pattern to search for :return: a list of objects that match…, SceneEdit, unflatten_json()

### Community 130 - "VIEW_EMBED"
Cohesion: 0.67
Nodes (3): Viewer-only 3-D embedding: the moving B-link origin in physical space. The…, VIEW_EMBED(), test_view_embed_returns_the_moving_b_link_origin()

### Community 131 - "State accessor: ABD transforms/velocities & FEM positions"
Cohesion: 0.50
Nodes (4): State accessor: ABD transforms/velocities & FEM positions, Polyscope state accessor UI with ABD/FEM cubes, Rigid-body contact, chain constraints & impact scattering, Wrecking ball on chain smashing cube wall

### Community 144 - "jansen_walker.py"
Cohesion: 0.40
Nodes (4): Eight independently pinned Jansen legs locked to a shared, phased shaft., Display the shared crank phase on its canonical unit circle. This is a viewer-…, VIEW_EMBED(), test_view_embed_places_the_master_crank_on_a_unit_circle()

### Community 147 - "solver_spec.py"
Cohesion: 0.09
Nodes (13): Enum, BodySpec, ConfigurationType, ConstraintObservation, ContactModel, LocalStratum, MechanismSpec, Small, solver-independent input and local-stratum records. This is deliberately… (+5 more)

### Community 153 - "test_competitive_cae.py"
Cohesion: 0.05
Nodes (39): _chart_quality_passes(), _condition_gauge_covariance(), _decoder_spectral_product(), _fps_warm_start_indices(), Decoder Lipschitz bound from its affine layers' spectral norms. Each spectral…, Floor learned gauge covariance eigenvalues to bound triangular solves., Use FPS separation, not ``n / K``, to define an initial chart patch., Persistent per-chart territory only ever grows, and admission comes from graph-… (+31 more)

### Community 164 - "AssetDir"
Cohesion: 0.15
Nodes (5): ContactSystemFeature, AssetDir, ContactInfo, Display the contact information for this contact primitive. Gradient and…, Geometry

### Community 208 - "84_abd_revolute_joint_local/main.py"
Cohesion: 0.15
Nodes (16): change_right_tf(), check_attachments_equal(), make_abd_cube(), on_update(), print_angles(), process_surface(), AngleAxis, ndarray (+8 more)

### Community 209 - "85_abd_prismatic_joint_local/main.py"
Cohesion: 0.15
Nodes (16): change_right_tf(), check_axes_collinear(), make_abd_cube(), on_update(), print_distances(), process_surface(), AngleAxis, ndarray (+8 more)

### Community 237 - "MyEngine"
Cohesion: 0.10
Nodes (8): array, dtype, AssetDir, increment_positions(), MyEngine, kernel, PyIEngine, vec3d

### Community 238 - "87_scissor_lift_joint/main.py"
Cohesion: 0.08
Nodes (19): AssetDir, create_cloth(), AssetDir, check_prismatic_axes_collinear(), check_revolute_attachments(), make_abd_body(), make_io(), on_update() (+11 more)

### Community 253 - "86_panda_hydro_traj_cubes/main.py"
Cohesion: 0.17
Nodes (15): _clamped_frame(), make_abd_cube(), on_update(), print_state(), prismatic_anim(), process_surface(), ndarray, SimplicialComplex (+7 more)

### Community 277 - "AssetDir"
Cohesion: 0.14
Nodes (6): AssetDir, make_joint_animator(), process_surface(), SimplicialComplex, Four-box revolute joint chain along the X axis. Four cubes are placed at x =…, Create an animator callback for a specific joint.

### Community 314 - "AssetDir"
Cohesion: 0.15
Nodes (4): AffineBodyStateAccessorFeature, AssetDir, StateAccessorInspector, FiniteElementStateAccessorFeature

### Community 331 - "graphify"
Cohesion: 0.15
Nodes (12): Deliverables and visualization, Execution environment, Explicit coding permission and action preview, General pipeline: no per-case shortcuts, graphify, Physical chart atlas: methods before gates, Physical chart atlas: strict no-priors / no-cheating policy, Roles: analyst vs experiment runner (+4 more)

### Community 350 - "AssetDir"
Cohesion: 0.20
Nodes (3): AssetDir, UpdateInfo, screw_animation()

### Community 355 - "AssetDir"
Cohesion: 0.20
Nodes (3): AssetDir, animate_cube(), UpdateInfo

### Community 361 - "AssetDir"
Cohesion: 0.20
Nodes (3): AssetDir, arm_animation(), UpdateInfo

### Community 370 - "AssetDir"
Cohesion: 0.24
Nodes (3): AssetDir, async_run(), on_update()

### Community 376 - "AssetDir"
Cohesion: 0.20
Nodes (3): AssetDir, animate_cube(), UpdateInfo

### Community 377 - "AssetDir"
Cohesion: 0.20
Nodes (4): AssetDir, UpdateInfo, update_articulation(), update_ref_dof_prev()

### Community 380 - "AssetDir"
Cohesion: 0.18
Nodes (3): AssetDir, animate(), UpdateInfo

### Community 381 - "AssetDir"
Cohesion: 0.15
Nodes (6): AssetDir, animate_joint(), process_surface(), SimplicialComplex, UpdateInfo, Driving prismatic joint + external force. Two phases controlled by the Animator…

### Community 383 - "AssetDir"
Cohesion: 0.15
Nodes (6): AssetDir, animate_joint(), process_surface(), SimplicialComplex, UpdateInfo, Driving revolute joint + external torque. Two phases controlled by the Animator…

### Community 385 - "AssetDir"
Cohesion: 0.20
Nodes (3): AssetDir, animate_tet(), UpdateInfo

### Community 388 - "AssetDir"
Cohesion: 0.18
Nodes (4): AssetDir, animation(), IO, UpdateInfo

### Community 389 - "AssetDir"
Cohesion: 0.16
Nodes (6): AssetDir, cube_animation(), ground_animation(), process_surface(), SimplicialComplex, UpdateInfo

### Community 391 - "AssetDir"
Cohesion: 0.18
Nodes (5): AssetDir, build_mesh(), process_surface(), SimplicialComplex, Object

### Community 398 - "AssetDir"
Cohesion: 0.20
Nodes (3): AssetDir, arm_animation(), UpdateInfo

### Community 411 - "AssetDir"
Cohesion: 0.20
Nodes (3): AssetDir, animate_cube(), UpdateInfo

### Community 417 - "AssetDir"
Cohesion: 0.20
Nodes (3): AssetDir, gear_animation(), UpdateInfo

### Community 445 - "Q: Fixed-support micro-atlas pipeline ran end-to-end and halted with 18 of 94 uncertified patches. Decide whether to investigate large patches, widen the network, relax LIMITS, or use another repair before continuing."
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Fixed-support micro-atlas pipeline ran end-to-end and halted with 18 of 94 uncertified patches. Decide whether to investigate large patches, widen the network, relax LIMITS, or use another repair before continuing., Source Nodes

### Community 446 - "Q: summarize the current direction; it feels like we may be overcomplicating things"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: summarize the current direction; it feels like we may be overcomplicating things, Source Nodes

### Community 447 - "Q: for sdf boundary cases: make the clamp project along phi_M^-1 grad phi_s, so all three boundary paths agree on the metric. C we should do this for correctness right"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: for sdf boundary cases: make the clamp project along phi_M^-1 grad phi_s, so all three boundary paths agree on the metric. C we should do this for correctness right, Source Nodes

### Community 448 - "Q: read both docs in /mapper, thoughts?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: read both docs in /mapper, thoughts?, Source Nodes

### Community 449 - "Q: any new ideas or ideas from literature we missed?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: any new ideas or ideas from literature we missed?, Source Nodes

### Community 450 - "Q: read B, what do you think"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: read B, what do you think, Source Nodes

### Community 451 - "Q: read C, lets see whats missing still"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: read C, lets see whats missing still, Source Nodes

### Community 452 - "Q: The ROM jumps because its charts are not yet valid local coordinate charts everywhere they claim support. At a bad 1 to 0 transition, is this an encoder collision caused by weak overlap replay? Thoughts for latest Jansen ROM with high transition deltas?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: The ROM jumps because its charts are not yet valid local coordinate charts everywhere they claim support. At a bad 1 to 0 transition, is this an encoder collision caused by weak overlap replay? Thoughts for latest Jansen ROM with high transition deltas?, Source Nodes

### Community 453 - "Q: What should change in the training process to eliminate the Jansen chart encoder collision while preserving the existing two-chart split?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: What should change in the training process to eliminate the Jansen chart encoder collision while preserving the existing two-chart split?, Source Nodes

### Community 454 - "Q: How can we construct the entire reachable directed handoff collar generically, without case-specific knowledge?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: How can we construct the entire reachable directed handoff collar generically, without case-specific knowledge?, Source Nodes

### Community 455 - "Q: Should support be derived from map reconstruction or transition success?"
Cohesion: 0.40
Nodes (4): Answer, Outcome, Q: Should support be derived from map reconstruction or transition success?, Source Nodes

### Community 456 - "Install (uv recommended)"
Cohesion: 0.40
Nodes (4): Install (uv recommended), libuipc-samples, pyuipc from PyPI, pyuipc from source

### Community 467 - "State Accessor Feature"
Cohesion: 0.50
Nodes (3): State Accessor Feature, Summary, Usage

### Community 468 - "External Articulation Constraint Example"
Cohesion: 0.50
Nodes (3): Core Snippets, External Articulation Constraint Example, Overview

## Knowledge Gaps
- **537 isolated node(s):** `format`, `case`, `label`, `n_bodies`, `body_order` (+532 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1676 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **81 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Work-memory lessons

**Preferred sources** — corroborated by past sessions; start here.
- `certification` (3× useful, score=2.439653736) _(code changed — re-verify)_
- `chart_autoencoder.py` (2× useful, score=1.32193629) _(code changed — re-verify)_

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `load_spec()` connect `load_spec` to `view_pantograph_drag.py`, `CompetitiveCAEFitter`, `generic_manifold_diagnostics.py`, `competitive_cae.py`, `load_sdf_system`, `selfcheck.py`, `_points`, `run_patch_atlas.py`, `test_generic_so3.py`, `view_sample_set.py`, `._fit_inner`, `oracle_eval.py`, `loader.py`, `PatchAtlasROM`, `test_obstacle_constrained_2r.py`?**
  _High betweenness centrality (0.076) - this node is a cross-community bridge._
- **Why does `write_to_disk()` connect `88_maltese_cross/main.py` to `87_scissor_lift_joint/main.py`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `make_abd_body()` connect `87_scissor_lift_joint/main.py` to `AssetDir`?**
  _High betweenness centrality (0.055) - this node is a cross-community bridge._
- **Are the 3 inferred relationships involving `MechanismSpec` (e.g. with `BodySpec` and `JointSpec`) actually correct?**
  _`MechanismSpec` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `format`, `case`, `label` to the rest of the system?**
  _537 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Expectations` be split into smaller, more focused modules?**
  _Cohesion score 0.05738615327656423 - nodes in this community are weakly interconnected._
- **Should `view_pantograph_drag.py` be split into smaller, more focused modules?**
  _Cohesion score 0.061018437225636525 - nodes in this community are weakly interconnected._