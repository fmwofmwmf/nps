# Competitive CAE Mechanism Test Suite

Production atlas construction uses the `competitive_cae` fitter exclusively.
The former `physical_ae_patch` fitter is deprecated and rejected by the
`patch_atlas` entry point.

## Purpose

This suite tests whether the competitive CAE pipeline can discover and maintain
a physically valid chart atlas for attractive, recognizable mechanisms with
meaningfully different configuration spaces. It is a topology-first suite, but
not a collection of synthetic topology demonstrations: each primary case must
be a plausible real mechanism with useful visible motion.

The supported target class is a connected, fixed-dimensional configuration
manifold, possibly with boundary and corners. Smooth manifolds are the normal
case. Contact may introduce boundary strata and piecewise-smooth transitions,
provided the selected test component retains a fixed intrinsic dimension.

## Information boundary

The learning pipeline may receive only simulator-produced frames and permitted
physical evidence. It must not receive declared generalized coordinates,
intrinsic dimension, topology, tangent frames, chart count, chart memberships,
routing labels, or training-data membership lookups.

Mechanism parameters may contain the information required to construct and run
the physical system. Oracle coordinates, topology, contact modes, and dense
reference samples may be retained by the evaluation harness, but must remain
isolated from fitting, chart selection, transitions, and runtime routing.

A passing diagnostic does not validate a run that crossed this information
boundary.

## Primary roster

The order below forms the intended topology and behavior progression. The
five-bar high-genus case deliberately follows the open bevel differential: it is
a special topology regression, not the template for higher-dimensional cases.

| Order | Intrinsic dimension | Mechanism | Target configuration space | Model | Principal challenge |
|---:|---:|---|---|---|---|
| 1 | 1 | Coupled scissor lift with physical stops | Interval, `I` | Analytic with limits | Many-body closure and two physical boundaries |
| 2 | 1 | Crank-coupled Jansen leg | Circle, `S^1` | Analytic | Periodicity and strongly nonlinear visible motion |
| 3 | 2 | Stephenson reversing gear | Cylinder, `S^1 x I` | Analytic with limits | One periodic and one bounded direction |
| 4 | 2 | Open bevel differential | Torus, `T^2` | Analytic gear constraint | Coupled periodic directions and repeated gear states |
| 5 | 2 | Five-bar topology regression | Genus-four surface, `Sigma_4` | Analytic | High-genus closed surface and necessary chart transitions |
| 6 | 2 | Obstacle-constrained full-rotation 2R arm | Torus minus certified collision disks | Contact | Collision-carved boundary, holes, grazing, and bottlenecks |
| 7 | 2 | Spherical pendulum | Sphere, `S^2` | Analytic distance constraint | Closed curved surface with no global Euclidean chart |
| 8 | 3 | Ball-and-socket orientation stage | `SO(3)`, equivalent to `RP^3` | Analytic spherical joint | Nontrivial closed 3-manifold and orientation seams |
| 9 | 3 | Locked 3-SPS platform | Certified connected regular 3-manifold | Analytic closed chain | Native spatial motion and non-product geometry |
| 10 | 3 | Delta robot on one conservative branch | Cube with corners, `I^3` | Analytic with limits | Bounded 3D closed-chain motion and conditioning |
| 11 | 4 | Compact continuous-rotation SCARA | `T^3 x I` | Analytic with limits | Tiny real 4-DOF scaling test |

The suite intentionally does not include Geneva, Sarrus, dedicated
near-self-approach fixtures, generic boxes, or product mechanisms constructed
only to manufacture topology. Generic six- and seven-bar polygon spaces are
also excluded from the primary roster. The five-bar remains because `Sigma_4`
is unusually informative and already relevant to the pipeline.

The current scissor implementation is a coupled eight-stage, one-DOF stack.
Its existing numerical sampling bounds are not by themselves physical
boundaries. The qualification fixture must enforce joint or rail stops at
regular configurations strictly before the collapsed and vertical
singularities. The resulting physical configuration space is then a compact
interval.

The differential should use a symmetric open bevel design. Carrier angle and
relative axle angle provide two periodic evaluator coordinates; side-gear and
planet orientations follow from the gear constraints. Exact tooth counts and
marked-body symmetries must be checked for finite identifications and connected
components before fixing the `T^2` assertion.

The five-bar row refers to the existing geometry whose full linkage
configuration space is the genus-four surface, not merely its end-effector
workspace or motor-angle image.

## Feature coverage matrix

`P` denotes a primary feature of a case and `V` denotes a required controlled
variant. Empty entries are not required coverage.

| Feature | Scissor | Jansen | Stephenson | Differential | Five-bar | Contact 2R | Pendulum | Ball joint | 3-SPS | Delta | SCARA |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Periodic direction |  | P | P | P | P | P |  | P |  |  | P |
| Physical boundary | P |  | P |  |  | P |  |  |  | P | P |
| Boundary corner |  |  | V |  |  | V |  |  |  | P | V |
| Nontrivial topology |  | P | P | P | P | P | P | P | V |  | P |
| Multiple necessary charts |  | P | P | P | P | P | P | P | V | V | P |
| Strong curvature | V | P | V |  | V | V | P | P | P | P | V |
| Poor conditioning | V | V | V | V | V | V |  | V | V | P | V |
| Contact boundary |  |  |  | V |  | P |  |  |  | V |  |
| Grazing contact |  |  |  |  |  | P |  |  |  | V |  |
| Narrow bottleneck |  | V | V |  | V | P |  |  | V | V |  |
| Repeated pose motif | V | P | V | P | V | V |  | V | V |  | V |
| Observation redundancy | P | P | P | P | P | P | V | P | P | P | P |

Feature variants should normally modify physical geometry, limits, contacts,
or marker placement rather than introduce a separate mechanism. Each important
feature must appear in at least two cases so that a pass cannot depend on one
particular simulator or linkage family.

## Case-specific correctness conditions

The obstacle-constrained 2R arm begins with two full-turn revolute joints and
therefore an unconstrained `T^2` joint space. A small obstacle will generally
have more than one inverse-kinematic collision preimage. The fixture must not
assume a once-punctured torus: dense evaluator sampling must establish the
number of disk-like forbidden components, regularity of their boundaries, and
connectivity of the complement before its topology is frozen.

The spherical pendulum represents a fixed-length direction by physical point
positions. Axial spin must not be inserted as a hidden rigid-body coordinate,
and an arbitrary fixed rotation must not be manufactured to force the desired
answer. If the runtime representation requires full rigid transforms, use a
physical orientation constraint or a particle-native observation path.

The ball-and-socket case is one asymmetric rigid platform whose centre is
pinned by an ideal spherical joint. This full-body configuration space is
exactly `SO(3)`. A serial three-revolute wrist is not a substitute: with all
link frames observed its joint configuration space is `T^3`, even though its
end-effector orientation maps into `SO(3)`.

Use the stock, regular locked 3-SPS fixture for qualification. The retuned
singular-node fixture is an assumption-violation diagnostic because its local
dimension jumps at the node.

The Delta case must use a conservative actuator box contained in one regular,
collision-free forward-kinematic branch. Only after the forward map is shown to
be injective over that box may the full mechanism component be asserted to be
diffeomorphic to `I^3`; the unrestricted end-effector workspace is not the
configuration space.

The SCARA assertion requires three genuinely continuous rotary bearings,
collision-free vertically separated links, and one bounded vertical slide.
With realistic rotary stops its topology would instead be `I^4`, so the
physical design must match the claimed `T^3 x I` topology exactly.

If distinct physical states produce exactly the same observable frame, the
problem is not identifiable from the allowed inputs. Such a case must report an
identifiability failure rather than silently joining the states or claiming a
valid atlas.

## Case requirements

Every primary mechanism must define:

1. a physical simulator and frame representation;
2. a connected regular evaluation component;
3. smoke, qualification, and robustness sample budgets;
4. independently generated training and held-out frames;
5. physical validity checks for decoded off-anchor probes;
6. an evaluator-only reference characterization;
7. explicit evidence that prohibited structural information never reached the
   fitter or runtime router.

Contact cases must additionally identify which contact transitions preserve
the fixed-dimensional manifold-with-corners assumption. Impacts, branching
hybrid trajectories, dimension-changing active sets, and disconnected feasible
sets must not be silently folded into an ordinary passing case.

## Robustness protocol

A qualification result requires more than one favorable run. Unless a case has
an explicitly justified exception, evaluate:

- at least three training seeds;
- at least two independently generated held-out sets;
- uniform, nonuniform, and sparse-region sampling variants;
- shuffled sample order;
- small link-length, limit, contact-location, and marker perturbations;
- off-anchor decoder probes throughout learned chart interiors and frontiers;
- overlap round trips and transition Jacobians wherever multiple charts meet;
- chart-count and coverage stability across seeds;
- explicit detection of non-identifying observation layouts.

The robustness report must separate training coverage, held-out coverage,
reconstruction, physical validity, injectivity, overlap consistency, selector
calibration, and topology-sensitive evaluation. Aggregate `pass` is valid only
when all mandatory categories pass.

## Topology-sensitive evaluation

Coverage and reconstruction alone cannot establish a valid atlas. The offline
evaluator should compare the learned atlas nerve and transition structure with
connectivity and homology obtained independently from dense simulator evidence.
This information is evaluation-only and must not influence training, chart
creation, pruning, or routing.

Topology expectations such as `T^2`, `Sigma_4`, or `SO(3)` are test assertions,
not learner inputs. For cases such as locked 3-SPS whose regular component
depends strongly on geometry, the reference characterization must be computed
and frozen for the selected physical design before qualification runs.

## Four-dimensional limit

The compact SCARA is the only required 4D case. It should have approximately
four moving bodies, a sparse but identifying marker set, no contact, and a
deliberately small smoke-scale frame budget. Its purpose is to detect
catastrophic dimensional-scaling failures, not to claim dense or general
coverage of arbitrary 4D manifolds.

No 4D result should weaken the physical validity, no-priors, or held-out
requirements merely to obtain a nominal pass.

## Implementation sequence

1. Freeze the evaluator/learner information boundary and common report schema.
2. Add physical regular-stratum stops to the coupled scissor lift and qualify
   it together with the existing Jansen case.
3. Qualify Stephenson, implement the open bevel differential, and retain the
   existing five-bar `Sigma_4` regression in that order.
4. Design and certify the obstacle-constrained 2R collision complement.
5. Implement the spherical pendulum with a physically faithful observation
   representation and qualify the existing ball-and-socket `SO(3)` fixture.
6. Port and qualify the stock locked 3-SPS fixture, then add the conservative
   regular-branch Delta case.
7. Add the compact continuous-rotation SCARA smoke case only after 1D--3D
   qualification reporting is stable.
8. Promote individual cases from smoke to qualification only after their
   physical and evaluator-only reference checks are reproducible.
