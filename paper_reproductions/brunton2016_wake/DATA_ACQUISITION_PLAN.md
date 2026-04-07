# Data Acquisition Plan

## Objective

Get a `Re = 100` 2D cylinder-wake dataset that is close enough to the Brunton 2016 wake example to support a faithful SINDy reproduction.

## Recommended Path

Use `hydrogym` to generate uncontrolled cylinder-wake trajectories with a restart checkpoint.

Why this is the cheapest realistic option:

- the cylinder wake is already implemented
- `Re = 100` is explicitly supported in the docs
- restart checkpoints are supported, which reduces spin-up cost
- the library already targets reduced-order modeling and flow-control benchmarks

## Step-by-Step Plan

### Phase 1: Environment Setup

1. Acquire `hydrogym`
2. Prefer the documented Docker-based route if available on this machine
3. Use the Firedrake-backed cylinder environment
4. Verify that a `Re = 100` uncontrolled cylinder-wake case runs from a restart file

Expected output:

- a working cylinder-wake environment
- access to a restart / checkpointed state

### Phase 2: Minimal Data Probe

1. Run a short uncontrolled rollout with `action = 0`
2. Confirm periodic shedding appears
3. Save a tiny pilot dataset:
   - `50` to `200` snapshots
   - full field or vorticity field
   - fixed sample interval
4. Check that snapshots can be exported to NumPy arrays or HDF5

Expected output:

- proof that we can actually extract state data, not just lift/drag observations

### Phase 3: Study Dataset Generation

1. Start from a restart state, not from rest
2. Run an uncontrolled wake trajectory long enough to cover many shedding cycles
3. Save a consistent snapshot sequence for POD
4. Save metadata:
   - Reynolds number
   - timestep
   - sampling stride
   - solver backend
   - checkpoint source

Recommended first-pass dataset target:

- `2000` to `5000` snapshots
- constant sample spacing
- one clean post-transient trajectory

### Phase 4: Reduced Coordinates

1. Build POD modes from the saved snapshots
2. Recover the low-order wake coordinates
3. Check for the expected geometry:
   - oscillatory pair
   - shift / mean-flow mode
4. Only then move into SINDy identification

## What Data We Actually Need

Minimum useful data:

- state snapshots on a fixed mesh
- enough time resolution to estimate derivatives or a reliable map
- long enough post-transient trajectory to reveal the limit cycle and mean-flow distortion

Best-case data:

- velocity or vorticity snapshots
- access to the mean flow and unstable steady / restart state
- metadata on mesh, nondimensionalization, and sample interval

## Cost Estimate

These are rough order-of-magnitude estimates, not guarantees.

### Setup Cost

- `hydrogym` install plus Firedrake/Docker setup:
  - time: `30` to `120` minutes
  - disk: roughly `5` to `20+ GB`
  - risk: medium

This is the largest upfront cost.

### Checkpoint / Environment Download

- checkpoint download on first use:
  - time: `5` to `30` minutes
  - disk: `hundreds of MB` to `a few GB`
  - risk: low to medium

### Pilot Rollout

- short uncontrolled trajectory:
  - time: `5` to `20` minutes
  - disk: tiny if saving only a probe dataset
  - risk: low

### Full Study Dataset

- `2000` to `5000` saved snapshots:
  - time: `20` to `90` minutes
  - disk: about `1` to `10+ GB`, depending on field type, mesh resolution, and file format
  - risk: medium

### POD + SINDy After Data Exists

- POD / reduced coordinates:
  - time: `minutes` to `tens of minutes`
- SINDy fitting / sweeps:
  - time: `minutes`
- this is much cheaper than the CFD data-generation step

## Cheapest Sensible First Attempt

If we want to minimize cost before committing:

1. install or access `hydrogym`
2. run one short `Re = 100` uncontrolled cylinder test from restart
3. export `100` to `200` snapshots
4. confirm we can turn them into a NumPy matrix for POD

Only after that should we generate the full wake dataset.

## Success Criteria For Moving On

We can move from "data acquisition" to "paper reproduction" once we have:

- a post-transient uncontrolled cylinder-wake trajectory
- fixed-interval state snapshots on disk
- a verified export format we can feed into POD / SINDy
- enough data to estimate the first few reduced coordinates cleanly

## Notes

The main uncertainty is not whether `hydrogym` can simulate a wake. It can.
The main uncertainty is whether it exposes the full-state snapshots conveniently enough for our POD/SINDy path without extra extraction work.

## Update: Cheaper Direct Data Path Chosen

We now have a cheaper immediate path than `hydrogym`:

- downloaded Zenodo wake dataset:
  - [fixed_cylinder_atRe100](c:/Users/nadir/Desktop/MAT6215/paper_reproductions/brunton2016_wake/data/fixed_cylinder_atRe100)

This shifts the near-term plan to:

1. parse and inspect the downloaded dataset
2. verify that the mesh is fixed across time
3. build velocity-snapshot matrices for POD
4. use `hydrogym` only if this dataset turns out to be too far from the wake-paper needs

### Revised Immediate Cost

- already paid:
  - about `1.2 GB` disk for the raw data file
- next parsing / inspection cost:
  - time: `minutes`
  - additional disk: small unless we materialize full NumPy snapshot arrays

So the immediate path is now much cheaper than spinning up a fresh CFD environment.
