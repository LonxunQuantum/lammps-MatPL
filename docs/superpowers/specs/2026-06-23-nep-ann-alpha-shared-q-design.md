# NEP ANN Alpha Cache and Shared-Q Design

## Scope

Optimize the non-templated ANN path introduced by commit `e5db547`. Preserve
the descriptor kernels and all force kernels. Deliver V2-A and V2-B as separate
commits with independent numerical, resource, Nsight Systems, and Nsight
Compute verification.

## V2-A: Alpha Cache and Tiled Derivative

Replace `apply_ann` with two kernels. `apply_ann_forward` retains one thread per
atom, evaluates hidden-neuron activations and energy, and writes
`alpha = w1 * (1 - tanh(z)^2)` in `[neuron][n1]` SoA layout. Allocate the alpha
buffer once as `max_nlocal * num_neurons1` floats.

`apply_ann_derivative` processes eight `n1` values per 256-thread block. It
loads each tile's alpha values once into shared memory, then maps work items to
`(descriptor, atom-in-tile)` and writes the scaled derivative to SoA `g_Fp`.
This removes `Fp[MAX_DIM]` and its local-memory load/store traffic. V2-A leaves
q reads unchanged so its effect is independently measurable.

## V2-B: Shared-Q Forward Pass

Extend only `apply_ann_forward`. Cooperatively load scaled q into strided
shared memory `[descriptor][thread]` before the neuron loop. All threads reach
the synchronization point, including inactive tail threads. The neuron loop
then reads q from shared memory. Start with 64 threads per block and measure
32/64/128 only if the full NCU result regresses.

## Verification

For each version: compile build-32, compare sorted per-atom force and energy
dumps against the preceding committed version with a `1e-5` threshold, inspect
resource usage, run Nsight Systems on the 2M-atom WMoTaV case, and collect five
full NCU launches for every new ANN kernel. Commit only after that version
passes.

