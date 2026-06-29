import argparse
from pathlib import Path

import numpy as np


def load_accumulated_heatflux(path: Path, start: int, dt_ps: float, sample_interval: int,
                              block_length: float):
    data = np.loadtxt(path)
    data = data[start:]
    time_ns = dt_ps * np.arange(1, len(data) + 1) * sample_interval / 1000.0

    jpy = data[:, 2] - data[:, 5]
    jpy = jpy / block_length / 10.0 * 1000.0
    accum_jpy = np.cumsum(jpy) * dt_ps / 1000.0
    return time_ns, -accum_jpy


def compare_heatflux(old_path: Path, new_path: Path, start: int, max_time_ns: float,
                     dt_ps: float, sample_interval: int, block_length: float):
    old_t, old_hf = load_accumulated_heatflux(old_path, start, dt_ps, sample_interval,
                                              block_length)
    new_t, new_hf = load_accumulated_heatflux(new_path, start, dt_ps, sample_interval,
                                              block_length)

    npts = min(len(old_hf), len(new_hf))
    old_t = old_t[:npts]
    old_hf = old_hf[:npts]
    new_hf = new_hf[:npts]

    mask = old_t <= max_time_ns
    old_sel = old_hf[mask]
    new_sel = new_hf[mask]

    ref_max = np.max(np.abs(old_sel))
    if ref_max < 1e-12:
        raise ValueError("Legacy heat flux is essentially zero in the selected window")

    diff = np.abs(new_sel - old_sel)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    rel_max_err = max_diff / ref_max

    ss_res = float(np.sum((new_sel - old_sel) ** 2))
    ss_tot = float(np.sum((old_sel - np.mean(old_sel)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0

    print("=== MatPL Heat Flux Path Comparison ===")
    print(f"Legacy file: {old_path}")
    print(f"MatPL file:  {new_path}")
    print(f"Points compared: {int(np.sum(mask))}")
    print(f"Time window: [0, {max_time_ns}] ns")
    print(f"Max absolute difference: {max_diff:.6e} keV")
    print(f"Mean absolute difference: {mean_diff:.6e} keV")
    print(f"Relative max error: {rel_max_err:.6%}")
    print(f"R-squared: {r_squared:.8f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare legacy and MatPL heat flux outputs from the same run"
    )
    parser.add_argument("legacy", nargs="?", default="compute_HeatFlux_legacy.out")
    parser.add_argument("matpl", nargs="?", default="compute_HeatFlux_matpl.out")
    parser.add_argument("--start", type=int, default=500)
    parser.add_argument("--max-time-ns", type=float, default=1.5)
    parser.add_argument("--dt-ps", type=float, default=0.001)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--block-length", type=float, default=426.0)
    args = parser.parse_args()

    compare_heatflux(
        Path(args.legacy),
        Path(args.matpl),
        args.start,
        args.max_time_ns,
        args.dt_ps,
        args.sample_interval,
        args.block_length,
    )


if __name__ == "__main__":
    main()