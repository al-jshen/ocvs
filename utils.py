from datasets import load_dataset
from functools import partial
import numpy as np
from numba import njit
from scipy.stats import binned_statistic


@njit
def fold_and_norm_kernel(time, flux, period):
    folded = time % period
    cycles = (time // period).astype(np.int32)
    unique_cycles = np.unique(cycles)

    norm_f = np.empty_like(flux)
    for c in unique_cycles:
        mask = cycles == c
        if np.any(mask):
            # Numba's np.median is efficient
            m = np.median(flux[mask])
            norm_f[mask] = flux[mask] / m
    return folded, norm_f


def phase_fold_batch_numba(examples):
    # Convert numpy object arrays to standard lists so Numba can iterate
    times = examples["time"]
    fluxes = examples["flux"]
    periods = examples["period"]

    folded_list = []
    norm_list = []

    # We call the NJIT kernel for each lightcurve
    # The loop is in Python, but the heavy math is in Numba
    for i in range(len(times)):
        f_t, f_f = fold_and_norm_kernel(times[i], fluxes[i], periods[i])
        folded_list.append(f_t.astype(np.float32))
        norm_list.append(f_f.astype(np.float32))

    examples["folded_time"] = np.array(folded_list, dtype=object)
    examples["norm_flux"] = np.array(norm_list, dtype=object)
    return examples


def normalize_folded_time(example):
    # Normalize folded time to range [0, 1]
    folded_time = example["folded_time"]
    min_t = np.min(folded_time)
    max_t = np.max(folded_time)
    normalized_time = (folded_time - min_t) / (max_t - min_t)
    # overwrite
    example["folded_time"] = normalized_time.astype(np.float32)
    return example


def bin_data(example, num_bins=50):
    """
    Bins phase-folded data into equal-width intervals.

    Parameters:
    - example: A dictionary containing "time", "flux", and "period".
    - num_bins: How many bins to divide the period into.

    Returns:
    - Updated example with "bin_centers", "bin_means", and "bin_errors".
    """
    time = example["time"]
    flux = example["flux"]
    period = example["period"]

    bin_edges = np.linspace(0, 1., num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    bin_means, _, _ = binned_statistic(
        time, flux, statistic="mean", bins=num_bins, range=(0, 1)
    )

    bin_stds, _, _ = binned_statistic(
        time, flux, statistic="std", bins=num_bins, range=(0, 1)
    )
    bin_counts, _, _ = binned_statistic(
        time, flux, statistic="count", bins=num_bins, range=(0, 1)
    )
    bin_errors = bin_stds / np.sqrt(bin_counts)

    # Update example with binned data
    example["bin_centers"] = bin_centers.astype(np.float32)
    example["bin_means"] = bin_means.astype(np.float32)
    example["bin_errors"] = bin_errors.astype(np.float32)
    example["bin_counts"] = bin_counts

    return example


def make_binned_version(num_bins=48, num_proc=8):
    ds = load_dataset("j-shen/ocvs-folded", split="train", num_proc=num_proc)
    ds = ds.map(partial(bin_data, num_bins=num_bins), num_proc=num_proc)
    ds = ds.filter(lambda x: ~np.isnan(x['bin_means']).any(), num_proc=num_proc)
    ds = ds.remove_columns(["flux", "time"])
    ds = ds.rename_columns({"bin_centers": "time", "bin_means": "flux", "bin_errors": "flux_err"})
    return ds


def to_generator(ds):
    """Convert from HF dataset to generator."""

    for i in range(len(ds)):
        sample_data = ds[i]

        # Create aligned data (time series)
        aligned_data = {
            "time": sample_data["time"],
            "flux": sample_data["flux"],
            "flux_err": sample_data["flux_err"],
            # "istd": sample_data["istd"],
            # "ivar": sample_data["ivar"],
            # "mask": sample_data["mask"],
            "folded_time": sample_data["folded_time"],
            "norm_flux": sample_data["norm_flux"],
        }

        # Create metadata
        meta_data = {
            "ra": sample_data["ra"],
            "dec": sample_data["dec"],
            "notes": sample_data["notes"],
            "variable_type": sample_data["variable_type"],
            "period": sample_data["period"],
            "period_err": sample_data["period_err"],
        }

        yield {"id": str(sample_data["id"]), "aligned": aligned_data, "meta": meta_data}
