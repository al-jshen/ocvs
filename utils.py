import pandas as pd
import numpy as np
from numba import njit

def phase_fold(example):
    df = pd.DataFrame({"time": example["time"], "flux": example["flux"]})

    # 1. Fold the time
    df["folded_time"] = df["time"] % example["period"]

    # 2. Identify cycles
    df["cycle"] = (df["time"] // example["period"]).astype(int)

    # 3. Normalize: Group by cycle and divide the flux by the median of that group
    # The 'transform' function ensures the output has the same shape as the input
    df["norm_flux"] = df.groupby("cycle")["flux"].transform(lambda x: x / x.median())

    example["folded_time"] = df["folded_time"].values
    example["norm_flux"] = df["norm_flux"].values

    return example


def phase_fold_batch(examples):
    # 1. Capture original lengths and flatten the ragged arrays
    lengths = [len(t) for t in examples["time"]]
    flat_time = np.concatenate(examples["time"])
    flat_flux = np.concatenate(examples["flux"])
    
    # 2. Vectorize the periods so they match the flattened data
    # This creates an array where the period is repeated for every timestamp in that example
    flat_periods = np.repeat(examples["period"], lengths)
    
    # 3. Vectorize the batch index to ensure cycles are unique across different examples
    # (Example 0, Cycle 1 is different from Example 1, Cycle 1)
    batch_idx = np.repeat(np.arange(len(lengths)), lengths)

    # 4. Perform calculations on the flat arrays
    folded_time = flat_time % flat_periods
    cycles = (flat_time // flat_periods).astype(int)
    
    # Create a unique ID for every cycle in every batch element
    # This prevents "Cycle 1" of star A from being grouped with "Cycle 1" of star B
    unique_cycle_id = (batch_idx.astype(np.int64) << 32) | cycles.astype(np.int64)

    # 5. Fast grouping using Pandas (highly optimized for flat arrays)
    df = pd.DataFrame({"flux": flat_flux, "gid": unique_cycle_id})
    norm_flux = df.groupby("gid")["flux"].transform(lambda x: x / x.median()).values

    # 6. Resplit back into the original ragged structure
    # We use the cumulative sum of lengths to find split points
    split_indices = np.cumsum(lengths)[:-1]
    
    examples["folded_time"] = np.split(folded_time, split_indices)
    examples["norm_flux"] = np.split(norm_flux, split_indices)

    return examples


import numpy as np
from numba import njit

@njit
def fold_and_norm_kernel(time, flux, period):
    folded = time % period
    cycles = (time // period).astype(np.int32)
    unique_cycles = np.unique(cycles)
    
    norm_f = np.empty_like(flux)
    for c in unique_cycles:
        mask = (cycles == c)
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
        folded_list.append(f_t)
        norm_list.append(f_f)
        
    examples["folded_time"] = np.array(folded_list, dtype=object)
    examples["norm_flux"] = np.array(norm_list, dtype=object)
    return examples
