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
        folded_list.append(f_t.astype(np.float32))
        norm_list.append(f_f.astype(np.float32))
        
    examples["folded_time"] = np.array(folded_list, dtype=object)
    examples["norm_flux"] = np.array(norm_list, dtype=object)
    return examples
