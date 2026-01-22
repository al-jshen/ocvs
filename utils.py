import pandas as pd


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
