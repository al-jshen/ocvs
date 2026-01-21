import argparse
import os

import astropy.units as u
import pandas as pd
from astropy.coordinates import SkyCoord
from tqdm.auto import tqdm


def convert_radec(ra, dec):
    """
    Convert RA and Dec from HMS and DMS formats to decimal degrees.
    """
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.hourangle, u.deg), frame="icrs")
    return coord.ra.deg, coord.dec.deg


def read_period(path):
    # ok, this is nasty. have to handle case separately.
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    try:
        df = pd.read_fwf(path, header=None)
        filename = os.path.basename(path)
        if filename in [
            "cepF.dat",
            "cep1O.dat",
            "dsct.dat",
            "RRab.dat",
            "RRc.dat",
            "RRd.dat",
            "t2cep.dat",
            "acepF.dat",
            "acep1O.dat",
        ]:
            df = df[[0, 3, 4]]
            df.columns = ["id", "period", "period_err"]

        elif filename in ["ecl.dat", "ell.dat", "hb.dat", "Miras.dat"]:
            df = df[[0, 3]]
            df.columns = ["id", "period"]
            df["period_err"] = float("nan")

        elif filename in ["rot.dat"]:
            df = df[[0, 5]]
            df.columns = ["id", "period"]
            df["period_err"] = float("nan")

        else:
            raise ValueError(f"Unknown period file format: {filename}")

        return df
    except Exception as e:
        raise ValueError(f"Error reading or processing the file {path}: {e}")


def read_ident(path):
    """
    Reads the ident file, performs processing, and returns a DataFrame.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")

    try:
        df = pd.read_fwf(path, header=None)
        if "blg/rot" in path:  # special case for blg/rot files, no notes column
            df = df[[0, 1, 2]]
            df.columns = ["id", "ra", "dec"]
            df["notes"] = ""  # add empty notes column
        else:
            df = df[[0, 1, 2, 3]]
            df.columns = ["id", "notes", "ra", "dec"]
        df["ra"], df["dec"] = convert_radec(df["ra"], df["dec"])
        # extract location and variable type from the id column
        # id format: OGLE-<location>-<variable_type>-<number>
        df["location"] = df["id"].str.extract(r"OGLE-(\w+)-")
        df["variable_type"] = df["id"].str.extract(r"OGLE-\w+-(\w+)-")
        return df
    except Exception as e:
        raise ValueError(f"Error reading or processing the file {path}: {e}")


def main(args):
    # find all `.dat` files in the given directory recursively
    print("Searching recursively for .dat files...")
    ident_files = []
    period_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".dat"):
                if file == "ident.dat":
                    ident_files.append(os.path.join(root, file))
                else:
                    period_files.append(os.path.join(root, file))

    if not ident_files:
        raise FileNotFoundError("No ident.dat files found in the specified directory.")
    if not period_files:
        raise FileNotFoundError(
            "No period .dat files found in the specified directory."
        )

    print(f"Found {len(ident_files)} ident.dat files.")
    print(f"Found {len(period_files)} period .dat files.")

    # read all ident files and concatenate them into a single DataFrame
    # using the read_ident function
    ident_dfs = [read_ident(file) for file in tqdm(ident_files)]
    ident_df = pd.concat(ident_dfs, ignore_index=True)

    # read all period files and concatenate them into a single DataFrame
    # using the read_period function
    period_dfs = [read_period(file) for file in tqdm(period_files)]
    period_df = pd.concat(period_dfs, ignore_index=True)

    # merge the ident and period DataFrames on the 'id' column
    full_df = ident_df.merge(period_df, on="id", how="inner")

    # save the concatenated DataFrame to a new file
    # using the path specified in the args
    # if the path does not exist, create it
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "ident.csv")
    full_df.to_csv(output_path, index=False)
    print(f"Ident and period data saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare ident data from ident.dat files."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Path to the directory containing ident.dat files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory for the concatenated ident.csv file.",
    )

    args = parser.parse_args()
    main(args)
