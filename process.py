import argparse
import os
import tarfile
from functools import partial, reduce

import healpy as hp
import joblib
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import process_map


def ang2pix(ra, dec, nside):  # input in degrees
    return hp.ang2pix(nside=nside, theta=ra, phi=dec, lonlat=True, nest=True)


def convert_to_arrow(data):
    object_ids = []
    bands = []
    times = []
    fluxes = []
    flux_errs = []

    for obj_id, band_dict in data.items():
        for band, values in band_dict.items():
            try:
                object_ids.append(obj_id)
                bands.append(band)
                times.append(values["hjd"].astype("float32").tolist())
                fluxes.append(values["flux"].astype("float32").tolist())
                flux_errs.append(values["flux_err"].astype("float32").tolist())
            except Exception as e:
                # dump data to a file for debugging
                joblib.dump(
                    {
                        "obj_id": obj_id,
                        "band": band,
                        "values": values,
                        "error": str(e),
                    },
                    f"error_{obj_id}_{band}.pkl",
                )
                raise ValueError(f"Error processing {obj_id} in band {band}: {e}")

    # Create list columns
    table = pa.table(
        {
            "id": pa.array(object_ids, type=pa.string()),
            "band": pa.array(bands, type=pa.string()),
            "time": pa.array(times, type=pa.list_(pa.float32())),
            "flux": pa.array(fluxes, type=pa.list_(pa.float32())),
            "flux_err": pa.array(flux_errs, type=pa.list_(pa.float32())),
        }
    )

    return table


def main(args):
    tar_files = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".tar.gz"):
                tar_files.append(os.path.join(root, file))
    if not tar_files:
        raise ValueError("No .tar.gz files found in the input directory.")

    print(f"Found {len(tar_files)} .tar.gz files in the input directory.")

    # tar_files = [i for i in tar_files if "blg/ecl" not in i] # debugging, just use one file...
    __import__("pprint").pprint(tar_files)

    os.makedirs(args.output_dir, exist_ok=True)

    dfs = process_map(
        partial(tar_to_dfs, tiny=args.tiny),
        tar_files if not args.tiny else tar_files[:5],  # limit files if tiny
        max_workers=args.num_workers,
        desc="Extracting data from tar files",
    )
    results = reduce(lambda x, y: {**x, **y}, dfs)

    # dfs = tar_to_dfs(tar_files[0], tiny=args.tiny) # debugging...
    # results = dfs

    print(f"Total number of unique identifiers: {len(results)}")

    print("Converting to pyarrow")

    table = convert_to_arrow(
        results
    ).to_pandas()  # going through arrow is faster than direct to pandas??

    print("Merging with ident.csv")

    ident = pd.read_csv(args.ident_file)

    joined = table.merge(ident, on="id", how="inner")

    # convert back to pyarrow
    table = pa.Table.from_pandas(joined)

    # partition by healpix
    try:
        healpix = ang2pix(
            table["ra"].to_numpy(),
            table["dec"].to_numpy(),
            nside=args.nside,
        )
    except Exception as e:
        joblib.dump(
            {
                "ra": table["ra"].to_numpy(),
                "dec": table["dec"].to_numpy(),
                "id": table["id"].to_numpy(),
                "nside": args.nside,
                "error": str(e),
            },
            "healpix_error.pkl",
        )
        raise ValueError(f"Error converting RA/DEC to Healpix: {e}")
    table = table.append_column("healpix", pa.array(healpix, type=pa.int64()))

    print("Writing to disk")
    pq.write_to_dataset(table, args.output_dir, partition_cols=["healpix"])


def pd_to_dict(df):
    """Converts a DataFrame to a dictionary with all columns as keys."""
    return {col: df[col].to_numpy() for col in df.columns}


def tar_to_dfs(tar_file, tiny=False):
    """Extracts all CSV files from a tar.gz file and returns them as a list of
    DataFrames."""
    results = {}
    with tarfile.open(tar_file, "r:gz") as tar:
        members = tar.getmembers()
        if tiny:
            members = members[:1000]
        print(f"Processing {len(members)} members in {tar_file}")
        for member in tqdm(members):
            if member.name.endswith(".dat") and not os.path.basename(
                member.name
            ).startswith("._"):
                f = tar.extractfile(member)
                i_or_v, fname = os.path.split(
                    member.name
                )  # phot/V/OGLE-SMC-ACEP-122.dat for example
                i_or_v = os.path.split(i_or_v)[-1]  # turn from phot/V -> V
                fname = fname.replace(".dat", "")  # strip extension
                if f is not None:
                    try:
                        df = pd.read_csv(
                            f,
                            header=None,
                            delimiter=r"\s+",
                            names=["hjd", "flux", "flux_err"],
                        )  # a bit more robust than read_fwf but still sometimes fails?
                    except Exception as e1:
                        print(
                            f"Error reading {member.name} with read_csv: {e1}, trying read_fwf"
                        )
                        try:
                            df = pd.read_fwf(
                                f, header=None, names=["hjd", "flux", "flux_err"]
                            )
                        except Exception as e2:
                            raise ValueError(
                                f"Error reading {member.name} with read_fwf: {e2}"
                            )
                    df = pd_to_dict(df)
                    if fname not in results:
                        results[fname] = {}
                    results[fname][i_or_v] = df
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing the .tar.gz files",
    )
    parser.add_argument(
        "--ident_file",
        type=str,
        required=True,
        help="Path to consolidated ident.csv.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the processed dataset.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of worker processes to use for parallel processing.",
    )
    parser.add_argument(
        "--nside",
        type=int,
        default=16,
        help="Nside parameter for Healpix partitioning.",
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a tiny dataset for testing.",
        default=False,
    )
    args = parser.parse_args()

    main(args)
