import argparse
import os


def main(args):
    base_path = "https://www.astrouw.edu.pl/ogle/ogle4/OCVS/"
    paths = {
        "blg": {
            "cep": ("phot", ["cepF.dat", "cep1O.dat"]),
            "dsct": ("phot_ogle4", ["dsct.dat"]),
            "ecl": ("phot_ogle4", ["ecl.dat", "ell.dat"]),
            "hb": ("phot/phot", ["hb.dat"]),
            "lpv": ("phot_ogle4", ["Miras.dat"]),
            "rot": ("phot_ogle4", ["rot.dat"]),
            "rrlyr": ("phot", ["RRab.dat", "RRc.dat", "RRd.dat"]),
            "t2cep": ("phot", ["t2cep.dat"]),
        },
        "gal": {"acep": ("phot", ["acepF.dat", "acep1O.dat"])},
        "gd": {
            "cep": ("phot", ["cepF.dat", "cep1O.dat"]),
            "dsct": ("phot_ogle4", ["dsct.dat"]),
            "lpv": ("phot_ogle4", ["Miras.dat"]),
            "rrlyr": ("phot", ["RRab.dat", "RRc.dat", "RRd.dat"]),
            "t2cep": ("phot", ["t2cep.dat"]),
        },
        "lmc": {
            "acep": ("phot", ["acepF.dat", "acep1O.dat"]),
            "cep": ("phot", ["cepF.dat", "cep1O.dat"]),
            "dsct": ("phot", ["dsct.dat"]),
            "ecl": ("phot", ["ecl.dat", "ell.dat"]),
            "hb": ("phot/phot", ["hb.dat"]),
            "rrlyr": ("phot", ["RRab.dat", "RRc.dat", "RRd.dat"]),
            "t2cep": ("phot", ["t2cep.dat"]),
        },
        "smc": {
            "acep": ("phot", ["acepF.dat", "acep1O.dat"]),
            "cep": ("phot", ["cepF.dat", "cep1O.dat"]),
            "dsct": ("phot", ["dsct.dat"]),
            "ecl": ("phot", ["ecl.dat", "ell.dat"]),
            "hb": ("phot/phot", ["hb.dat"]),
            "rrlyr": ("phot", ["RRab.dat", "RRc.dat", "RRd.dat"]),
            "t2cep": ("phot", ["t2cep.dat"]),
        },
    }

    to_download = []
    for path, subpaths in paths.items():
        for subpath, (subsubpath, period_files) in subpaths.items():
            out_path = os.path.join(
                args.output_dir,
                path,
                subpath,
            )
            os.makedirs(out_path, exist_ok=True)

            data_path = os.path.join(base_path, path, subpath, subsubpath + ".tar.gz")
            to_download.append((data_path, out_path))

            ident_path = os.path.join(base_path, path, subpath, "ident.dat")
            to_download.append((ident_path, out_path))

            for period_file in period_files:
                period_path = os.path.join(base_path, path, subpath, period_file)
                to_download.append((period_path, out_path))

    with open("ogle_data_paths.txt", "w") as f:
        for file_path, out_path in to_download:
            write_path = os.path.join(out_path, os.path.basename(file_path))
            f.write(f"{file_path}\n\tout={write_path}\n")

    os.system("aria2c -j16 -s16 -x16 -c -i ogle_data_paths.txt")

    os.remove("ogle_data_paths.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare OGLE data files.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="ogle_data",
        help="Output directory for downloaded files.",
    )
    args = parser.parse_args()
    main(args)
