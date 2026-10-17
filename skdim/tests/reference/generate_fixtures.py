# SPDX-License-Identifier: BSD-3-Clause
"""Regenerate fixtures.json from reference implementations: pixi run -e ref make-fixtures

R: intrinsicDimension, ider, PCAtools, scran (reference.R). Python: DADApy.
Only their outputs are stored; no code is taken from them."""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import id_datasets, rank_datasets  # noqa: E402


def r_reference():
    with tempfile.TemporaryDirectory() as tmp:
        for kind, sets in (("id", id_datasets()), ("rank", rank_datasets())):
            for name, X in sets.items():
                np.savetxt(os.path.join(tmp, f"{kind}_{name}.csv"), X, delimiter=",", fmt="%.17g")
        out = os.path.join(tmp, "out.json")
        subprocess.run(["Rscript", os.path.join(HERE, "reference.R"), tmp, out], check=True)
        with open(out) as f:
            return json.load(f)


def dadapy_reference():
    import dadapy

    res = {}
    for name, X in id_datasets().items():
        data = lambda: dadapy.Data(X.copy(), verbose=False)  # noqa: E731
        ids, errs, _ = data().return_id_scaling_gride(range_max=64)
        res[name] = {
            "twonn_base_mu0.9": data().compute_id_2NN(algorithm="base", mu_fraction=0.9)[0],
            "twonn_ml": data().compute_id_2NN(algorithm="ml", mu_fraction=1.0)[0],
            "gride_range64_ids": list(ids),
            "gride_range64_errs": list(errs),
        }
    return res


def main():
    import dadapy

    fx = r_reference()
    fx["versions"]["dadapy"] = getattr(dadapy, "__version__", "git d729b88")
    fx["versions"]["numpy"] = np.__version__
    fx["dadapy"] = dadapy_reference()
    with open(os.path.join(HERE, "fixtures.json"), "w") as f:
        json.dump(fx, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
