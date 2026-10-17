# SPDX-License-Identifier: BSD-3-Clause
"""Regenerate fixtures.json from reference implementations: pixi run -e ref make-fixtures

R: intrinsicDimension, ider, PCAtools, scran (reference.R). Python: DADApy.
Octave: the original MATLAB TLE/MOM scripts and FisherSeparabilityAnalysis (reference.m),
downloaded at pinned commits.
Only their outputs are stored; no code is taken from them."""
import json
import os
import subprocess
import sys
import tempfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data import dup_datasets, id_datasets, rank_datasets  # noqa: E402


def r_reference():
    with tempfile.TemporaryDirectory() as tmp:
        for kind, sets in (("id", id_datasets()), ("rank", rank_datasets())):
            for name, X in sets.items():
                np.savetxt(os.path.join(tmp, f"{kind}_{name}.csv"), X, delimiter=",", fmt="%.17g")
        out = os.path.join(tmp, "out.json")
        subprocess.run(["Rscript", os.path.join(HERE, "reference.R"), tmp, out], check=True)
        with open(out) as f:
            return json.load(f)


# original MATLAB sources, pinned
MATLAB_SOURCES = {
    "https://raw.githubusercontent.com/radacha/tle/cb11c7f2760babfc96f175be30e7d074d7fb16ae/": ["idtlec.m", "idmom.m"],
    "https://raw.githubusercontent.com/auranic/FisherSeparabilityAnalysis/cb0d2ab26408f045215b859582a2637aa0241cf0/MATLAB/": [
        "SeparabilityAnalysis.m", "checkSeparabilityMultipleAlpha.m", "dimension_uniform_sphere.m",
        "probability_unseparable_sphere.m", "preprocessing.m"],
}


def octave_reference(n_neighbors=20):
    from skdim._commonfuncs import get_nn

    res = {}
    with tempfile.TemporaryDirectory() as tmp:
        for base, names in MATLAB_SOURCES.items():
            for name in names:
                urllib.request.urlretrieve(base + name, os.path.join(tmp, name))
        # Octave lacks cumsum(..., 'reverse'); substitute the identical flip(cumsum(flip(.)))
        f = os.path.join(tmp, "checkSeparabilityMultipleAlpha.m")
        src = open(f).read()
        assert "cumsum(counts, 2, 'reverse')" in src
        open(f, "w").write(src.replace("cumsum(counts, 2, 'reverse')", "flip(cumsum(flip(counts, 2), 2), 2)"))
        for name, X in {**id_datasets(), **dup_datasets()}.items():
            dists, idx = get_nn(X, n_neighbors)
            np.savetxt(os.path.join(tmp, f"{name}_X.csv"), X, delimiter=",", fmt="%.17g")
            np.savetxt(os.path.join(tmp, f"{name}_idx.csv"), idx + 1, delimiter=",", fmt="%d")
            np.savetxt(os.path.join(tmp, f"{name}_dists.csv"), dists, delimiter=",", fmt="%.17g")
        subprocess.run(["octave-cli", "-q", os.path.join(HERE, "reference.m"), tmp], check=True)
        for name in {**id_datasets(), **dup_datasets()}:
            out = lambda key: np.loadtxt(os.path.join(tmp, f"{name}_out_{key}.csv"), delimiter=",")  # noqa: E731
            fisher = out("fisher")
            res[name] = {"n_neighbors": n_neighbors, "tle_pw": out("tle").tolist(), "mom_pw": out("mom").tolist(),
                         "fisher_n_single": float(fisher[0]), "fisher_n_alpha": fisher[1:].tolist()}
    return res


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
    fx["matlab"] = octave_reference()
    fx["versions"]["octave"] = subprocess.run(["octave-cli", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    with open(os.path.join(HERE, "fixtures.json"), "w") as f:
        json.dump(fx, f, indent=1, sort_keys=True)
        f.write("\n")


if __name__ == "__main__":
    main()
