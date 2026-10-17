# SPDX-License-Identifier: BSD-3-Clause
"""Regenerate skdim/id/DANCoFit/danco_fit_table.npz, the calibration table of DANCo(ver="DANCoFit").

For each dimension d = 1..D and cardinality N, averages DANCo's calibration statistics (dhat,
mu_nu, mu_tau) over `iterations` uniform samples of the d-ball, as DANCo does with one sample.
The shipped table used the defaults below (from the original DANCoTrain utility). Expensive:
D * len(CARDINALITIES) * iterations calls. Usage: python tools/make_danco_fit_table.py [out.npz]
"""
import sys

import numpy as np
from skdim.datasets import hyperBall
from skdim.id import DANCo

K, D, ITERATIONS = 10, 100, 100
CARDINALITIES = list(range(20, 410, 20)) + [500, 1000, 2000, 5000]


def make_table(k=K, D=D, cardinalities=CARDINALITIES, iterations=ITERATIONS, n_jobs=1):
    stats = {key: np.zeros((D, len(cardinalities))) for key in ("dhat", "mu_nu", "mu_tau")}
    est = DANCo(k=k)
    est._k = k
    for d in range(1, D + 1):
        for j, n in enumerate(cardinalities):
            runs = [
                est._dancoDimEstNoCalibration(hyperBall(n, d, random_state=i).reshape(n, d), d * 2 + 5, n_jobs=n_jobs)
                for i in range(iterations)
            ]
            for key in stats:
                stats[key][d - 1, j] = np.nanmean([r[key] for r in runs])
    return stats


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "skdim/id/DANCoFit/danco_fit_table.npz"
    np.savez_compressed(out, **make_table(), cardinalities=np.array(CARDINALITIES), k=K, iterations=ITERATIONS)
