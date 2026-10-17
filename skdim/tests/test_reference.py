# SPDX-License-Identifier: BSD-3-Clause
"""Agreement with reference implementations, via outputs stored in reference/fixtures.json.

Regenerate with: pixi run -e ref make-fixtures (R intrinsicDimension, ider, PCAtools, scran; DADApy).
Tolerances: integer ranks exact; closed forms rtol 1e-6; Gride rtol 1e-6 (DADApy bisects to 1e-7)."""
import json
import os
import sys

import numpy as np
import pytest
import skdim

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference")
sys.path.insert(0, HERE)
from data import EASY, id_datasets, rank_datasets  # noqa: E402

with open(os.path.join(HERE, "fixtures.json")) as f:
    FX = json.load(f)

ID = id_datasets()
RANK = rank_datasets()


@pytest.mark.parametrize("name", sorted(ID))
def test_intrinsicDimension(name):
    X, ref = ID[name], FX["intrinsicDimension"][name]
    assert skdim.id.lPCA(ver="FO").fit(X).dimension_ == ref["pca_FO"]
    assert skdim.id.lPCA(ver="Fan").fit(X).dimension_ == ref["pca_fan"]
    assert skdim.id.lPCA(ver="maxgap").fit(X).dimension_ == ref["pca_maxgap"]
    assert skdim.id.MLE().fit(X, n_neighbors=20).dimension_ == pytest.approx(ref["mle_k20"], rel=1e-6)
    assert skdim.id.MLE(unbiased=True).fit(X, n_neighbors=20).dimension_ == pytest.approx(ref["mle_k20_unbiased"], rel=1e-6)
    assert skdim.id.ESS(ver="a").fit_once(X[:30]).dimension_ == pytest.approx(ref["ess_a_n30"], rel=1e-6)
    assert skdim.id.ESS(ver="b").fit_once(X[:30]).dimension_ == pytest.approx(ref["ess_b_n30"], rel=1e-6)
    # R's labels are swapped relative to Rozza et al. (2012), where MLi is the integer estimate:
    # R's MIND_MLi returns the fractional and MIND_MLk the integer estimate
    assert skdim.id.MiND_ML(ver="MLk", k=20, D=10).fit(X).dimension_ == pytest.approx(ref["mind_mli_k20_D10"], rel=1e-6)
    assert skdim.id.MiND_ML(ver="MLi", k=20, D=10).fit(X).dimension_ == pytest.approx(ref["mind_mlk_k20_D10"], rel=1e-6)


@pytest.mark.parametrize("name", sorted(ID))
def test_ider(name):
    X, ref = ID[name], FX["ider"][name]
    assert skdim.id.CorrInt(k1=10, k2=20).fit(X).dimension_ == pytest.approx(ref["corint_k10_k20"], rel=1e-6)
    np.testing.assert_allclose(skdim.id.MADA().fit(X, n_neighbors=20).dimension_pw_, ref["mada_local_k20"], rtol=1e-6)


@pytest.mark.parametrize("name", sorted(RANK))
def test_PCAtools_scran(name):
    X, pt, sc = RANK[name], FX["PCAtools"][name], FX["scran"][name]
    assert skdim.id.lPCA(ver="MP", noise_var=1.0).fit(X).dimension_ == pt["MP_noise1"]
    assert skdim.id.lPCA(ver="GD", noise_var=1.0).fit(X).dimension_ == pt["GD_noise1"]
    assert skdim.id.lPCA(ver="denoised", noise_var=1.0).fit(X).dimension_ == sc["denoised_noise1"]
    if name not in EASY:
        return
    # different RNGs: equal only because the spikes are far from the null
    pa = skdim.id.lPCA(ver="parallel", n_permutations=100, quantile=0.95, random_state=0).fit(X)
    assert pa.dimension_ == pt["parallel_niters100_thr0.05"]


@pytest.mark.parametrize("name", sorted(ID))
def test_dadapy(name):
    X, ref = ID[name], FX["dadapy"][name]
    assert skdim.id.TwoNN(discard_fraction=0.1).fit(X).dimension_ == pytest.approx(ref["twonn_base_mu0.9"], rel=1e-6)
    assert skdim.id.TwoNN(method="mle", discard_fraction=0.0).fit(X).dimension_ == pytest.approx(ref["twonn_ml"], rel=1e-6)
    g = skdim.id.GRIDE(multiscale=True, range_max=64).fit(X)
    np.testing.assert_allclose(g.dimension_scales_, ref["gride_range64_ids"], rtol=1e-6)
    np.testing.assert_allclose(g.error_scales_, ref["gride_range64_errs"], rtol=1e-6)
