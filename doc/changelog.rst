.. role:: small

Changelog
=========

Version 0.4.1 :small:`October 2026`
-----------------------------------
- Documentation: fix the build on Read the Docs (source links failed when skdim is installed
  non-editably); the CI documentation job now installs skdim the same way.
- Remove invalid escape sequences from ``ESS`` docstrings (``SyntaxWarning`` on recent Python).
  No change to results.

Version 0.4.0 :small:`October 2026`
-----------------------------------

New estimators and options
^^^^^^^^^^^^^^^^^^^^^^^^^^
- ``id.GRIDE`` (Denti et al. 2022), reimplemented from the paper with DADApy's conventions.
- ``TwoNN(method="mle")``: closed-form unbiased maximum-likelihood estimate.
- ``id.PH``: persistent-homology (PH0) dimension, ported from the kmyim/scikit-dimension fork (Ka Man Yim).
- ``lPCA`` signal-rank versions:

  - random matrix theory: ``MP`` (Marchenko-Pastur edge), ``GD`` (Gavish-Donoho threshold),
    ``KN`` (Kritchman-Nadler sequential Tracy-Widom test);
  - ``parallel``: Horn's parallel analysis with a permutation or sign-flip null;
  - ``Minka``: PPCA Laplace evidence, ported from the fork (Ka Man Yim, John Harvey);
  - ``denoised``: scran's ``denoisePCANumber`` rule (technical variance from ``noise_var``).

- ``DANCo(ver="DANCoFit")``: calibration from a precomputed table (k=10, D <= 100,
  20 <= n_samples <= 5000), 10-60 times faster than simulated calibration.
- Datasets ``product``, ``hyperTwinPeaksUniform``, ``toroidal_spiral``, ``torus``, ``lorenz_attractor``,
  ``dumbbell``, ``SurfaceOfRevolution``, ``hyperSphere(radius, center)``, and ``skdim.embeddings``
  (``delay_embedding``, ``monomial_embedding``), ported from the fork (John Harvey, Ka Man Yim).

Changes to existing results
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Every pre-existing estimator and option was run on master (0.3.x) and on this version with the
same inputs; only the following differ.

- ``lPCA(ver="broken_stick")``: off-by-one fixed; estimates decrease by 1 (and can be 0 when no
  component exceeds the broken-stick expectation).
- ``FisherS``: inseparability probabilities are divided by n - 1 (the other points), as in the
  original MATLAB code, instead of n; estimates change by about 0.1-0.2% at n = 300, more for small n.
- ``TLE``: degenerate neighbourhoods now follow the original MATLAB code: NaN measurements from
  duplicate points are dropped, distances below ``epsilon`` are excluded correctly (they previously
  discarded all distances), and the estimate is 0 when (almost) all kNN distances are equal.
  Results on data without duplicate or tied points are unchanged.

API and compatibility
^^^^^^^^^^^^^^^^^^^^^
- scikit-learn >= 1.6 (``__sklearn_tags__``, ``n_features_in_``), Python 3.13+ and numpy 2.x.
- Fit-time attributes ``n_neighbors`` and ``comb`` of local estimators renamed ``n_neighbors_`` and ``comb_``.
- ``CorrInt`` and ``MLE`` no longer overwrite ``k1``/``k2``/``K`` during fit; ``KNN`` gains ``random_state``
  (it previously used numpy's global random generator).
- ``lPCA(ver="Fan")`` no longer modifies a user-supplied explained-variance array.
- ``transform(X)`` raises when ``X`` has a different number of features than the fitted data.
- pandas is imported only by ``datasets.BenchmarkManifolds``.
- ``datasets.hyperSphere`` gains ``radius`` and ``center`` after ``random_state``, so existing positional
  calls are unchanged; all pre-existing dataset generators return identical samples for a given seed.
- Requires Python >= 3.10.

Packaging, CI and documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Packaging metadata moves from ``setup.py``/``setup.cfg`` to ``pyproject.toml``; the README is the PyPI description.
- GitHub Actions replace Travis, AppVeyor and CircleCI: tests on Python 3.10-3.13 (Linux) and 3.13
  (macOS, Windows), a documentation build and a package build check on every pull request; releases
  are published to PyPI from GitHub releases with trusted publishing.
- Documentation uses the pydata-sphinx-theme, builds with Sphinx >= 7, and gains an estimator overview
  and this changelog; outdated pinned documentation dependencies are removed.
- ``pixi.toml`` provides development, documentation and reference-implementation environments.

Validation
^^^^^^^^^^
- ``skdim/tests/test_reference.py`` checks agreement with reference implementations on identical
  inputs: R intrinsicDimension (lPCA FO/Fan/maxgap, MLE and its options, ESS, MiND_ML, DANCo given
  shared calibration data), R ider (CorrInt, MADA), PCAtools (MP, GD, parallel), scran (denoised),
  DADApy (TwoNN, GRIDE), and the original MATLAB code run in Octave (TLE, MOM, FisherS).
  Fixtures are regenerated with ``pixi run -e ref make-fixtures``.
- No runnable reference: KNN (random resampling), KN, PH, lPCA ratio/participation_ratio/Kaiser/broken_stick.

Licensing and credits
^^^^^^^^^^^^^^^^^^^^^
- Per-file licence headers shortened to SPDX identifiers; third-party notices in ``LICENSES_bundled.txt``;
  original implementations and their authors listed in ``AUTHORS.md``; ``CITATION.cff`` added.

Version 0.3 :small:`April 2021`
-------------------------------
API change: predict methods changed to transform.

Version 0.2 :small:`December 2020`
----------------------------------
First release of skdim.
