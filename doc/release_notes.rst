.. role:: small
.. role:: smaller

Release Notes
=============

Version 0.4 (unreleased, branch dev-0.4)
----------------------------------------
New estimators and options:

- ``id.GRIDE`` (Denti et al. 2022) and ``TwoNN(method="mle")``.
- ``id.PH``: persistent-homology (PH0) dimension, ported from the kmyim/scikit-dimension fork (Ka Man Yim).
- ``lPCA`` signal-rank versions ``MP``, ``GD``, ``KN``, ``parallel`` and ``Minka`` (Minka ported from the fork, Ka Man Yim and John Harvey).
- Datasets ``product``, ``hyperTwinPeaksUniform``, ``toroidal_spiral``, ``torus``, ``lorenz_attractor``, ``dumbbell``,
  ``SurfaceOfRevolution``, ``hyperSphere(radius, center)``, and ``skdim.embeddings`` (John Harvey, Ka Man Yim).

Behaviour changes:

- ``lPCA(ver="broken_stick")`` off-by-one fixed: estimates can decrease by 1.
- ``lPCA(ver="Fan")`` no longer modifies a user-supplied explained-variance array.
- ``ver="DANCoFit"`` removed (it only raised an error).
- Fit-time attributes ``n_neighbors`` and ``comb`` of local estimators renamed ``n_neighbors_`` and ``comb_``.
- ``CorrInt`` and ``MLE`` no longer overwrite ``k1``/``k2``/``K``; ``KNN`` gains ``random_state``.
- Compatible with scikit-learn >= 1.6 (``__sklearn_tags__``, ``n_features_in_``).
- Per-file licence headers shortened to SPDX identifiers; third-party notices in ``LICENSES_bundled.txt``.

Version 0.3 :small:`April 2021`
-----------------------------------
API change: predict methods changed to transform

Version 0.2 :small:`December 2020`
-----------------------------------
First release of skdim.