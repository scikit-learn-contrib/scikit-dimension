# Authors

Maintainer: Jonathan Bac ([@j-bac](https://github.com/j-bac))

## Contributors

In alphabetical order, from the git history:

- Jackson Burns
- JesseCresswell
- Rukun Dou
- karthikviswanathn
- Nikolar1
- Jayesh Suryavanshi
- Andrei Zinovyev

### From the kmyim/scikit-dimension fork

Code ported from https://github.com/kmyim/scikit-dimension, accompanying
Binnie, Dłotko, Harvey, Malinowski, Yim, "A Survey of Dimension Estimation Methods"
(arXiv:2507.13887). Git authorship of the original files is preserved.

- Ka Man Yim ([@kmyim](https://github.com/kmyim)): `id.PH`, Laplace-evidence code behind `lPCA(ver="Minka")`, datasets, embeddings
- John Harvey ([@johnharveymath](https://github.com/johnharveymath)): datasets, embeddings, `lPCA(ver="Minka")`
- Jakub Malinowski: GRIDE reference-test design (`id.GRIDE` itself is reimplemented from Denti et al. 2022)

## Original implementations

scikit-dimension builds on the following implementations. Licences and notices are in
`LICENSES_bundled.txt`; method papers are cited in each estimator's docstring.

| Code in scikit-dimension | Original implementation | Authors | Licence |
|---|---|---|---|
| `DANCo`, `ESS`, `KNN`, `MLE`, `MiND_ML`, `lPCA` (FO, Fan, maxgap, ratio), `hyperTwinPeaks`, `swissRoll3Sph` | R package [intrinsicDimension](https://cran.r-project.org/package=intrinsicDimension) | Kerstin Johnsson | MIT |
| `CorrInt`, `MADA`, `lineDiskBall` | R package [ider](https://cran.r-project.org/package=ider) | Hideitsu Hino | GPL-2 (BSD-3 with permission) |
| `TLE`, `MOM` | MATLAB [TLE scripts](https://github.com/radacha/tle) | Laurent Amsaleg, Oussama Chelly, Michael E. Houle, Ken-ichi Kawarabayashi, Miloš Radovanović, Weeris Treeratanajaru | CC BY-SA 4.0 (BSD-3 with permission) |
| `TwoNN` | Python [TWO-NN](https://github.com/fmottes/TWO-NN) | Francesco Mottes | MIT |
| `FisherS` | [FisherSeparabilityAnalysis](https://github.com/auranic/FisherSeparabilityAnalysis) | Andrei Zinovyev, Evgeny Mirkes, Luca Albergante, Jonathan Bac | none stated (BSD-3 with permission) |
| `lPCA(ver="scRMT")` | Python [randomly](https://github.com/RabadanLab/randomly) | Luis Aparicio, Mykola Bordyuh | MIT |
| `BenchmarkManifolds` | [GeoMLE](https://github.com/stat-ml/GeoMLE) | Mokrov Nikita, Marina Gomtsyan, Maxim Panov, Yury Yanovich | MIT |
| `lorenz_attractor` | [scipython-maths](https://github.com/scipython/scipython-maths) | Christian Hill | MIT |
| `PH`, `lPCA(ver="Minka")`, fork datasets, `embeddings` | [kmyim/scikit-dimension](https://github.com/kmyim/scikit-dimension) | Ka Man Yim, John Harvey | BSD-3 |

Reimplemented from the papers, with these implementations used as references (no code taken):
DADApy (Glielmo et al. 2022; `GRIDE`, `TwoNN(method="mle")`), Gabriele Lombardi's MATLAB
toolbox (`DANCo`, `MiND_ML`), Elena Facco's TWO-NN (`TwoNN`), and PCAtools and scran
(`lPCA` rank versions; only used to generate test fixtures).
