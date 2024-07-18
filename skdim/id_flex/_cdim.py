import numpy as np
from .._commonfuncs import FlexNbhdEstimator
from sklearn.utils.parallel import Parallel, delayed
from joblib import effective_n_jobs


class CDim(FlexNbhdEstimator):
    """
    Conical dimension method with either knn or epsilon neighbourhood

    Yang et al.
    Conical dimension as an intrinsic dimension estimator and its applications
    https://doi.org/10.1137/1.9781611972771.16
    """

    def __init__(
        self,
        nbhd_type="knn",
        metric="euclidean",
        comb="mean",
        smooth=False,
        n_jobs=1,
        radius=1.0,
        n_neighbors=5,
    ):
        """
        nbhd_type: either 'knn' (k nearest neighbour) or 'eps' (eps nearest neighbour) or 'custom'
        metric: defaults to standard euclidean metric; if X a distance matrix, then set to 'precomputed'
        comb: method of averaging either 'mean', 'median', or 'hmean'
        smooth: if true, average over dimension estimates of local neighbourhoods using comb
        n_jobs: number of parallel processes in inferring local neighbourhood
        radius: radius parameter for nearest neighbour construction
        n_neighbors: number of neighbors for k nearest neighbourhood construction
        """

        super().__init__(
            pw_dim=True,
            nbhd_type=nbhd_type,
            pt_nbhd_incl_pt=False,
            metric=metric,
            comb=comb,
            smooth=smooth,
            n_jobs=n_jobs,
            radius=radius,
            n_neighbors=n_neighbors,
            sort_radial=False,
        )

    def _fit(self, X, nbhd_indices, radial_dists):

        if effective_n_jobs(self.n_jobs) > 1:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                # Asynchronously apply the `fit` function to each data point and collect the results
                results = parallel(
                    delayed(self._local_cdim)(X, idx, nbhd)
                    for idx, nbhd in enumerate(nbhd_indices)
                )
            self.dimension_pw_ = np.array(results)
        else:
            self.dimension_pw_ = np.array(
                [
                    self._local_cdim(X, idx, nbhd)
                    for idx, nbhd in enumerate(nbhd_indices)
                ]
            )

    @staticmethod
    def _local_cdim(X, idx, nbhd):
        if len(nbhd) == 0:
            return 0
        neighbors = np.take(X, nbhd, 0)
        vectors = neighbors - X[idx]
        norms = np.linalg.norm(vectors, axis=1)
        vectors = np.take(vectors, np.argsort(norms), axis=0)
        vectors = vectors * np.sign(np.tensordot(vectors, vectors[0], axes=1))[:, None]
        T = np.matmul(vectors, vectors.T)
        label_set = [[i] for i, _ in enumerate(nbhd)]
        de = 0
        while len(label_set) > 0:
            de += 1
            new_label_set = []
            for label in label_set:
                for i, _ in enumerate(nbhd):
                    if i in label:
                        continue
                    for vec_idx in label:
                        if T[i][vec_idx] >= 0:
                            break
                    else:
                        new_label = label + [i]
                        new_label_set.append(new_label)
            label_set = new_label_set
        return de
