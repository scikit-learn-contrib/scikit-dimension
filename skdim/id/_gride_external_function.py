def _compute_id_gride_multiscale(mus, d0, d1, eps):
    """Compute the id using the gride algorithm.

    Helper of return return_id_gride.

    Args:
        mus (np.ndarray(float)): ratio of the distances of nth and 2nth nearest neighbours (Ndata x log2(range_max))
        d0 (float): minimum intrinsic dimension considered in the search;
        d1 (float): maximum intrinsic dimension considered in the search;
        eps (float): precision of the approximate id calculation.

    Returns:
        intrinsic_dim (np.ndarray(float): array of id estimates
        intrinsic_dim_err (np.ndarray(float): array of error estimates
    """
    # array of ids (as a function of the average distance to a point)
    ids_scaling = np.zeros(mus.shape[1])
    # array of error estimates (via fisher information)
    ids_scaling_err = np.zeros(mus.shape[1])

    for i in range(mus.shape[1]):
        n1 = 2**i

        intrinsic_dim, id_error = self._compute_id_gride_single_scale(
            d0, d1, mus[:, i], n1, 2 * n1, eps
        )

        ids_scaling[i] = intrinsic_dim
        ids_scaling_err[i] = id_error

    return ids_scaling, ids_scaling_err

def _compute_id_gride_single_scale(self, d0, d1, mus, n1, n2, eps):
    id_ = ut._argmax_loglik(
        self.dtype, d0, d1, mus, n1, n2, eps=eps
    )  # eps=precision id calculation
    id_err = (
        1
        / ut._fisher_info_scaling(
            id_, mus, n1, 2 * n1, eps=5 * self.eps
        )  # eps=regularization small numbers
    ) ** 0.5

    return id_, id_err