import numpy as np
import inspect
from scipy.cluster.hierarchy import dendrogram
import sklearn.cluster
from sklearn.base import ClusterMixin, BaseEstimator
from scipy.special import softmax


def __dir__():
    return ["getLinkage", "dendrogram", "AgglomerativeClustering", "CutPursuit"]


def CV(x, thresh=0.5):
    return max(0, np.std(x) - thresh) / np.mean(x)


def gaussw(d, sigma=1, axis=0):
    # yML =  gaussw(
    #    np.abs(y.flatten() - np.tile(np.arange(int(np.ceil(np.max(y))))[:,None],len(X))), sigma=1,
    # ).copy('F')
    """ turn distance d into probabilities"""
    return softmax((-(d ** 2)) / (sigma ** 2), axis=axis)


def getLinkage(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix


class AgglomerativeClustering(sklearn.cluster.AgglomerativeClustering):
    def __init__(
        self,
        n_clusters=None,
        distance_threshold=None,
        affinity="euclidean",
        memory=None,
        connectivity=None,
        compute_full_tree="auto",
        linkage="ward",
        compute_distances=False,
    ):

        kwargs = inspect.getargvalues(inspect.currentframe())[-1]
        kwargs.pop("self")
        super().__init__()
        # super().set_params(**{k:kwargs[k] for k in super().get_params()})
        for arg, val in kwargs.items():
            setattr(self, arg, val)

    def fit(self, lid):

        if len(lid.shape) == 1:
            lid = lid.reshape(-1, 1)
        if not (self.affinity in ["euclidean", "l1", "l2"]):
            raise ValueError("affinity must be one of 'euclidean', 'l1', 'l2'")
        if self.connectivity is None:
            print(
                "WARNING: connectivity matrix is None."
                + " Clustering will not take spatial relationships into account"
            )
        if (self.n_clusters is None) and (self.distance_threshold is None):
            self.distance_threshold = np.around(np.std(lid), 2)
            print(
                f"n_clusters and distance_threshold are both None."
                + " Setting distance_threshold as std(lid) =",
                round(self.distance_threshold, 2),
            )

        super().fit(lid)

        self.comp_mean_lid = np.array(
            [np.mean(lid[self.labels_ == c]) for c in np.unique(self.labels_)]
        )
        return self


class CutPursuit(ClusterMixin, BaseEstimator):
    """Cut-pursuit algorithms. Minimize functionals over a graph G = (V, E)

    connectivity: scipy sparse CSR matrix
        Graph adjacency matrix. Non-zero values will be used as edge weights if edge_weights='adj'.
    mode: str, default='l0'
        Which cut-pursuit algorithm to use:
            'l0': l0 (weighted contour length) penalization,  with a separable loss term. F(x) = f(x) + ||x||_l0
            'l1_lsx': l1 (weighted total variation) penalization, with a separable loss term and simplex constraints.  F(x) = f(x) + ||x||_l1 + i_{simplex}(x)
            'l1_gtv': l1 (weighted total variation) penalization, with quadratic loss term. F(x) = 1/2 ||y - x||^2 + ||x||_l1
    loss: int or float
        0 for linear
        1 for quadratic
        0 < loss < 1 for smoothed Kullback-Leibler
    edge_weights: str or float
        "conn" to take weights from connectivity matrix
        scalar for homogeneous weights
    vert_weights: (real) array of length V or empty for no weights
        Weights on vertices. 
        for multidimensional data, weighs the coordinates in the
        l1 norms of finite differences; all weights must be strictly positive,
        it is advised to normalize the weights so that the first value is unity
    cp_dif_tol: float
        stopping criterion on iterate evolution; algorithm stops if
        relative changes (in Euclidean norm) is less than dif_tol;
        1e-3 is a typical value; a lower one can give better precision
        but with longer computational time and more final components
    cp_it_max: int
        maximum number of iterations (graph cut and subproblem)
        10 cuts solve accurately most problems
    max_num_threads: if greater than zero, set the maximum number of threads
        used for parallelization with OpenMP
    balance_parallel_split: if true, the parallel workload of the split step 
        is balanced; WARNING: this might trade off speed against optimality
    l0_solver: dict
        Parameters for the l0 solver
            K: number of alternative values considered in the split step
            split_iter_num: number of partition-and-update iterations in the split 
                step
            split_damp_ratio: edge weights damping for favoring splitting; edge
                weights increase in linear progression along partition-and-update
                iterations, from this ratio up to original value; real scalar between 0
                and 1, the latter meaning no damping
            kmpp_init_num: number of random k-means initializations in the split step
            kmpp_iter_num: number of k-means iterations in the split step
    l1_solver: dict
        Parameters for the l1 solver
            pfdr_rho: relaxation parameter, 0 < rho < 2
                1 is a conservative value; 1.5 often speeds up convergence
            pfdr_cond_min: stability of preconditioning; 0 < cond_min < 1;
                corresponds roughly the minimum ratio to the maximum descent metric;
                1e-2 is typical; a smaller value might enhance preconditioning
            pfdr_dif_rcd: reconditioning criterion on iterate evolution;
                a reconditioning is performed if relative changes of the iterate drops
                below dif_rcd; WARNING: reconditioning might temporarily draw minimizer
                away from the solution set and give bad subproblem solutions
            pfdr_dif_tol: stopping criterion on iterate evolution; algorithm stops if
                relative changes (in Euclidean norm) is less than dif_tol
                1e-2*cp_dif_tol is a conservative value
            pfdr_it_max: maximum number of iterations
                1e4 iterations provides enough precision for most subproblems

    """

    # def __new__(cls):
    #    try:
    #        cls.cutpursuit = __import__('cutpursuit')
    #        return super(CutPursuit,cls).__new__(cls)
    #    except ModuleNotFoundError as err:
    #        print(f'{err}, install with: \n',
    #        'conda install gxx-compiler -y \n',
    #        'pip install git+https://github.com/j-bac/parallel-cut-pursuit')
    #        raise

    def _check_package(self):
        try:
            self.cutpursuit = __import__("cutpursuit")
        except ModuleNotFoundError as err:
            print(
                f"To use this class, install cutpursuit with: \n",
                "conda install gxx-compiler -y \n",
                "pip install git+https://github.com/j-bac/parallel-cut-pursuit",
            )
            raise

    def __init__(
        self,
        connectivity,
        mode="l0",
        loss=1,
        min_comp_weight=0.0,
        edge_weights=1.0,
        vert_weights=None,
        coor_weights=None,
        cp_dif_tol=0.0001,
        cp_it_max=10,
        max_num_threads=0,
        balance_parallel_split=True,
        l0_solver=dict(
            K=2,
            split_iter_num=2,
            split_damp_ratio=1.0,
            kmpp_init_num=3,
            kmpp_iter_num=3,
        ),
        l1_solver=dict(
            pfdr_rho=1.0,
            pfdr_cond_min=0.01,
            pfdr_dif_rcd=0.0,
            pfdr_dif_tol=None,
            pfdr_it_max=10000,
        ),
    ):
        self._check_package()
        kwargs = inspect.getargvalues(inspect.currentframe())[-1]
        kwargs.pop("self")

        for arg, val in kwargs.items():
            if arg == "edge_weights":
                if arg == "conn":
                    self.edge_weights = connectivity.data
                else:  # None or real number
                    self.edge_weights = edge_weights
            else:
                setattr(self, arg, val)

    def fit(self, X):
        """
        X: observations, (real) D-by-V array
        For Kullback-Leibler loss, the value at each vertex must lie on the probability simplex
        careful to the internal memory representation of multidimensional
        arrays; the C++ implementation uses column-major order (F-contiguous);
        usually numpy uses row-major order (C-contiguous), but this can often 
        be taken care of without actually copying data by using transpose();
        """

        if self.mode == "l0":
            self.components_, self.components_values_ = self.cutpursuit.cp_kmpp_d0_dist(
                loss=self.loss,
                Y=X,
                first_edge=self.connectivity.indptr.astype("uint32"),
                adj_vertices=self.connectivity.indices.astype("uint32"),
                edge_weights=self.edge_weights,
                vert_weights=self.vert_weights,
                coor_weights=self.coor_weights,
                cp_dif_tol=self.cp_dif_tol,
                cp_it_max=self.cp_it_max,
                min_comp_weight=self.min_comp_weight,
                max_num_threads=self.max_num_threads,
                balance_parallel_split=self.balance_parallel_split,
                **self.l0_solver,
            )

        if self.mode == "l1":
            self.components_, self.components_values_ = self.cutpursuit.cp_prox_tv(
                Y=X,
                first_edge=self.connectivity.indptr.astype("uint32"),
                adj_vertices=self.connectivity.indices.astype("uint32"),
                edge_weights=self.edge_weights,
                cp_dif_tol=self.cp_dif_tol,
                cp_it_max=self.cp_it_max,
                max_num_threads=self.max_num_threads,
                balance_parallel_split=self.balance_parallel_split,
                **self.l1_solver,
            )

        return self

    def get_confidence(self):
        pass
