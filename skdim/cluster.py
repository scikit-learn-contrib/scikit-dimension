import numpy as np
import inspect
from scipy.cluster.hierarchy import dendrogram
import sklearn.cluster
from sklearn.base import ClusterMixin, BaseEstimator

def __dir__(): return ['getLinkage','dendrogram','AgglomerativeClustering','CutPursuit']
    
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
    def __init__(self,
                 n_clusters=None,
                 distance_threshold=None, 
                 affinity="euclidean",
                 memory=None,
                 connectivity=None, 
                 compute_full_tree='auto',
                 linkage='ward', 
                 compute_distances=False):
        
        kwargs = inspect.getargvalues(inspect.currentframe())[-1]
        kwargs.pop("self")        
        super().__init__()
        #super().set_params(**{k:kwargs[k] for k in super().get_params()})
        for arg, val in kwargs.items():
            setattr(self, arg, val)
            
    def fit(self,lid):
        
        if len(lid.shape) == 1:
            lid = lid.reshape(-1,1)
        if not(self.affinity in ['euclidean', 'l1', 'l2']):
            raise ValueError("affinity must be one of 'euclidean', 'l1', 'l2'")
        if self.connectivity is None:
            print("WARNING: connectivity matrix is None."+
                  " Clustering will not take spatial relationships into account")   
        if (self.n_clusters is None) and (self.distance_threshold is None):
            self.distance_threshold = np.around(np.std(lid),2)
            print(f"n_clusters and distance_threshold are both None."+
            " Setting distance_threshold as std(lid) =",round(self.distance_threshold,2)) 
        
        super().fit(lid)
        
        self.comp_mean_lid = np.array([np.mean(lid[self.labels_==c]) for c in np.unique(self.labels_)])
        return self    

class CutPursuit(ClusterMixin,BaseEstimator):
    #def __new__(cls):
    #    try: 
    #        cls.cutpursuit = __import__('cutpursuit')
    #        return super(CutPursuit,cls).__new__(cls)
    #    except ModuleNotFoundError as err:
    #        print(f'{err}, install with: \n',
    #        'conda install gxx-compiler -y \n',
    #        'pip install git+https://github.com/j-bac/parallel-cut-pursuit')
    #        raise

    def __init__(self):
        try: 
            self.cutpursuit = __import__('cutpursuit')
        except ModuleNotFoundError as err:
            print(f'To use this class, install cutpursuit with: \n',
            'conda install gxx-compiler -y \n',
            'pip install git+https://github.com/j-bac/parallel-cut-pursuit')
            raise
