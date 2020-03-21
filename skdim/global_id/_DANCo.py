#
# MIT License
#
# Copyright (c) 2020 Jonathan Bac
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

# Credits to Kerstin Johnsson
# https://cran.r-project.org/web/packages/intrinsicDimension/index.html
# for the R implementation


from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_random_state

import sys
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0, i1, digamma
from scipy.interpolate import interp1d
from .._commonfuncs import binom_coeff, get_nn, randball, lens, indnComb


class DANCo(BaseEstimator):

    """ Intrinsic dimension estimation with the DANCo (Ceruti et al. 2012), MIND_MLi and MIND_MLk (Rozza et al. 2012) methods. 

    ----------
    Attributes

    k :	
        Neighborhood parameter.
    D :	
        Maximal dimension
    ver : str, default='DANCo'
        Version to use. possible values: 'DANCo', 'MIND_MLi', 'MIND_MLk'.
    calibration_data : dict, default=None
        Precomputed calibration data. 
    fractal : bool, default=True
        Whether to return fractal rather than integer dimension
    verbose : bool, default=False

    ----------
    Returns

    dimension_ : int or float
        The estimated intrinsic dimension
    kl_divergence : float
        The KL divergence between data and reference data for the estimated dimension (if ver == 'DANCo').
    calibration_data : dict
        Calibration data that can be reused when applying DANCo to data sets of the same size with the same neighborhood parameter k.

    ----------
    References

    Ceruti, C. et al. (2012) DANCo: Dimensionality from Angle and Norm Concentration. arXiv preprint 1206.3881.

    Rozza, A et al. (2012) Novel high intrinsic dimensionality estimators. Machine learning 89, 37-65. 
    """

    def __init__(self, k=10, D=10, calibration_data=None, ver='DANCo', fractal=True, verbose=False, random_state=None):
        self.k = k
        self.D = D
        self.calibration_data = calibration_data
        self.ver = ver
        self.verbose = verbose
        self.fractal = fractal
        self.random_state = random_state

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A data set for which the intrinsic dimension is estimated.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=False)
        if len(X) == 1:
            raise ValueError("Can't fit with 1 sample")
        if X.shape[1] == 1:
            raise ValueError("Can't fit with n_features = 1")
        if not np.isfinite(X).all():
            raise ValueError("X contains inf or NaN")

        if self.k >= len(X):
            warnings.warn(
                'k larger or equal to len(X), using len(X)-2')
            self._k = len(X)-2
        else:
            self._k = self.k

        self.random_state_ = check_random_state(self.random_state)

        self.dimension_, self.kl_divergence_, self.calibration_data_ = self._dancoDimEst(
            X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _KL(self, nocal, caldat):
        kld = self._KLd(nocal['dhat'], caldat['dhat'])
        klnutau = self._KLnutau(nocal['mu_nu'], caldat['mu_nu'],
                                nocal['mu_tau'], caldat['mu_tau'])
        # print(klnutau)
        return(kld + klnutau)

    def _KLd(self, dhat, dcal):
        H_k = np.sum(1/np.arange(1, self._k+1))
        quo = dcal/dhat
        a = np.power(-1, np.arange(self._k+1))*np.array(list(binom_coeff(self._k, i)
                                                             for i in range(self._k+1)))*digamma(1 + np.arange(self._k+1)/quo)
        return(H_k*quo - np.log(quo) - (self._k-1)*np.sum(a))

    @staticmethod
    def _KLnutau(nu1, nu2, tau1, tau2):
        return(np.log(min(sys.float_info.max, i0(tau2))/min(sys.float_info.max, i0(tau1))) +
               min(sys.float_info.max, i1(tau1))/min(sys.float_info.max, i0(tau1))*(tau1 - tau2*np.cos(nu1-nu2)))

    def _nlld(self, d, rhos, N):
        return(-self._lld(d, rhos, N))

    def _lld(self, d, rhos, N):
        if (d == 0):
            return(np.array([-1e30]))
        else:
            return N*np.log(self._k*d) + (d-1)*np.sum(np.log(rhos)) + (self._k-1)*np.sum(np.log(1-rhos**d))

    def _nlld_gr(self, d, rhos, N):
        if (d == 0):
            return(np.array([-1e30]))
        else:
            return -(N/d + np.sum(np.log(rhos) - (self._k-1)*(rhos**d)*np.log(rhos)/(1 - rhos**d)))

    def _MIND_MLk(self, rhos, D):
        N = len(rhos)
        d_lik = np.array([np.nan]*D)
        for d in range(D):
            d_lik[d] = self._lld(d, rhos, N)
        return(np.argmax(d_lik))

    def _MIND_MLi(self, rhos, D, dinit):
        res = minimize(fun=self._nlld,
                       x0=np.array([dinit]),
                       jac=self._nlld_gr,
                       args=(rhos, len(rhos)),
                       method='L-BFGS-B',
                       bounds=[(0, D)])
        return(res['x'])

    def _MIND_MLx(self, X, D):
        nbh_data, idx = get_nn(X, self._k+1)
        rhos = nbh_data[:, 0]/nbh_data[:, -1]

        d_MIND_MLk = self._MIND_MLk(rhos, D)
        if (self.ver == 'MIND_MLk'):
            return(d_MIND_MLk)

        d_MIND_MLi = self._MIND_MLi(rhos, D, d_MIND_MLk)
        if (self.ver == 'MIND_MLi'):
            return(d_MIND_MLi)
        else:
            raise ValueError("Unknown version: ", self.ver)

    @staticmethod
    def _Ainv(eta):
        if (eta < .53):
            return(2*eta + eta**3 + 5*(eta**5)/6)
        elif (eta < .85):
            return(-.4 + 1.39*eta + .43/(1-eta))
        else:
            return(1/((eta**3)-4*(eta**2)+3*eta))

    @staticmethod
    def _loc_angles(pt, nbs):
        vec = nbs-pt
       # if(len(pt) == 1):
       #     vec = vec.T
        vec_len = lens(vec)
        combs = indnComb(len(nbs), 2).T
        sc_prod = np.sum(vec[combs[0, :]]*vec[combs[1, :]], axis=1)
        # if (length(pt) == 1) {
        # print(sc.prod)
        #print((vec.len[combs[1, ]]*vec.len[combs[2, ]]))
        # }
        cos_th = sc_prod/(vec_len[combs[0, :]]*vec_len[combs[1, :]])
        if (any(abs(cos_th) > 1)):
            print(cos_th[np.abs(cos_th) > 1])
        return(np.arccos(cos_th))

    def _angles(self, X, nbs):
        N = len(X)

        thetas = np.zeros((N, binom_coeff(self._k, 2)))
        for i in range(N):
            nb_data = X[nbs[i, ], ]
            thetas[i, ] = self._loc_angles(X[i, ], nb_data)
        return(thetas)

    def _ML_VM(self, thetas):
        sinth = np.sin(thetas)
        costh = np.cos(thetas)
        nu = np.arctan(np.sum(sinth)/np.sum(costh))
        eta = np.sqrt(np.mean(costh)**2 + np.mean(sinth)**2)
        tau = self._Ainv(eta)
        return dict(nu=nu, tau=tau)

    def _dancoDimEstNoCalibration(self, X, D, n_jobs=1):
        nbh_data, idx = get_nn(X, self._k+1, n_jobs=n_jobs)
        rhos = nbh_data[:, 0]/nbh_data[:, -1]
        d_MIND_MLk = self._MIND_MLk(rhos, D)
        d_MIND_MLi = self._MIND_MLi(rhos, D, d_MIND_MLk)

        thetas = self._angles(X, idx[:, :self._k])
        ml_vm = list(map(self._ML_VM, thetas))
        mu_nu = np.mean([i['nu'] for i in ml_vm])
        mu_tau = np.mean([i['tau'] for i in ml_vm])
        if(X.shape[1] == 1):
            mu_tau = 1

        return dict(dhat=d_MIND_MLi, mu_nu=mu_nu, mu_tau=mu_tau)

    def _DancoCalibrationData(self, N):
        me = dict(k=self._k,
                  N=N,
                  calibration_data=list(),
                  maxdim=0)
        return(me)

    def _increaseMaxDimByOne(self, dancoCalDat):
        newdim = dancoCalDat['maxdim'] + 1
        MIND_MLx_maxdim = newdim*2+5
        dancoCalDat['calibration_data'].append(self._dancoDimEstNoCalibration(randball(dancoCalDat['N'], newdim, 1, center=[0]*newdim, random_state=self.random_state_),
                                                                              dancoCalDat['k'],
                                                                              MIND_MLx_maxdim))
        dancoCalDat['maxdim'] = newdim
        return(dancoCalDat)

#    @staticmethod
#    def increaseMaxDimByOne_precomputedSpline(dancoCalDat,DANCo_splines):
#        newdim = dancoCalDat['maxdim'] + 1
#        dancoCalDat['calibration_data'].append({'dhat':DANCo_splines['spline_dhat'](newdim,dancoCalDat['N']),
#                                                'mu_nu':DANCo_splines['spline_mu'](newdim,dancoCalDat['N']),
#                                                'mu_tau':DANCo_splines['spline_tau'](newdim,dancoCalDat['N'])})
#        dancoCalDat['maxdim'] = newdim
#        return(dancoCalDat)

    def _computeDANCoCalibrationData(self, N):
        print('Computing calibration X...\nCurrent dimension: ', end=' ')
        cal = self._DancoCalibrationData(self._k, N)
        while (cal['maxdim'] < self.D):
            if cal['maxdim'] % 10 == 0:
                print(cal['maxdim'], end=' ')
            cal = self._increaseMaxDimByOne(cal)
        return cal

    def _dancoDimEst(self, X):

        cal = self.calibration_data
        N = len(X)

        if cal is not None:
            if (cal['k'] != self._k):
                raise ValueError("Neighborhood parameter k = %s does not agree with neighborhood parameter of calibration data, cal$k = %s",
                                 self._k, cal['k'])
            if (cal['N'] != N):
                raise ValueError("Number of data points N = %s does not agree with number of data points of calibration data, cal$N = %s",
                                 N, cal['N'])

        if self.ver not in ['DANCo', 'DANCoFit']:
            return(self._MIND_MLx(X))

        nocal = self._dancoDimEstNoCalibration(X, self.D)
        if any(np.isnan(val) for val in nocal.values()):
            return dict(de=np.nan, kl_divergence=np.nan, calibration_data=cal)

        if (cal is None):
            cal = self._DancoCalibrationData(N)

        if (cal['maxdim'] < self.D):

            if self.ver == 'DANCoFit':
                if self.verbose:
                    print("Generating DANCo calibration data from precomputed spline interpolation for cardinality 50 to 5000, k = 10, dimensions 1 to 100")
                raise ValueError('DANCoFit not yet implemented')
                # load precomputed splines as a function of dimension and dataset cardinality
                #DANCo_splines = {}
                # for spl in ['spline_dhat','spline_mu','spline_tau']:
                #    with open(path_to_estimators+'/DANCoFit/DANCo_'+spl+'.pkl', 'rb') as f:
                #        DANCo_splines[spl]=pickle.load(f)
                # compute interpolated statistics
                # while (cal['maxdim'] < D):
                #    cal = self.increaseMaxDimByOne_precomputedSpline(cal,DANCo_splines)

            else:
                if self.verbose:
                    print("Computing DANCo calibration data for N = {}, k = {} for dimensions {} to {}".format(
                        N, self._k, cal['maxdim']+1, self.D))

                # compute statistics
                while (cal['maxdim'] < self.D):
                    cal = self._increaseMaxDimByOne(cal)

        kl = np.array([np.nan]*self.D)
        for d in range(self.D):
            kl[d] = self._KL(nocal, cal['calibration_data'][d])

        de = np.argmin(kl)+1

        if self.fractal:
            # Fitting with a cubic smoothing spline:
            f = interp1d(np.arange(1, self.D+1), kl, kind='cubic',
                         bounds_error=False, fill_value=(1, self.D+1))
            # Locating the minima:
            de_fractal = minimize(
                f, de, bounds=[(1, self.D+1)], tol=1e-3)['x']
            return de_fractal[0], kl[de-1], cal
        else:
            return de[0], kl[de-1], cal
