#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# from skdim.id._DANCo import DANCo
# from skdim.datasets import hyperBall
# from scipy.interpolate import interp2d
# import pickle
# import plotly.graph_objs as go
# from plotly.subplots import make_subplots
#
# def DANCoTrain(k=10,D=100,
#               cardinalities=list(range(20,410,20))+[500,1000,2000,5000],
#               iterations=100,
#               n_cpus=1,
#               save_output_path=''):
#    ''' Pre-compute splines to approximate DANCo statistics as a function of
#        dimension D and data cardinalities (DANCoFit)'''
#
#    N=len(cardinalities)
#
#    dHat = np.zeros((D,N))
#    mu = np.zeros((D,N))
#    tau = np.zeros((D,N))
#
#    for d in range(1,D+1):
#        for n in range(N):
#            # Initializing the data of this iteration:
#            dhats = []
#            nus = []
#            taus = []
#
#            danco = DANCo(k=k,D=d)
#            danco._k = k
#
#            if d >=10 and n >=1000:
#                n_jobs=n_cpus
#            else:
#                n_jobs=1
#
#            # Printing to the user:
#            print('(d = {} \tn = {})...'.format(d,cardinalities[n]))
#            # Executing the experiments:
#            for i in range(iterations):
#
#                # Generating the dataset:
#                data = skdim.datasets.hyperBall(cardinalities[n],d,radius=1)
#                if d==1: data = data.reshape(-1,1)
#
#                # Estimating the parameters:
#                res = danco._dancoDimEstNoCalibration(data, d*2+5,n_jobs=n_jobs)
#                dhats.append(res['dhat'])
#                nus.append(res['mu_nu'])
#                taus.append(res['mu_tau'])
#
#            # Storing the results:
#            dHat[d-1,n] = np.nanmean(dhats)
#            mu[d-1,n] = np.nanmean(nus)
#            tau[d-1,n] = np.nanmean(taus)
#
#    #fit splines
#    kind='cubic'
#
#    spline_dhat=interp2d(x=np.repeat(np.arange(1,D+1),N),
#           y=np.tile(cardinalities,D),
#           z=dHat.ravel(),
#           kind=kind)
#
#    spline_mu=interp2d(x=np.repeat(np.arange(1,D+1),N),
#           y=np.tile(cardinalities,D),
#           z=mu.ravel(),
#           kind=kind)
#
#    spline_tau=interp2d(x=np.repeat(np.arange(1,D+1),N),
#           y=np.tile(cardinalities,D),
#           z=tau.ravel(),
#           kind=kind)
#
#    DANCo_splines = {}
#    for spl in ['spline_dhat','spline_mu','spline_tau']:
#        DANCo_splines[spl]=vars()[spl]
#
#    if type(save_output_path) == str:
#        np.savetxt(save_output_path+'dhat',dHat)
#        np.savetxt(save_output_path+'mu',mu)
#        np.savetxt(save_output_path+'tau',tau)
#
#        for spl in ['spline_dhat','spline_mu','spline_tau']:
#            with open(save_output_path+'DANCo_'+spl+'.pkl', 'wb') as f:
#                pickle.dump(vars()[spl], f)
#
#        #loading and returning saved splines to make sure everything is ok
#        DANCo_splines = {}
#        for spl in ['spline_dhat','spline_mu','spline_tau']:
#            with open(save_output_path+'DANCo_'+spl+'.pkl', 'rb') as f:
#                DANCo_splines[spl]=pickle.load(f)
#
#        #alternative way to save splines as explicit parameters
#        #from scipy.interpolate import BivariateSpline
#        #tck = DANCo_splines['spline_dhat'].tck
#        #spl2 = BivariateSpline._from_tck(tck)
#
#        return dHat,mu,tau,DANCo_splines
#
#    return dHat,mu,tau,DANCo_splines
#
# def load_DANCoTrain_results(path=''):
#
#    dHat = np.loadtxt(path+'dhat')
#    mu = np.loadtxt(path+'mu')
#    tau = np.loadtxt(path+'tau')
#
#    DANCo_splines = {}
#    for spl in ['spline_dhat','spline_mu','spline_tau']:
#        with open(path+'DANCo_'+spl+'.pkl', 'rb') as f:
#            DANCo_splines[spl]=pickle.load(f)
#
#    return dHat,mu,tau,DANCo_splines
#
# def plot_DANCoTrain(cardinalities,dHat,mu,tau,DANCo_splines):
#
#    fig = make_subplots(
#        rows=1, cols=3,
#        subplot_titles=['dHat','mu','tau'],
#        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'},{'type': 'scatter3d'}]]
#    )
#
#    N=len(cardinalities)
#    DANCostats=[dHat,mu,tau]
#    for i in range(1,4):
#        interp=DANCo_splines[list(DANCo_splines.keys())[i-1]](x=np.arange(1,len(dHat)),y=cardinalities)
#
#        fig.add_trace(go.Scatter3d(x=np.repeat(np.arange(1,len(dHat)),N),
#                                   y=np.tile(cardinalities,len(dHat)),
#                                   z=DANCostats[i-1].ravel(),
#                               marker=dict(size=1),
#                               mode='markers'),
#                      row=1,col=i)
#
#        fig.add_trace(go.Surface(x=np.arange(1,len(dHat)),
#                                   y=cardinalities,
#                                   z=interp),
#                      row=1,col=i)
#
#
#    fig.update_layout(
#        title_text='',
#        height=400,
#        width=1000)
#    fig.show()
