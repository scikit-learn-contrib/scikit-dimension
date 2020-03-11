import multiprocessing
import pickle
#import plotly.graph_objs as go
#from plotly.offline import init_notebook_mode
#from plotly.subplots import make_subplots
#init_notebook_mode(connected=False)

def DANCoTrain(k=10,D=100,
               cardinalities=list(range(20,410,20))+[500,1000,2000,5000],
               iterations=100,
               n_cpus=1,
               save_output_path=''):
    ''' Pre-compute splines to approximate DANCo statistics as a function of
        dimension D and data cardinalities (DANCoFit)'''
    
    N=len(cardinalities)

    dHat = np.zeros((D,N))
    mu = np.zeros((D,N))
    tau = np.zeros((D,N))
    
    for d in range(1,D+1):
        for n in range(N):
            # Initializing the data of this iteration:
            dhats = []
            nus = []
            taus = []

            if d >=10 and n >=1000:
                n_jobs=n_cpus
            else:
                n_jobs=1

            # Saying to the user what I'm doing:
            print('(d = {} \tn = {})...'.format(d,cardinalities[n]))
            # Executing the experiments:
            for i in range(iterations):
                # Generating the dataset:
                data = randsphere(cardinalities[n],d,radius=1)[0]

                # Estimating the parameters:        
                res=dancoDimEstNoCalibration(data, k, d*2+5,n_jobs=n_jobs)
                dhats.append(res['dhat'])
                nus.append(res['mu_nu'])
                taus.append(res['mu_tau'])

            # Storing the results:
            dHat[d-1,n] = np.mean(dhats)
            mu[d-1,n] = np.mean(nus)
            tau[d-1,n] = np.mean(taus)
    
    #fit splines
    kind='cubic'

    spline_dhat=interp2d(x=np.repeat(np.arange(1,D+1),N), 
           y=np.tile(cardinalities,D), 
           z=dHat.ravel(),
           kind=kind)

    spline_mu=interp2d(x=np.repeat(np.arange(1,D+1),N), 
           y=np.tile(cardinalities,D), 
           z=mu.ravel(),
           kind=kind)

    spline_tau=interp2d(x=np.repeat(np.arange(1,D+1),N), 
           y=np.tile(cardinalities,D), 
           z=tau.ravel(),
           kind=kind)
    
    DANCo_splines = {}
    for spl in ['spline_dhat','spline_mu','spline_tau']:
        DANCo_splines[spl]=vars()[spl]
                
    if type(save_output_path) == str:
        np.savetxt(path+'dhat',dHat)
        np.savetxt(path+'mu',mu)
        np.savetxt(path+'tau',tau)
        
        for spl in ['spline_dhat','spline_mu','spline_tau']:
            with open(path+'DANCo_'+spl+'.pkl', 'wb') as f:
                pickle.dump(vars()[spl], f)
                
        #loading and returning saved splines to make sure everything is ok
        DANCo_splines = {}
        for spl in ['spline_dhat','spline_mu','spline_tau']:
            with open(path+'DANCo_'+spl+'.pkl', 'rb') as f:
                DANCo_splines[spl]=pickle.load(f)
         
        #alternative way to save splines as explicit parameters
        #from scipy.interpolate import BivariateSpline
        #tck = DANCo_splines['spline_dhat'].tck
        #spl2 = BivariateSpline._from_tck(tck)
        
        return dHat,mu,tau,DANCo_splines
    
    return dHat,mu,tau,DANCo_splines

def load_DANCoTrain_results(path=''):

    dHat = np.loadtxt(path+'dhat')
    mu = np.loadtxt(path+'mu')
    tau = np.loadtxt(path+'tau')
    
    DANCo_splines = {}
    for spl in ['spline_dhat','spline_mu','spline_tau']:
        with open(path+'DANCo_'+spl+'.pkl', 'rb') as f:
            DANCo_splines[spl]=pickle.load(f)
            
    return dHat,mu,tau,DANCo_splines

def plot_DANCoTrain(cardinalities,dHat,mu,tau,DANCo_splines):
    
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'},{'type': 'scatter3d'}]]
    )
    
    N=len(cardinalities)
    DANCostats=[dHat,mu,tau]
    for i in range(1,4):
        interp=DANCo_splines[list(DANCo_splines.keys())[i-1]](x=np.arange(1,len(dHat)),y=cardinalities)

        fig.add_trace(go.Scatter3d(x=np.repeat(np.arange(1,len(dHat)),N), 
                                   y=np.tile(cardinalities,len(dHat)), 
                                   z=DANCostats[i-1].ravel(),
                               marker=dict(size=1),
                               mode='markers'),                 
                      row=1,col=i)

        fig.add_trace(go.Surface(x=np.arange(1,len(dHat)), 
                                   y=cardinalities, 
                                   z=interp),
                      row=1,col=i)


    fig.update_layout(
        title_text='R datasets. Joint color range',
        height=400,
        width=1000)
    fig.show()
