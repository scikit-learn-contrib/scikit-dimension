import scipy.io
import numpy as np
import subprocess
from pathlib import Path
#if installation doesnt work, copy/paste this into the notebook and replace path_to_estimators with :
#from os.path import dirname, abspath
#path_to_estimators = dirname(abspath(''))+'/estimators/'

path_to_estimators = str(Path(__file__).resolve().parent)
path_to_matlab = '/home/utilisateur/MATLAB/R2020a/bin/matlab'

def TwoNN(data,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run C++ TwoNN script '''
    path_to_estimators += '/TWO-NN-master/'
    np.savetxt(path_to_estimators+'temp_data.txt',data)
    proc= subprocess.check_output("cd "+path_to_estimators+" && ./TWONN.o"+' -input '+path_to_estimators+'temp_data.txt '+'-coord',shell=True)
    t=str(proc)
    s=t.find('estimated')
    e=t[s:].find('\\')
    dim=float(t[s:e+s].split()[2])
    return dim

def run_singleGMST(data,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run matlab script from shell '''
    path_to_estimators += '/drtoolbox/'
    np.savetxt(path_to_estimators+'temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                        "data=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');d = intrinsic_dim(data,'GMST');save('variables.mat');exit;"""])
    return scipy.io.loadmat(path_to_estimators+'variables.mat')['d'][0]   

def run_singleCorrDim(data,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run matlab script from shell '''
    path_to_estimators += '/drtoolbox/'
    np.savetxt('temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                        "data=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');d = intrinsic_dim(data,'CorrDim');save('variables.mat');exit;"""])
    return scipy.io.loadmat(path_to_estimators+'variables.mat')['d'][0]   

def runDANCo(data,k=10,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run matlab script from shell '''
    path_to_estimators += '/idEstimation/idEstimation/'
    np.savetxt(path_to_estimators+'temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                        "data=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');k="+str(k)+";[d,kl,mu2,tau2] = DANCoFit(data',k);save('variables.mat');exit;"""])
    return scipy.io.loadmat(path_to_estimators+'variables.mat')['d'][0]

def runDANCoStats(data,k=10,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run matlab script from shell '''
    path_to_estimators += '/idEstimation/idEstimation/'
    np.savetxt(path_to_estimators+'temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                        "data=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');k="+str(k)+";[d,kl,dHat,mu,tau,mu2,tau2] = DANCoFit(data',k);save('variables.mat');exit;"""])
    return scipy.io.loadmat(path_to_estimators+'variables.mat')

def runDANColoop(data,k=10,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run matlab script from shell '''
    path_to_estimators += '/idEstimation/idEstimation/'
    np.savetxt(path_to_estimators+'temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                        "data=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');d=dancoLoop(data',"+str(k)+");save('variables.mat');exit;"""])
    return scipy.io.loadmat(path_to_estimators+'variables.mat')['d'][0]

def runANOVAglobal(data,k=None,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run matlab script from shell '''
    path_to_estimators += '/ANOVA_dimension_estimator-master/'
    if k is None:
        #ANOVA default
        k = min(500,max(round(10*np.log10(data.shape[0])), 10))
    np.savetxt(path_to_estimators+'/temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                        "data=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');k="+str(k)+";[d,dh]=ANOVA_global_estimator(data','k',k);save('variables.mat');exit"""])

    return scipy.io.loadmat(path_to_estimators+'variables.mat')['dh']

def runANOVAlocal(data,k=None,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run matlab script from shell '''
    path_to_estimators += '/ANOVA_dimension_estimator-master/'
    if k is None:
        #ANOVA default
        k = min(500,max(round(10*np.log10(data.shape[0])), 10))
    np.savetxt(path_to_estimators+'/temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                        "data=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');k="+str(k)+";[estimates, first_moments,indicesknn]=ANOVA_local_estimator(data',data','k',k,'maxdimension',100);save('variables.mat');exit"""])

    return scipy.io.loadmat(path_to_estimators+'variables.mat')['estimates']

def radovanovic_estimators_matlab(data, k = 100, path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    path_to_estimators += '/id-tle-code'
    np.savetxt(path_to_estimators+'/temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                    "X=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');k="+str(k)+";radovanovic_estimators;exit"""])
    
    estimators = ['id_mle','id_tle','id_mom','id_ed','id_ged','id_lpca','id_mind_ml1','id_mind_mli']
    return dict(zip(estimators,
                    [np.genfromtxt(path_to_estimators+'/'+est+'.csv',delimiter=',') for est in estimators]))


def radovanovic_estimators_matlab_onerun(data, k = 100, path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    path_to_estimators += '/id-tle-code'
    np.savetxt(path_to_estimators+'/temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                    "X=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');k="+str(k)+";radovanovic_estimators_onerun;exit"""])
    
    estimators = ['id_mle','id_tle','id_mom','id_ed','id_ged','id_lpca','id_mind_ml1','id_mind_mli']
    return dict(zip(estimators,
                    [np.genfromtxt(path_to_estimators+'/'+est+'.csv',delimiter=',') for est in estimators]))

def Hidalgo(data,path_to_estimators=path_to_estimators, path_to_matlab = path_to_matlab):
    ''' Run matlab script from shell '''
    path_to_estimators += '/Hidalgo/'
    np.savetxt(path_to_estimators+'/temp_data.txt',data)
    subprocess.call([path_to_matlab,"-nodisplay", "-nosplash", "-nodesktop","-nojvm","-r",
                        "data=dlmread('"+path_to_estimators+"/temp_data.txt');cd ('"+path_to_estimators+"');out=HeterogeneousID(data);save('variables.mat');exit"""])

    return scipy.io.loadmat(path_to_estimators+'variables.mat')['out']
