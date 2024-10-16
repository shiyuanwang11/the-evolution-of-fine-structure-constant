import numpy as np
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck18')
from astropy.cosmology import Planck18
c=Planck18
from colossus import utils
from cobaya.run import run
from scipy import stats
from scipy.integrate import quad


######################## get power spectrum and noise
Y = 0.24
rhob0 = 4.2e-25 #g/m^3
mp = utils.constants.M_PROTON #1.672621898e-24(g)
c_speed = 1e-2*utils.constants.C #m/s
#dm_host0 = 100 # pc cm^-3
z0 = 1.5
bias_FRB = 1
sigma_dm = 30
fsky = 0.8

l_max = 500
ell_values = np.arange(l_max)[2:]

def N(z):
    return z**2*np.exp(-z*2)

z1 = np.linspace(0,z0,100)
nbar = np.trapz(N(z1),z1)


#W_IGM, a constant figm
def Wigm(z,figm):
    z_int = np.linspace(z,20.,100)
    w_int = N(z)
    wint = np.trapz(w_int,z_int)
    Wigm = wint*(1-0.5*Y)*figm*rhob0/mp*(1+z)/cosmo.Hz(z)
    return Wigm/nbar

#W_host
def Sfr(z):
    return (0.0156+0.118*z)/(1+z/3.23)**4.66

def Whost(z,dm_host0):
    dm_host = dm_host0*np.sqrt(Sfr(z)/Sfr(0))
    whost1 = dm_host/(1+z)*N(z)
    return whost1/nbar


# Power spectrum, delta_b = 1 (large scale)
def Cligm_core(ell, gamma, figm):
    zz = np.linspace(1e-4,z0,100)
    kl = (ell+0.5)/cosmo.comovingDistance(z_min=0.0,z_max=zz)
    fine = 1.-gamma*np.log(1.+zz)
    cligm1 = Wigm(zz,figm)**2*cosmo.Hz(zz)/cosmo.comovingDistance(z_min=0.0,z_max=zz)**2*cosmo.matterPowerSpectrum(kl,zz) * fine**2
    cligm2 = np.trapz(cligm1,zz)*c_speed/1000
    return cligm2

def Cligm(gamma,figm):
    return np.array([Cligm_core(el, gamma,figm) for el in ell_values])

def Clhost_core(ell, gamma, dm_host0):
    zz = np.linspace(1e-4,z0,100)
    kl = (ell+0.5)/cosmo.comovingDistance(z_min=0.0,z_max=zz)
    fine = 1.-gamma*np.log(1.+zz)
    clhost1 = Whost(zz, dm_host0)**2*cosmo.Hz(zz)/cosmo.comovingDistance(z_min=0.0,z_max=zz)**2*bias_FRB*cosmo.matterPowerSpectrum(kl,zz) * fine**2
    clhost2 = np.trapz(clhost1,zz)/c_speed*1000
    return clhost2

def Clhost(gamma, dm_host0):
    return np.array([Clhost_core(el, gamma, dm_host0) for el in ell_values])

def Clih_core(ell, gamma, figm, dm_host0):
    zz = np.linspace(1e-4,z0,100)
    kl = (ell+0.5)/cosmo.comovingDistance(z_min=0.0,z_max=zz)
    fine = 1.-gamma*np.log(1.+zz)
    clih1 = 2*Whost(zz, dm_host0)*Wigm(zz,figm)*cosmo.Hz(zz)/cosmo.comovingDistance(z_min=0.0,z_max=zz)**2*bias_FRB*cosmo.matterPowerSpectrum(kl,zz) * fine**2
    clih2 = np.trapz(clih1,zz)
    return clih2   

def Clih(gamma,figm, dm_host0):
    return np.array([Clih_core(el, gamma,figm, dm_host0) for el in ell_values])

def Cl(gamma,figm, dm_host0):
    #print('Cl...')
    return Cligm(gamma, figm)+Clhost(gamma, dm_host0)+Clih(gamma, figm, dm_host0)

# Noise
def Nl(cldm,nfrb):
    nhost = 4.*np.pi*fsky*(sigma_dm)**2/nfrb
    return 1. / np.sqrt((2.*ell_values+1.)*fsky) * (cldm+nhost)


########################## likelihood

print('Cl...')
cl_data = Cl(0., 0.75, 100)
noise = Nl(cl_data, 1e+7)

def log_likelihood(gamma, figm, dm_host0):
        chi1 = cl_data - Cl(gamma, figm, dm_host0)
        chi2 = noise
        chi = np.sum(chi1**2 / chi2**2)
        return -0.5 * chi


info = {"likelihood": {"loglike": log_likelihood}, \
        "params": {"gamma": {"prior": {"min": -0.1, "max": 0.1},'ref': 0.,"latex": r'\gamma',"proposal": 0.0002}, \
                   "figm": {"prior": {"min": 0., "max": 1.0},'ref': 0.75,"latex": r'f_{igm}',"proposal":0.001}, \
                   "dm_host0": {"prior": {"dist": "norm", "loc": 100., "scale": 10.},'ref': 100.0,"latex": r'DM_{host,0}',"proposal":0.1}   # chains7_1014 
                   }, \
        "sampler": {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 10000},},\
        "output": "chains7_1014_bigger/frb7"
        }
# 
updated_info, sampler = run(info,force=True)


# _bigger: gamma prior -- 0.0002 , and it is 0.0001 for normal version

