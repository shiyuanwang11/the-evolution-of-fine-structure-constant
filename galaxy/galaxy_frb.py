from cobaya.run import run
import numpy as np
from colossus.cosmology import cosmology
cosmo=cosmology.setCosmology('planck18')
h = cosmo.H0/100

from scipy.integrate import quad, dblquad
from functools import partial

import hmvec as hm
import Func

## different from galaxies
zg=0.15
ngal = 3.2e-2*h**3    # /Mpc^3, 3.2e-2 1.1e-2 5.5e-3
fsky=17500./41253.
dz = 0.15
sigmaD = 300
Nfrb = 1e+5
zfrb = zg+dz+0.1

## fixed
zs = np.linspace(0.,1.,101)
ms = np.geomspace(2e10,1e17,200)
c = 3e+5  #km/s
c_speed = c
ne0 = Func.ne0_()    #1/m^3 0.17174573422830966
chig = cosmo.comovingDistance(0.,zg)/h # Mpc
dchig = cosmo.comovingDistance(z_min=zg-dz,z_max=zg+dz)/h   # Mpc
V = Func.Vs(fsky,zg-dz,zg+dz)

l_max = 500
ell_values = np.arange(l_max)[2:]
kk = ell_values/chig

index = np.where(zs==zg)[0].astype(int)

## init setting for halo model
hcos1 = hm.HaloModel(zs,kk,ms=ms)
hcos1.add_hod("g",ngal=ngal+zs*0.,corr="max")
hcos1.add_battaglia_profile("electron",family="AGN",xmax=20,nxs=5000)


## power spectrum
def Clgg():
    pgg_1h = hcos1.get_power_1halo(name="g")[index,:][0,:]
    pgg_2h = hcos1.get_power_2halo(name="g")[index,:][0,:]
    pgg = pgg_1h + pgg_2h

    clgg = pgg/chig**2/dchig

    return clgg

def Clge(alpha):
    pge_1h = hcos1.get_power_1halo(name="g", name2="electron")[index,:][0,:]
    pge_2h = hcos1.get_power_2halo(name="g", name2="electron")[index,:][0,:]
    pge = pge_1h + pge_2h

    cldg = ne0*(1+zg)/chig**2*pge * (1+alpha)

    return cldg

def Cldd_int(ell):
    """
    caculate integral part of Cldd, get Cldd in (pc/cm^3)^2 after multiply ne0(m^-3)
    ell is a given value
    """

    z_int = np.linspace(0.,zfrb,20)  # len = 20
    comoving_d = cosmo.comovingDistance(z_min=0.,z_max=z_int[1:])/h   #len = 19
    kl = np.flip(ell/comoving_d)    #len = 19
    dzint = np.diff(z_int)[5]
    
    hcos = hm.HaloModel(z_int,kl,ms=ms)
    hcos.add_battaglia_profile("electron",family="AGN",xmax=20,nxs=5000)
    pee_full = hcos.get_power("electron","electron",verbose=False )[1:,:]    #shape:(20,19) changes to (19,19)
    pee_diag = np.diagonal(pee_full)
    pee = pee_diag.reshape(-1, 1)
    
    z_chi_h = (1+z_int[1:])**2/comoving_d**2*c/cosmo.Hz(z_int[1:])
    zchih = z_chi_h#[np.newaxis, :]
    
    result = 0
    for i in range(len(z_int)-1):
        result = result+zchih[i]*pee[i]*dzint

    return result*ne0**2


## noise
def Nldd(sigmad,fsky,N):
    """
    caculate DM noise power spectrum, sigmad can be 100, 300, 1000 pc/cm^3
    nf2d is the number density (per steradian) of FRBs
    return Nl in (pc/cm^3)^2
    """
    omega = 4*np.pi*fsky
    nf2d = N/omega
    return sigmad**2/nf2d

def Nlgg(ngal,fsky,V):
    omega1 = 4*np.pi*fsky
    ng2d = ngal*V/omega1
    return 1/ng2d

def Nldg2(clgg, nlgg, cldd, nldd):
    return (clgg+nlgg)*(cldd+nldd)



## caculation of noise
print("cldd...")
#cldd = np.array([Cldd_int(l) for l in ell_values])
#np.savetxt('/home/wangsy2/fs_const/galaxy/cldd_575.txt', cldd, fmt='%.10e',comments='')
cldd = np.loadtxt('/home/wangsy2/fs_const/galaxy/cldd_515.txt', delimiter=' ', dtype='str').astype(float)

clgg = Clgg()
nl_gg = Nlgg(ngal,fsky=fsky,V=V)
nl_dd = Nldd(sigmad=sigmaD,N=Nfrb,fsky=fsky)
nl_dg2 = Nldg2(clgg,nl_gg,cldd,nl_dd)
np.savetxt('/home/wangsy2/fs_const/galaxy/nl_dg2_515.txt', nl_dg2, fmt='%.10e',comments='')
#nl_dg2 = np.loadtxt('/home/wangsy2/fs_const/galaxy/nl_dg2.txt', delimiter=' ', dtype='str').astype(float)



## caculation of Cldg_fid

cldg_fid = Clge(0.)


## likelihood

def log_likelihood(alpha):

    chi1 = cldg_fid - Clge(alpha)
    chi2 = nl_dg2
    chi = np.sum(chi1**2 / chi2)
    return -0.5 * chi

print("mcmc...")
info = {"likelihood": {"loglike": log_likelihood}, \
        "params": {"alpha": {"prior": {"min": -0.1, "max": 0.1},'ref': 0.,"latex": r'\alpha'}, \
                   }, \
        "sampler": {"mcmc": {"Rminus1_stop": 0.01, "max_tries": 10000},},\
        "output": "chains575/frb575"
        }
# , "proposal":0.0001                   
updated_info, sampler = run(info,force=True)