# This file is part of SIGWfast.

# SIGWfast is free software: you can use, copy, modify, merge, publish and
# distribute, sublicense, and/or sell copies of it, and to permit persons to
# whom it is furnished to do so it under the terms of the MIT License.

# SIGWfast is distributed in the hope that it will be useful,
# but WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. See the MIT License for more details.

# You should have received a copy of the MIT License along with SIGWfast. If
# not, see <https://spdx.org/licenses/MIT.html>.

### Global
import os, sys, time
import numpy as np
import tqdm as tqdm
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

### GWfast
sys.path.append('libraries/')
import sdintegral as sd

#=============================================================================#
                              # CONFIGURATION #
#=============================================================================#

# Name of file where Omega_GW(k) is to be stored as a .npz file in the data 
# subdirectory. The k-values and Omega-GW-values will be stored with keywords 
# 'karray' and 'OmegaGW' respectively.   
filenameGW = 'OmegaGW_of_k'

# Choose whether to regenerate the data. If True, Omega_GW(k) is recomputed.
# If False, the data in data/filenameGW.npz is plotted and new data is only   
# computed if this file is absent. This is a safety-measure to not overwrite
# previous data.
regenerate = True #False

# Choose whether to compute Omega_GW from a primordial scalar power spectrum
# given in numerical or analytical form. Set Num_Pofk = True for using 
# numerical data as input for P(k). Set Num_Pofk = False for using an anlytical 
# formula for P(k) to be defined below.
Num_Pofk = False #True

# If numerical data is to be used as input for P(k) (Num_Pofk = True), declare
# the name of the .npz file containing the data. This file is to be placed in 
# the data subdrectory and prepared in a way so that k-values and associated 
# P(k)-values are accessed via the keywords 'karray' and 'Pzeta'. This file is 
# not needed if P(k) is provided via an analytic formula instead and no name
# needs to be declared.
filenamePz = 'P_of_k'

# Choose whether to compile a C++ module to perform the integration. If set to 
# False the entire computation is peformed using existing python modules only.
# Note that Use_Cpp = True will only work for Linux / MacOS, but not Windows.
Use_Cpp = False #True

# Set equation of state parameter w with 0 < w < 1
w = 1/3 # 0 < w < 1

# Set cs_equal_one = True if the sound speed c_s is unity, i.e. c_s^2=1, as for
# a universe dominated by a canonical scalar field. 
# Set cs_equal_one = False for c_s^2=w, i.e. a universe behaving like an 
# adiabatic perfect fluid, like e.g. for radiation domination (c_s^2=w=1/3).
cs_equal_one = False #True

# Set the normalisation factor that multiplies Omega_GW.
norm = 1 

# Set limits kmin and kmax of the interval in k for which Omega_GW is to be 
# computed. Also set the number nk of entries of the k-array.
kmin = 0.01 # | in some arbitrary reference units
kmax = 2.50 # | denoted by k_{ref} in the plots.
nk   = 200

# Declare the array of k-values
komega = np.linspace(kmin,kmax,nk,dtype=np.float64) # linear spacing
#komega = np.geomspace(kmin,kmax,nk,dtype=np.float64) # logarithmic spacing

#=============================================================================#
                    # PRIMORDIAL SCALAR POWER SPECTRUM #
#=============================================================================#

# This block of code needs to be modified only if an analytical formula for 
# P(k) is to be used, i.e. Num_Pofk = False has been set above. In this case 
# the scalar power spectrum has to be defined below as the function Pofk(k).
# This should take a single argument which is the wavenumber k in the units of 
# choice. All other parameters should be defined as either global or local 
# variables. 

###############################################################################
################# DEFINE YOUR OWN SCALAR POWER SPECTRUM HERE: #################
###############################################################################

# Default example: P(k) as arises for a strong sharp turn in the inflationary
# trajectory, see eq. (2.25) in arXiv:2012.02761. This exhibits O(1) 
# oscillations modulating a peaked envelope. Here P(k) is normalised by a 
# factor np.exp(-2*delta*etap) to avoid excessively large values. To account 
# for this, one should multiply the final result for Omega_GW by a factor
# np.exp(4*delta*etap).

# Define the model parameters as global variables with fixed values.
# Duration delta of the turn in e-folds
delta = 0.5
# Strength of the turn, i.e. turn rate in units of the Hubble scale
etap  = 14
# Normalisation of power spectrum 
P0    = 1 # 2.4*10**(-9) for CMB value

def Pofk(k):
    # P(k) in eq. (2.25) of arXiv:2012.02761 is valid for 0 < k < 2.
    # For some values of delta and etap one finds unphysical spikes for
    # k -> 0 or k -> 2. We remove these by cutting P(k) at k=kcut and k=2-kcut.
    kcut  = 0.001
    if k > 2-kcut:
        P = 0
    elif k < kcut:
        P = 0
    else:
        # Eq. (2.25) in arXiv:2012.02761 multiplied by np.exp(-2*delta*etap)
        P = np.exp(2*(np.sqrt((2-k)*k)-1)*delta*etap)/4/(2-k)/k*(
            1+(k-1)*np.cos(2*np.exp(-delta/2)*etap*k)
             +np.sqrt(abs(2-k)*k)*np.sin(2*np.exp(-delta/2)*etap*k))
    return P0*P

###############################################################################
###############################################################################
###############################################################################

# Define the array of k-values over which P(k) is to be discretized and then 
# interpolated. This should include the entire interval where P(k) is not 
# negligible and will hence contribute to Omega_GW(k).  
# By default, this interval is defined to be wider than komega by a factor 
# fac^2 in log(k)-space and also sampled more densely by a factor (int(fac))^2.
# The default value of this factor is chosen as fac=2. 
fac = 2
kpzeta = np.linspace(np.amin(komega)/fac, np.amax(komega)*fac,
                     (len(komega))*(int(fac))**2)
# Uncomment lines below if logarithmic spacing is desired.
#kpzeta = np.geomspace(np.amin(komega)/fac, np.amax(komega)*fac,
#                     (len(komega))*(int(fac))**2)

#=============================================================================#
                        # COMPUTATION OF OMEGA_GW #
#=============================================================================#

# Prepare primordial power spectrum as an interpolation function.
# Prepare from numerical data.
if Num_Pofk:
    try: 
        #Load data.
        Pdata  = np.load('data/'+filenamePz+'.npz')
        kpzeta = Pdata['karray']
        # Define interpolation function.
        Pinter = interp1d(Pdata['karray'], Pdata['Pzeta'], 
                          fill_value='extrapolate')
    except FileNotFoundError:
        print('No file data/'+filenamePz+'.npz!')
        sys.exit()
# Prepare from analytical formula.
else:
    # Discretize P(k) by evaluating Pofk on kpzeta. As Pofk(k) is not 
    # guaranteed to be vectorizable do this in an element-wise manner. The  
    # small performance loss is acceptable as this has to be done only once.
    Pdisc = np.zeros(len(kpzeta))
    for i in range(0, len(kpzeta)):
        Pdisc[i] = Pofk(kpzeta[i])
    # Define interpolation function.     
    Pinter = interp1d(kpzeta, Pdisc, fill_value='extrapolate')
    # For a vectorizable function Pofk(k) one could have used instead:
    #Pinter = interp1d(kpzeta, Pofk(kpzeta), fill_value='extrapolate')
    # Uncomment to save analytic power spectrum in numerical form
    #np.savez('data/'+filenameP, karray=kpzeta, Pzeta=Pinter(kpzeta))

def compute_w():
    # Declare beta=(1-3w)/(1+3w)
    beta=sd.beta(w)
    # Declare arrays of integration variables d and s. The s-array is split
    # into two arrays for the interval s<1/sqrt(w), s1array, and one for the
    # interval s>1/sqrt(w), s2array. This split is done as for w >=1/3 the 
    # integration kernel diverges at s=1/sqrt(w). The argument kmin is needed
    # as this sets the cutoff smax of the s-array.
    nd,ns1,ns2, darray,d1array,d2array, s1array,s2array = sd.arrays_w(w,komega)
    
    if Use_Cpp:
        # Add all directories within libraries/lib/python containing files to 
        # python path to ensure the module sigwfast can be imported if it has
        # been already compiled.
        file_dir = os.path.dirname(__file__) + '/libraries'
        for path, subdirs, files in os.walk(file_dir+'/lib/python'):
            for name in files:
                sys.path.append(path)
        try:
            # Attempt to import the module sigwfast
            from sigwfast import sigwint_w
        except ModuleNotFoundError:
            # If sigwfastdoes not yet exist, initiate its compilation.
            # Import modules needed to compile the new module.
            import subprocess, shutil
            # Call libraries/setup.py to compile a C++ module.
            setup_name = os.path.join(file_dir, 'setup.py')
            subprocess.call(['python', setup_name, 'install', 
                             '--home=' + file_dir], cwd=file_dir)
            # Remove build directory to remove clutter.
            shutil.rmtree(file_dir + '/build/', ignore_errors=False)
            # Redo going over all files in libraries/lib/python and adding 
            # their path to python path to ensure the module sigwfast can be 
            # successfully imported.
            for path, subdirs, files in os.walk(file_dir+'/lib/python'):
                for name in files:
                    sys.path.append(path) 
            # Import the compiled module.
            from sigwfast import sigwint_w

        # Integration to compute Omega_GW.
        # Define array to hold the final result from the integration
        nk = len(komega) # in case nk was not defined before
        Int = np.zeros(nk)
        # Start timer.           
        start1 = time.time()
        # Fill in the integration kernels that multiplies the two factors of
        # the power spectrum as flattened arrays, with kernel1 to be used for
        # the integration over s<1/sqrt(w) and kernel2 for s>1/sqrt(w).
        kernel1 = sd.kernel1_w(d1array, s1array, beta)
        kernel2 = sd.kernel2_w(d2array, s2array, beta)
        # Compute Omega_GW for every value of k in komega by performing   
        # discrete integration over d and s.
        for k in tqdm.tqdm(range(0,nk)):
            # Fill in integrands as flattened arrays by multiplying the kernels
            # by the two factors of the power spectrum
            Int_ds1 = kernel1*sd.Psquared(d1array, s1array, Pinter, komega[k])
            Int_ds2 = kernel2*sd.Psquared(d2array, s2array, Pinter, komega[k])
            # Perform integration with function sigwint_w from compiled module
            Int[k] = sigwint_w(np.array(Int_ds1),np.array(Int_ds2),
                               np.array(darray),np.array(s1array),
                               np.array(s2array),nd,ns1,ns2)
        # Multiply the result from the integration by the normalization and the
        # k-dependent redshift factor to get the final result of Omega_GW
        OmegaGW = norm*(komega)**(-2*beta)*Int
        #Stop timer 
        end1 = time.time()
        #Total time is printed as output
        print('total computation time =',end1-start1)
    
    else:        
        # Integration to compute Omega_GW.
        # Define arrays to hold final and intermediate results from integration
        nk = len(komega) # in case nk was not defined before
        Int   = np.zeros(nk)
        Int_d = np.zeros(nd)
        # Start timer.           
        start1 = time.time()
        # Fill in the integration kernels that multiplies the two factors of
        # the power spectrum as flattened arrays, with kernel1 to be used for
        # the integration over s<1/sqrt(w) and kernel2 for s>1/sqrt(w).
        kernel1 = sd.kernel1_w(d1array, s1array, beta)
        kernel2 = sd.kernel2_w(d2array, s2array, beta)
        # Compute Omega_GW for every value of k in komega by performing  
        # discrete integration over d and s.
        for k in tqdm.tqdm(range(0,nk)):
            # Fill in integrands as flattened arrays by multiplying the kernels
            # by the two factors of the power spectrum
            Int_ds1 = kernel1*sd.Psquared(d1array, s1array, Pinter, komega[k])
            Int_ds2 = kernel2*sd.Psquared(d2array, s2array, Pinter, komega[k])
            # Loop over the d-array
            for i in range(0,nd):
                # Implement the limits of integration over s on the indices
                # of the flattened integrands
                i1=i*ns1
                i2=(i+1)*ns1
                j1=i*ns2
                j2=(i+1)*ns2
                # Perform the s-integration via sd.intarray1D,performing the 
                # integral over s<1/sqrt(w) and s>1/sqrt(w) separately.
                Int_d[i] =(sd.intarray1D(Int_ds1[i1:i2],s1array[i1:i2])+
                           sd.intarray1D(Int_ds2[j1:j2],s2array[j1:j2]))
            # Perform the integral over d using sd.intarray1D
            Int[k] = sd.intarray1D(Int_d,darray)
        # Multiply the result from the integration by the normalization and the
        # k-dependent redshift factor to get the final result of Omega_GW
        OmegaGW = norm*(komega)**(-2*beta)*Int
        #Stop timer 
        end1 = time.time()
        #Total time is printed as output
        print('total computation time =',end1-start1)
    
    # Save results in npz file in data subfolder    
    np.savez('data/'+filenameGW, karray=komega, OmegaGW=OmegaGW)
    
    # Plot P(k) used in the last run
    fig, ax = plt.subplots()
    ax.plot(kpzeta, Pinter(kpzeta), color='red')
    ax.set_title(r'$P_\zeta$ vs. $k$')
    ax.set_xlabel(r'$k \ / \ k_{ref}$')
    ax.set_ylabel(r'$P_\zeta$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    return OmegaGW
    
def compute_1():
    # Declare beta=(1-3w)/(1+3w)
    beta=sd.beta(w)
    # Declare arrays of integration variables d and s. The s-array is split
    # into two arrays for the interval s<1/sqrt(w), s1array, and one for the
    # interval s>1/sqrt(w), s2array. This split is done as for w >=1/3 the 
    # integration kernel diverges at s=1/sqrt(w). The argument kmin is needed
    # as this sets the cutoff smax of the s-array.
    nd, ns, darray, ddarray, ssarray = sd.arrays_1(w,komega)
    
    if Use_Cpp:
        # Add all directories within libraries/lib/python containing files to 
        # python path to ensure the module sigwfast can be imported if it has
        # been already compiled.
        file_dir = os.path.dirname(__file__) + '/libraries'
        for path, subdirs, files in os.walk(file_dir+'/lib/python'):
            for name in files:
                sys.path.append(path)
        try:
            # Attempt to import the module sigwfast
            from sigwfast import sigwint_1
        except ModuleNotFoundError:
            # If sigwfastdoes not yet exist, initiate its compilation.
            # Import modules needed to compile the new module.
            import subprocess, shutil
            # Call libraries/setup.py to compile a C++ module.
            setup_name = os.path.join(file_dir, 'setup.py')
            subprocess.call(['python', setup_name, 'install', 
                             '--home=' + file_dir], cwd=file_dir)
            # Remove build directory to remove clutter.
            shutil.rmtree(file_dir + '/build/', ignore_errors=False)
            # Redo going over all files in libraries/lib/python and adding 
            # their path to python path to ensure the module sigwfast can be 
            # successfully imported.
            for path, subdirs, files in os.walk(file_dir+'/lib/python'):
                for name in files:
                    sys.path.append(path) 
            # Import the compiled module.
            from sigwfast import sigwint_1

        # Integration to compute Omega_GW.
        # Define array to hold the final result from the integration
        nk = len(komega) # in case nk was not defined before
        Int = np.zeros(nk)
        # Start timer.           
        start1 = time.time()
        # Fill in the integration kernels that multiplies the two factors of
        # the power spectrum as flattened arrays, with kernel1 to be used for
        # the integration over s<1/sqrt(w) and kernel2 for s>1/sqrt(w).
        kernel = sd.kernel_1(ddarray, ssarray, beta)
        # Compute Omega_GW for every value of k in komega by performing 
        # discrete integration over d and s.
        for k in tqdm.tqdm(range(0,nk)):
            # Fill in integrands as flattened arrays by multiplying the kernels
            # by the two factors of the power spectrum
            Int_ds = kernel*sd.Psquared(ddarray, ssarray, Pinter, komega[k])
            # Perform integration with function sigwint_1 from compiled module
            Int[k] = sigwint_1(np.array(Int_ds),np.array(darray),
                               np.array(ssarray),nd,ns)
        # Multiply the result from the integration by the normalization and the
        # k-dependent redshift factor to get the final result of Omega_GW
        OmegaGW = norm*(komega)**(-2*beta)*Int
        #Stop timer 
        end1 = time.time()
        #Total time is printed as output
        print('total computation time =',end1-start1)
    
    else:        
        # Integration to compute Omega_GW.
        # Define arrays to hold final and intermediate results from integration
        nk = len(komega) # in case nk was not defined before
        Int   = np.zeros(nk)
        Int_d = np.zeros(nd)
        # Start timer.           
        start1 = time.time()
        # Fill in the integration kernels that multiplies the two factors of
        # the power spectrum as flattened arrays, with kernel1 to be used for
        # the integration over s<1/sqrt(w) and kernel2 for s>1/sqrt(w).
        kernel = sd.kernel_1(ddarray, ssarray, beta)
        # Compute Omega_GW for every value of k in komega by performing 
        # discrete integration over d and s.
        for k in tqdm.tqdm(range(0,nk)):
            # Fill in integrands as flattened arrays by multiplying the kernels
            # by the two factors of the power spectrum
            Int_ds = kernel*sd.Psquared(ddarray, ssarray, Pinter, komega[k])
            # Loop over the d-array
            for i in range(0,nd):
                # Implement the limits of integration over s on the indices
                # of the flattened integrands
                j1=i*ns
                j2=(i+1)*ns
                # Perform the s-integration via sd.intarray1D,performing the 
                # integral over s<1/sqrt(w) and s>1/sqrt(w) separately.
                Int_d[i] = sd.intarray1D(Int_ds[j1:j2],ssarray[j1:j2])
            # Perform the integral over d using sd.intarray1D
            Int[k] = sd.intarray1D(Int_d,darray)
        # Multiply the result from the integration by the normalization and the
        # k-dependent redshift factor to get the final result of Omega_GW
        OmegaGW = norm*(komega)**(-2*beta)*Int
        #Stop timer 
        end1 = time.time()
        #Total time is printed as output
        print('total computation time =',end1-start1)
    
    # Save results in npz file in data subfolder    
    np.savez('data/'+filenameGW, karray=komega, OmegaGW=OmegaGW)
    
    # Plot P(k) used in the last run
    fig, ax = plt.subplots()
    ax.plot(kpzeta, Pinter(kpzeta), color='red')
    ax.set_title(r'$P_\zeta$ vs. $k$')
    ax.set_xlabel(r'$k \ / \ k_{ref}$')
    ax.set_ylabel(r'$P_\zeta$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    return OmegaGW

def main():
    if (w<=0 or w>=1):
        print('Need to choose 0 < w < 1.')
    else:
        if regenerate:
            kplot = komega
            if cs_equal_one:
                Omegaplot = compute_1()
            else:
                Omegaplot = compute_w()
        else:
            try:    
                GWdata    = np.load('data/'+filenameGW+'.npz')
                kplot     = GWdata['karray']
                Omegaplot = GWdata['OmegaGW']
            except FileNotFoundError:
                print('No result file data/'+filenameGW+'.npz found. ' \
                      'Initiate new computation.')
                kplot = komega
                if cs_equal_one:
                    Omegaplot = compute_1()
                else:
                    Omegaplot = compute_w()
        
        # Plot Omega_GW(k)
        fig, ax = plt.subplots()
        ax.plot(kplot, Omegaplot)
        ax.set_title(r'$\Omega_{GW}$ vs. $k$')
        ax.set_xlabel(r'$k \ / \ k_{ref}$')
        ax.set_ylabel(r'$\Omega_{GW} \times (k_{ref} \ / \ k_{rh})^{2b}$')
        ax.set_xscale('log')
        ax.set_yscale('log')

#=============================================================================#
                     # EXECUTE SCRIPT AS MAIN PROGRAMME #
#=============================================================================#

if __name__ == "__main__":
    #Ensure regenerate, Num_Pok, Use_Cpp and cs_equal_one are boolean variables
    if type(regenerate) != bool:
        print('regenerate can only take the values True or False.')
    if type(Num_Pofk) != bool:
        print('Num_Pofk can only take the values True or False.')
    if type(Use_Cpp) != bool:
        print('Use_Cpp can only take the values True or False.')
    if type(cs_equal_one) != bool:
        print('cs_equal_one can only take the values True or False.')
    if (type(regenerate) == bool and type(Num_Pofk) == bool and 
        type(Use_Cpp) == bool and type(cs_equal_one) == bool):
        # Check whether OS is Windows. If True, ensure that the compilation of
        # the C++ module is deactivated by setting Use_Cpp = False.
        if os.name == 'nt'  and Use_Cpp:
            print('C++ version not available for Windows!')
            print('Will use python-only version.')
            Use_Cpp = False
        # Perform the computation and show the plots.
        main()
        plt.show()
