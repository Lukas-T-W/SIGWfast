# SIGWfast documentation

## Description

SIGWfast release 1.0 (2022). This code has been written by Dr Lukas T. Witkowski and is distributed under the MIT License.

SIGWfast is a python code to compute the Scalar-Induced Gravitational Wave spectrum from a primordial scalar power spectrum that can be given in analytical or numerical form. SIGWfast was written with the aim of being easy to install and use, and to produce results fast, typically in a matter of a few seconds. To this end the code employs vectorization techniques within python, but there is also the option to compile a C++ module to perform the relevant integrations, further accelerating the computation. The python-only version should run on all platforms that support python3. The version employing the C++ module is only available for Linux and MacOS systems. For more details on the physics see the scientific documentation file "SIGWfastGuide.pdf".

## Prerequisites

SIGWfast can be used on any system that supports python3. We recommend using python environments and a package manager such as `conda`. The following python modules are required:

* math, 
* matplotlib, 
* numpy, 
* os, 
* scipy, 
* sys, 
* time, 
* tqdm. 

The modules `time` and `tqdm` are for timing the computation and displaying a progress bar, respectively, and could be dispensed with by commenting out appropriate lines of code. 

Optional C++ extension: This is only supported on systems running on Linux or MacOS. Compiling the C++ extension further requires the modules:

* distutils, 
* platform,
* shutil, 
* subprocess,  

and a working C++ compiler.

## Installation
Download SIGWfast.zip. After decompression the necessary files and directory structure are already in place in the parent directory "SIGWfast". 

## User guide

### File content

The parent directory contains two python files:
1. SIGWfast.py computes the gravitational wave spectrum induced during an era of radiation domination. This is expected to be the case of principal interest for most users and SIGWfast.py is a simple no-frills code to get this result quickly. 
2. In the second code file SIGWfastEOS.py the equation of state of the universe during gravitational wave generation can be chosen, and thus eras other than radiation domination can be considered. As a result, this code has more adjustable parameters than SIGWfast.py.

For more details on the precise quantities computed by the two codes and their differences see the scientific documentation file SIGWfastGuide.pdf. 

The subdirectory "libraries" contains files necessary for performing the computation:
* sdintegral.py contains the definitions and kernels for computing the relevant integrals.
* SIGWfast.cpp is a C++ script that performs the relevant integral.
* setup.py is the code that configures and executes the compilation of the python module that will execute the C++ code contained in SIGWfast.cpp. The resulting python module named "sigwfast" is also deposited in the "libraries" subfolder.

The subdirectory "data" will receive the result data in a .npz file. Also, if the input is a scalar power spectrum in numerical form, this needs to be provided in the data subdirectory as a .npz file. An example file "P_of_k.npz" is provided.

### Quick guide
Set flags and values for input parameters in the block of code titled "Configuration". Provide the scalar power spectrum either by defining a function `Pofk(k)` in the block of code titled "Primordial scalar power spectrum" or in form of numerical data in a file data/filenameP.npz. See the more detailed guide below for how this file is to be prepared. After this, you're good to go!

### Configuration of SIGWfast.py: step-by-step guide

This is a set of detailed instructions for configuring SIGWfast.py. After the header where the necessary modules are imported, the first block of code is labeled "Configuration". This is where we can adjust the code for our purposes. The necessary steps are as follows: 

1. Set `filenameGW`. Choose a name for the .npz file that will contain the results for Omega_GW(k) and that will be deposited in the "data" subdirectory. Note that if a file with this name already exists, a new run will in general overwrite the old file. To avoid this, see the next step.

2. Set the flag `regenerate`. If this is set to `True`, a run of the code will execute a new computation and save the result in a new file 'data/'+filenameGW+'.npz', possibly overwriting an old file of the same name. If the flag is set to `False`, after hitting run, the code checks whether a file 'data/'+filenameGW+'.npz' already exists. If this is the case, no new computation is performed and instead the data in the existing file is plotted. This is a safety-measure to avoid existing data to be overwritten by accident. If however 'data/'+filenameGW+'.npz' does not exist, the code proceeds to performing the computation and saving the new data.

3. Set the flag `Num_Pofk`. The code computes the scalar-induced gravitational wave spectrum using the primordial scalar power spectrum P(k) as input. P(k) can be provided in terms of numerical data or an analytic formula, the choice of which is declared by specifying the flag `Num_Pofk`. If set to `True`, Omega_GW(k) will be computed from a scalar power spectrum given by the numerical data in a .npz file in the "data" subdirectory. The name of the file can be specified in the next step. If the flag `Num_Pofk` is instead set to `False`, Omega_GW(k) will be computed from a scalar power spectrum that needs to declared explicitly as a function `Pofk(k)`. See the next section for for detailed instructions on how this is to be done.

4. Optional: declare `filenamePz`. In case Omega_GW(k) is to be computed from numerical data (`Num_Pofk = True`), give here the name of the .npz file located in the "data" subdirectory. The file is to be prepared so that k-values and associated P(k)-values are to be accessed via the keywords "karray" and "Pzeta", respectively. If P(k) is to be provided via an analytic formula instead (`Num_Pofk = False`), no input file is needed and this line of code is ignored.

5. Set the flag `Use_Cpp`. If set to `True`, the code imports methods from the compiled module `sigwfast` to perform the integration, or, if the module does not yet exist, initiates a compilation of it from /libraries/SIGWfast.cpp. This requires a functioning C++ compiler in addition to python. If set to `False`, the entire computation is done within python, using only the modules listed above under "Prerequisites".

6. Set `norm`. This is a normalization factor that multiplies the gravitational wave spectrum. For `norm = 1` the script SIGWfast.py (and SIGWfastEOS.py) computes the energy density fraction in gravitational waves Omega_GW at the time of radiation domination. To get the corresponding spectrum today requires a rescaling, which can be done by appropriately choosing `norm`. See the scientific documentation for more details. The data plotted and saved in 'data/'+filenameGW+'.npz' is the gravitational wave spectrum after multiplication with this normalization factor.

7. Declare the range of wavenumbers k stored in `komega` for which the gravitational wave spectrum is to be computed. Here this is done by defining both a lower limit `kmin`, an upper limit `kmax` and setting the number of entries `nk` of `komega`. This is then filled with values that are linearly spaced (`numpy.linspace`) or logarithmically spaced (`numpy.geomspace`). Alternative definitions of `komega` to these are perfectly allowed, as long as `komega` is a numpy array. Note that for an analytic primordial power spectrum as input (`Num_Pofk=False`), a good guideline is to choose `komega` such that P(k), when sampled over `komega`, exhibits all relevant features of the full scalar power spectrum.

### Configuration of SIGWfastEOS.py

All configuration steps of SIGWfast.py also apply to SIGWfastEOS.py. In addition we need to declare one more parameter and set one additional flag. 

8. In SIGWfastEOS.py we also need to set a value for the equation of state parameter `w` for the era during which the gravitational waves are induced. In SIGWfast.py this value was fixed to w=1/3 corresponding to radiation domination. In SIGWfastEOS.py the parameter w can take values in 0 < w < 1, corresponding to the range of validity of the transfer functions used.

9. Set the flag `cs_equal_one`. The computation of Omega_GW(k) can be done for a universe behaving like a perfect adiabatic fluid, or a universe whose energy is dominated by a canonically normalised scalar field. In the former case the propagation speed of scalar fluctuations c_s is related to the equation of state parameter as c_s^2=w, while in the latter case c_s^2=1. See the scientific documentation for more details. By setting `cs_equal_one = True` the computation is performed for the canonical scalar field case, while for `cs_equal_one = False` it is the adiabatic perfect fluid result that is computed.

### Primordial scalar power spectrum 

These instructions apply to both SIGWfast.py and SIGWfastEOS.py and concern the block of code after "Configuration" and titled "Primordial scalar power spectrum". The principal input for SIGWfast is the primordial scalar power spectrum P(k). This can be provided as an analytical formula or in terms of numerical data:

* **Analytical formula** (`Num_Pofk = False`): If an analytic scalar power spectrum is to be used as input, this is to be defined here as the function `Pofk(k)`. This should take a single argument which is the wavenumber k and return the corresponding value of P(k). Additional parameters of the power spectrum need to be declared as either global or local variables with given values. To keep the code as general as possible, there are no further restrictions on how `Pofk(k)` is to be defined. As long as calling the function `Pofk(k)` with a float argument returns a float, the script should run without any problems. The default example included with the code is the scalar power spectrum obtained for a strong sharp turn in the inflationary trajectory, see eq. (2.25) in [arXiv:2012.02761](https://arxiv.org/abs/2012.02761). In SIGWfast, `Pofk(k)` is first discretized by evaluating it on an array of k-values, before an interpolation function is then used in the computation. The discretization is performed by evaluating `Pofk(k)` on k-values given in the array `kpzeta` which by default is an extended and denser version of `komega`. If needed, `kpzeta` can be defined here by the user in any other way. For SIGWfast to produce meaningful results, `kpzeta` needs to be sufficiently dense so that the discretization of `Pofk(k)` faithfully captures the relevant features of P(k). To allow the user to check for this, an interpolation of of the discretized P(k) is plotted after every new computation.

* **Numerical data** (`Num_Pofk = True`): If numerical input is to be used for the scalar power spectrum, this is to be provided in a file 'data/'+filenamePz+'.npz'. Here, "filenamePz" refers to the name chosen for this file by the user and which can be declared in step 4 of the configuration. The .npz file should be prepared to contain an array of k-values and an array of corresponding P(k)-values, which should be accessible via the keywords "karray" and "Pzeta", respectively. That is, after loading the data in this file via the command `Pdata  = numpy.load('data/'+filenamePz+'.npz')`, the arrays of k-values and P(k)-values should be given by `Pdata['karray']` and `Pdata['Pzeta']`, respectively.

The script is now be ready to be run!

## Output
For every run the code produces two plots: one of the interpolated scalar power spectrum P(k) and one of the computed gravitational wave spectrum Omega_GW(k). The data for Omega_GW(k) is also saved in 'data/'+filenameGW+'.npz' and can be accessed via the keywords`'karray'` and `'OmegaGW'`. If `regenerate=False` and the file 'data/'+filenameGW+'.npz' exists, Omega_GW(k) is plotted from the data in the file without any new computation.

For the default settings the code is running the python-only routine and on the testing machine (M1 Mac) produced the result in O(1) seconds. 

## Troubleshooting
SIGWfast has been written for python3 and will not work with python2. It has been developed using python 3.9.7 and "conda" for environment and package management, but has also been tested on python 3.8. The development machine was a Macbook Pro with a M1 CPU and running MacOS 12.1 Monterey. Python was installed in its x86 version and was running on the M1 chip via the Rosetta2 translator. SIGWfast has also been tested on Ubuntu 20.04.4 running on a CPU with Intel x86 architecture.

### Compiling the C++ module
One possible source of errors is the compilation of the C++ module. This is activated by setting the flag `Use_Cpp = True` in the block of code titled `Configuration' and its use leads to a 20%-25% reduction in computation times. This option is only available for systems running on Linux and MacOS. When trying to use the C++ option on Windows, the code automatically reverts to the python-only version.

On an older system running python 3.8 on MacOS 10.12 Sierra we encountered the problem that the automatic compilation of the C++ from the code was not initiated. As a result the module "sigwfast" could not be found and the computation ended with an error. To overcome this, the module "sigwfast" can be compiled by hand from the command line. It can then be used indefinitely, as it only has to be compiled only once. To do so, open the terminal and go to the "libraries" subfolder in the parent directory. For definiteness, here we assume that the parent directory "SIGWfast" is located in the home directory `~'. Hence, on the command line enter: 

`cd ~/SIGWfast/libraries`. 

We have to work in this directory so that the file `SIGWfast.cpp` with the C++ code can be found. To compile the module by hand then enter:

`python3 setup.py install --home=~/SIGWfast/libraries`

We used the command `python3` to make sure that python3 is used, as the command `python` can sometimes refer to the version of python2 that is shipped together with MacOS. The flag `--home=...` ensures that the module is deposited within the "libraries" subdirectory, rather than added to the other modules of the python distribution. This makes it easier to remove it later if desired. 

## Licensing

SIGWfast is distributed under the MIT license. You should have received a copy of the MIT License along with SIGWfast. If not, see [https://spdx.org/licenses/MIT.html](https://spdx.org/licenses/MIT.html).

## Acknowledgements

During the development of SIGWfast Lukas T. Witkowski was supported by the European Research Council under the European Union's Horizon 2020 research and innovation programme (grant agreement No 758792, project GEODESI). We are indebted to Dr. Jacopo Fumagalli, without whom SIGWfast would have never been developed in this form and whose inputs vastly improved the code. We are also grateful to Prof. Sebastien Renaux-Petel, whose scientific insights and skilled leadership of the research group made the development of SIGWfast possible. We also thank Dr. John W. Ronayne, whose immense knowledge of python helped get this project off the ground.  
