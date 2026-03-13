# OpenVlasov6 1.3.1
3D-3V GetFEM-based Vlasov solver

This solver depends on the software package GetFEM (which can be found at [getfem.org](getfem.org)) to function - specifically its Python interface. Instructions to find and install it can be found at [getfem.org/download.html](getfem.org/download.html). OpenVlasov6 works on python versions 3.7 to 3.13 and GetFEM version 5.4, and both prebuilt GetFEM interfaces (such as the version of GetFEM 5.4 with Python 3.7 on Anaconda given on getfem.org/download.html) and custom-built ones will work.

### Changes in 1.3.1
Removed a for loop in Vlasov_testing_code_6D.py to improve performance.

### Changes in 1.3
Added nonlinear mesh transformations. Use the functions/nonlinear_mesh_transformations.py to make your own, and then add them to the params before execution. Plotting still untransformed, so use sparingly. Updates to update plotting coming soon, and plotting should still work with anything from 1.2.X.

### Changes in 1.2.3
Fixed bug where parallelization always happened, even when unwanted. Use the config/MPI_config.py file to change whether or not it is applied in the code. The default is yes.

### Changes in 1.2.2
Added parallelization of assembly procedures. Make sure to run with the arguments ---mpirun -n 4 python3--- to take advantage of this.

### Changes in 1.2.1
Bugfixes and more comments added.

### Changes in 1.2
Added SUPG hyperbolic stabilization for FEM method.

### Changes in 1.1
Buxfixes, added nonuniform grid generation and plotting and E&B field calculation to support it.

### Changes in 1.0.1
Minor edits and clarifications for comments.

### Changes in 1.0
First public release.

## Installation Instructions

### Installing prerequisites

First, compatible versions of Python and GetFEM must be installed. There are a few ways to do this.

#### Building GetFEM from scratch

This is the best option if you want to use the latest GetFEM and Python features, or if you are not on Windows. To do this, first install the latest version of Python, and then follow the instructions on [getfem.org/install/install_linux.html](getfem.org/install/install_linux.html) to download and install from the GetFEM github repository.

It is recommended to use the configure options
```
--with-blas=openblas --with-optimization=-O2 --with-pic --enable-python --disable-matlab --disable-superlu --enable-mumps --disable-openmp
```
for GetFEM, and this set of options has the best chance of working with OpenVlasov6.

#### Building GetFEM from Windows installable on Anaconda

This is the easiest option, especially for people on Windows machines. Go to [getfem.org/download.html](getfem.org/download.html) and find the Installer of the GetFEM 5.4 interface for 64bits Windows and Python 3.7 of Anaconda 3 (furnished by J.-F. Barthelemy), which can also be found at [download-mirror.savannah.gnu.org/releases/getfem/misc/getfem5.4win-amd64-py3.7.exe](download-mirror.savannah.gnu.org/releases/getfem/misc/getfem5.4win-amd64-py3.7.exe). Install Python 3.7 and Anaconda 3, and then download and run this installer to generate the necessary package interface files.

### Installing OpenVlasov6

OpenVlasov6 is written in Python, so its installation is simply cloning this repository onto your computer, and then opening it using a Python installation with GetFEM (either global, Anaconda, or virtual environment). Then main.py is the main file that you need to run to get the simulation results, and main.py and functions/params_generator.py are the files you will need to modify to run a custom simulation.

## Code structure

### main.py
This is the main file of OpenVlasov6, and is where you should place the code you would like to run. To run the code under set parameters, modify the part of the code that says "Modify this part!"

### config
This folder contains the config file(s) needed to run the code, which can be changed to edit global properties of each simulation, such as parallelization and MPI compatibility.

#### MPI_config.py
This file contains the configuration options for parallelization

### examples
This folder contains example codes that can be run to get used to the solver. 

#### test_code_snippets.txt
This file is a set of code snippets that can be run in the main.py file of OpenVlasov6 to replicate the results of the associated paper.

### functions
This folder contains the main functions that are used to support the OpenVlasov6 solver.

#### EB_calc.py
This file includes the framework for getting the electromagnetic fields from a plasma density distribution in six-dimensional position/momentum space, which is done  by summing contributions from all parts of the plasma fluid. While this is O(N^6) versus the number of cells on each side, so is the Vlasov simulation itself, and this typically takes only about 1% the time of the actual Vlasov simulation.

#### nonlinear_mesh_transformations.py
This file contains functions for applying transformations to the default rectangular mesh. You can add your own functions, which are called by params_generator and must be added to the input parameters for each run.

#### params_generator.py
This file builds the parameter structures needed for 6D Vlasov simulation solver. It also generates the value test functions, which are used to test results against analytical solutions.

This would also be the file where custom-built simulation parameters can be entered (i.e. new boundary conditions, external electromagnetic fields, )

#### plotting_6D.py
This file conducts the plotting needed for 6D Vlasov simulation, and has a whole bunch of modes of operation.

#### Vlasov_testing_code_6D.py
This file is the workhorse of the code, and includes the functions that run tests and handle the solve itself for each fluid, multifluid interactions, and the iteration of electromagnetic coupling.

### getfem
This folder contains a compressed copy of GetFEM, to ensure that the repository stays functional.

#### getfem-5.4.4.tar.gz
This is a compressed copy of GetFEM version 5.4.4. GetFEM also uses the GNU Lesser General Public License in exactly the same way OpenVlasov6 does. This compressed copy includes documentation and license information for GetFEM as well.

## License
This program is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.

## Citation
Academic works using OpenVlasov6 should cite the paper "OpenVlasov6: A 3D-3V Fully Kinetic Multifluid Vlasov Solver," by Eric A. Comstock and Álvaro Romero-Calvo. Additionally, since GetFEM is a requirement, "GetFEM: Automated FE Modeling of Multiphysics Problems Based on a Generic Weak Form Language" by Yves Renard and Konstantinos Poulios should also be cited.

## Questions and inquiries
Please submit any questions or inquiries to [eric.comstock@gatech.edu](eric.comstock@gatech.edu) or [alvaro.romerocalvo@gatech.edu](alvaro.romerocalvo@gatech.edu).
