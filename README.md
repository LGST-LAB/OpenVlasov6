# OpenVlasov6
3D-3V GetFEM-based Vlasov solver

This solver depends on the software package GetFEM (which can be found at getfem.org) to function - specifically its Python interface. Instructions to find and install it can be found at getfem.org/download.html. OpenVlasov6 works on python versions 3.7 to 3.13 and GetFEM version 5.4, and both prebuilt GetFEM interfaces (such as the version of GetFEM 5.4 with Python 3.7 on Anaconda given on getfem.org/download.html) and custom-built ones will work.

main.py is the main file of OpenVlasov6, and is where you should place the code you would like to run. To run the code under set parameters, modify the part of the code that says "Modify this part!"
Vlasov_testing_code_6D.py is the workhorse of the code, and includes the functions that run tests and handle the solve itself for each fluid, multifluid interactions, and the iteration of electromagnetic coupling.
EB_calc.py includes the framework for getting the electromagnetic fields from a plasma density distribution in six-dimensional position/momentum space, which is done  by summing contributions from all parts of the plasma fluid. While this is O(N^6) versus the number of cells on each side, so is the Vlasov simulation itself, and this typically takes only about 1% the time of the actual Vlasov simulation.
params_generator.py builds the parameter structures needed for 6D Vlasov simulation solver. It also generates the value test functions, which are used to test results against analytical solutions.
plotting_6D.py conducts the plotting needed for 6D Vlasov simulation, and has a whole bunch of modes of operation.
