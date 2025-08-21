This project was started by Nicole Tueni ca. 2019

FEniCS script for computational homogenization of 2D/3D periodic unit cells with two elastic phases. 
Applies elementary macroscopic strains, solves for displacement fluctuations with periodic BCs, computes effective stiffness tensor, and exports displacement/stress fields.


1) Install prerequisites

FEniCS/DOLFIN (2019.x or 2021.x), plus numpy, matplotlib.

Docker → docker run -ti -v $PWD:/home/fenics/shared quay.io/fenicsproject/stable.
