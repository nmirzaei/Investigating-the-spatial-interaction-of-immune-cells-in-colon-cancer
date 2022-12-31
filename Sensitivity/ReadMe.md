This folder contains file required for sensitivity analysis of total cancer cells, total cancer/immune ratio, norm of the velocity and the domain orientation
to the model parameters in presence of immune cells influx.

Corresponding paper: "Investigating the spatial interaction of immune cells in colon cancer"

Code Authors: Navid Mohammad Mirzaei (https://github.com/nmirzaei) (https://sites.google.com/view/nmirzaei)


Requirements:

1- FEniCS version : 2019.2.0.dev0

2- For these folder we need to pyadjoint as well. Since all the sensitivity analyses uses features from this package: (https://www.dolfin-adjoint.org/en/latest/download/index.html)

3- pandas, scipy, numpy


Run order:

Since pyadjoint occupies a lot of memory it is recommended you find the displacements separately. Use the following order:

1- Run Investigating-the-spatial-interaction-of-immune-cells-in-colon-cancer/Colon_Cancer_ImmuneCells_Sources_Inside/Main.py 
   to get the displacement fields. (Notice that this file does not call pyadjoint and it can be done via fenics alone)

2- Now you should have a displacement folder with all the displacement fields created from part 1. 
   (Note: For the norm and orientation sensitivity this step is not needed, since they calculate they have to calculate displacement at each step to get
   the measures required for the sensitivity analysis.)

3- Proceed with running the sensitivity codes using a pyadjoint inegrated FEniCS. This step can take a while. 

4- Due to high memory demand the code will suggest a maximum number of iterations. If you are running on a home computer it is recommended to break the runs into much smaller number of iterations (instead of 3000). After each run the code will automatically starts from where you finished the last run.

5- If you decide to break the runs into smaller iterations, due to additive property of the total sensitivity at the end you just need to add up all the calculated sensitivities for each parameter to get the sensitivity for a complete run from t=0 to t=3000.
