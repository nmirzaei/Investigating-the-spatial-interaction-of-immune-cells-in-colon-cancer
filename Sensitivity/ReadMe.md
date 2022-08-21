This folder contains file required for sensitivity analysis of total cancer cells, total cancer/immune ratio, norm of the velocity and the domain orientation
to the model parameters in presence of immune cells influx.

Corresponding paper: "Investigating the spatial interaction of immune cells in colon cancer"

Code Authors: Navid Mohammad Mirzaei (https://github.com/nmirzaei) (https://sites.google.com/view/nmirzaei)


Requirements:

1- FEniCS version : 2019.2.0.dev0

2- For these folder we need to pyadjoint as well. Since all the sensitivity analyses uses features from this package: (https://www.dolfin-adjoint.org/en/latest/download/index.html)

2- pandas

3- Mesh files are needed for this folder to run. Please download them from "Investigating-the-spatial-interaction-of-immune-cells-in-colon-cancer/Meshes/"

Run order:

Since pyadjoint occupies a lot of memory it is recommended you find the displacements separately. Use the following order:

1- Run Investigating-the-spatial-interaction-of-immune-cells-in-colon-cancer/Colon_Cancer_ImmuneCells_Sources_Inside/Main.py 
   to get the displacement fields. (Notice that this file does not call pyadjoint and it can be done via fenics alone)

2- Now you should have a displacement folder with all the displacement fields created from part 1. Copy it in the folder you run your sensitivity from.

3- Proceed with running the sensitivity codes using a pyadjoint inegrated FEniCS. This step can take a while. 
