'''
Investigating the spatial interaction of immune cells in colon cancer

**This code is used to convert .xml mesh files to .xmdf mesh files**
1- We first save the mesh build by GMSH in the file Mesh.msh
2- Then we conver the .msh into .xml through the terminal command:
    dolfin-convert Mesh.msh Mesh.xml
3- Step 2 creates 3 files: Mesh.xml, Mesh_physical_region.xml, Mesh_facet_region.xml
    used in this code

    
Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei

(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from fenics import *
###############################################################

###############################################################
#Reading the .xml files
###############################################################
mesh = Mesh("Mesh.xml")
Volume = MeshFunction("size_t", mesh, "Mesh_physical_region.xml")
bnd_mesh = MeshFunction("size_t", mesh, "Mesh_facet_region.xml")
###############################################################

###############################################################
#Converting to .xmdf and Saving
###############################################################
xdmf = XDMFFile(mesh.mpi_comm(),"Mesh.xdmf")
xdmf.write(mesh)
xdmf.write(Volume)
xdmf = XDMFFile(mesh.mpi_comm(),"boundaries.xdmf")
xdmf.write(bnd_mesh)
xdmf.close()
###############################################################
