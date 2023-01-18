'''
Investigating the spatial interaction of immune cells in colon cancer

xdmffile_creator: Creates XDMF file for saving outputs

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei

(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from dolfin import *
###############################################################

def xdmffile_creator():
    ###############################################################
    # Create XDMFFile files for visualization output
    ###############################################################
    vtkfile_1 = XDMFFile(MPI.comm_world,"reaction_system/Tn.xdmf")
    vtkfile_1.parameters["flush_output"] = True
    vtkfile_2 = XDMFFile(MPI.comm_world,"reaction_system/Th.xdmf")
    vtkfile_2.parameters["flush_output"] = True
    vtkfile_3 = XDMFFile(MPI.comm_world,"reaction_system/Tc.xdmf")
    vtkfile_3.parameters["flush_output"] = True
    vtkfile_4 = XDMFFile(MPI.comm_world,"reaction_system/Tr.xdmf")
    vtkfile_4.parameters["flush_output"] = True
    vtkfile_5 = XDMFFile(MPI.comm_world,"reaction_system/Dn.xdmf")
    vtkfile_5.parameters["flush_output"] = True
    vtkfile_6 = XDMFFile(MPI.comm_world,"reaction_system/D.xdmf")
    vtkfile_6.parameters["flush_output"] = True
    vtkfile_7 = XDMFFile(MPI.comm_world,"reaction_system/M.xdmf")
    vtkfile_7.parameters["flush_output"] = True
    vtkfile_8 = XDMFFile(MPI.comm_world,"reaction_system/C.xdmf")
    vtkfile_8.parameters["flush_output"] = True
    vtkfile_9 = XDMFFile(MPI.comm_world,"reaction_system/N.xdmf")
    vtkfile_9.parameters["flush_output"] = True
    vtkfile_10 = XDMFFile(MPI.comm_world,"reaction_system/H.xdmf")
    vtkfile_10.parameters["flush_output"] = True
    vtkfile_11 = XDMFFile(MPI.comm_world,"reaction_system/mu1.xdmf")
    vtkfile_11.parameters["flush_output"] = True
    vtkfile_12 = XDMFFile(MPI.comm_world,"reaction_system/mu2.xdmf")
    vtkfile_12.parameters["flush_output"] = True
    vtkfile_13 = XDMFFile(MPI.comm_world,"reaction_system/Igamma.xdmf")
    vtkfile_13.parameters["flush_output"] = True
    vtkfile_14 = XDMFFile(MPI.comm_world,"reaction_system/Gbeta.xdmf")
    vtkfile_14.parameters["flush_output"] = True
    vtkfile_15 = XDMFFile(MPI.comm_world,"reaction_system/Total_Cells.xdmf")
    vtkfile_15.parameters["flush_output"] = True
    vtkfile_16 = XDMFFile(MPI.comm_world,"reaction_system/rhs_mech.xdmf")
    vtkfile_16.parameters["flush_output"] = True
    vtkfile_17 = XDMFFile(MPI.comm_world,"reaction_system/total_Tn.xdmf")
    vtkfile_17.parameters["flush_output"] = True
    vtkfile_18 = XDMFFile(MPI.comm_world,"reaction_system/total_Th.xdmf")
    vtkfile_18.parameters["flush_output"] = True
    vtkfile_19 = XDMFFile(MPI.comm_world,"reaction_system/total_Tc.xdmf")
    vtkfile_19.parameters["flush_output"] = True
    vtkfile_20 = XDMFFile(MPI.comm_world,"reaction_system/total_Tr.xdmf")
    vtkfile_20.parameters["flush_output"] = True
    vtkfile_21 = XDMFFile(MPI.comm_world,"reaction_system/total_Dn.xdmf")
    vtkfile_21.parameters["flush_output"] = True
    vtkfile_22 = XDMFFile(MPI.comm_world,"reaction_system/total_D.xdmf")
    vtkfile_22.parameters["flush_output"] = True
    vtkfile_23 = XDMFFile(MPI.comm_world,"reaction_system/total_M.xdmf")
    vtkfile_23.parameters["flush_output"] = True
    vtkfile_24 = XDMFFile(MPI.comm_world,"reaction_system/total_C.xdmf")
    vtkfile_24.parameters["flush_output"] = True
    vtkfile_25 = XDMFFile(MPI.comm_world,"reaction_system/total_N.xdmf")
    vtkfile_25.parameters["flush_output"] = True
    vtkfile_26 = XDMFFile(MPI.comm_world,"reaction_system/curvature.xdmf")
    vtkfile_26.parameters["flush_output"] = True
    vtkfile_27 = XDMFFile(MPI.comm_world,"reaction_system/normal.xdmf")
    vtkfile_27.parameters["flush_output"] = True
    vtkfile_28 = XDMFFile(MPI.comm_world,"reaction_system/u.xdmf")
    vtkfile_28.parameters["flush_output"] = True
    ##############################################################
    return [vtkfile_1,vtkfile_2,vtkfile_3,vtkfile_4,vtkfile_5,vtkfile_6,vtkfile_7,vtkfile_8,vtkfile_9,\
    vtkfile_10,vtkfile_11,vtkfile_12,vtkfile_13,vtkfile_14,vtkfile_15,vtkfile_16,vtkfile_17,vtkfile_18,vtkfile_19,\
    vtkfile_20,vtkfile_21,vtkfile_22,vtkfile_23,vtkfile_24,vtkfile_25,vtkfile_26,vtkfile_27,vtkfile_28]
