'''
Investigating the spatial interaction of immune cells in colon cancer
**The main script**
GeoMaker: Remeshing Function

Curvature: Calculates Curvature and Normal vectors of a given domain
            Courtesy of http://jsdokken.com/

qspmodel: Calculates the ODE parameters.
            Courtesy of Arkadz Kirshtein:
https://github.com/ShahriyariLab/Data-driven-mathematical-model-for-colon-cancer

Source_Loc_rnd: Creates random sources for immune cells

RHS_sum: Calculates the right hand side of the ODE system

xdmffile_creator: Creates the XDMF outputs

check_dependencies: Checks for dependencies and installs them if missing

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei
(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from dolfin import *
import logging
import scipy.optimize as op
import pandas as pd
from subprocess import call
from GeoMaker import *
from Curvature import *
from qspmodel import *
from Source_Loc_rnd import *
from RHS_sum import *
from xdmffile_creator import *
from check_dependencies import *
from ufl import Min
import os
###############################################################

###############################################################
#Check required packages
###############################################################
check_dependencies()
###############################################################

###############################################################
#Solver parameters
###############################################################
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}
###############################################################

###############################################################
#time step variables
###############################################################
T = 3002            # final
num_steps= 3002   # number of time steps
dt = T / num_steps  # time step
eps = 1             # diffusion coefficient
t=0                 # initial time
k = Constant(dt)    # Constant time step object for the weak formulation
###############################################################

###############################################################
#Remeshing info
###############################################################
#For no refinement set refine=1 for refinement set the intensity >1
#Org_size is the original element size of your mesh. Can be extracted from the .geo mesh files
#Max_cellnum and Min_cellnum will assure that after remeshing the number of new cells are in [Min_cellnum,Max_cellnum]
#remesh_step is the step when remeshing is initiated.
prompt1 = input('Enter a positive integer as the remeshing_step number. If you do not want any remeshing (NOT RECOMMENDED) enter 0:')
if int(prompt1)>0:
    refine = 1
    Max_cellnum = 2700
    Min_cellnum = 2600
    Refine = str(refine)
    Org_size = 0.028
    prompt2 = input('Enter a positive integer for the refinement intensity. (No refinement (default)=1, Refinement>1)')
    prompt3 = input('What is the minimum number of cells allowable after remeshing? (default = 2600)')
    prompt4 = input('What is the maximum number of cells allowable after remeshing? (default = 2700)')
    prompt5 = input('What is the original size of your mesh cells. User should refer to the mesh .geo file. (default=0.028)')
    refine = int(prompt2)
    Max_cellnum = int(prompt4)
    Min_cellnum = int(prompt3)
    Refine = str(refine)
    Org_size = float(prompt5)
    remesh_step = int(prompt1)
elif int(prompt1)==0:
    remesh_step = num_steps+10
else:
    print("Error: Unacceptable input.")
    exit()
###############################################################

###############################################################
#Reporting options
###############################################################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rothemain.rothe_utils")
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)
###############################################################




###############################################################
# Load mesh
###############################################################
#Parallel compatible Mesh readings
mesh= Mesh()
xdmf = XDMFFile(mesh.mpi_comm(), "Random_Mesh/Mesh.xdmf")
xdmf.read(mesh)
mvc = MeshValueCollection("size_t", mesh, 2)
with XDMFFile("Random_Mesh/Mesh.xdmf") as infile:
    infile.read(mvc, "f")
Volume = cpp.mesh.MeshFunctionSizet(mesh, mvc)
xdmf.close()
mvc2 = MeshValueCollection("size_t", mesh, 1)
with XDMFFile("Random_Mesh/boundaries.xdmf") as infile:
    infile.read(mvc2, "f")
bnd_mesh = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
###############################################################

###############################################################
#Reference Mesh
###############################################################
mesh0 = Mesh('Random_Mesh/Mesh.xml')
Volume0 = MeshFunction('size_t' , mesh0 , 'Random_Mesh/Mesh_physical_region.xml' )  #saves the interior info of the mesh
bnd_mesh0 = MeshFunction('size_t', mesh0 , 'Random_Mesh/Mesh_facet_region.xml')  #saves the boundary info of the mesh
VV0 = VectorFunctionSpace(mesh0,'Lagrange',1)
SS0 = FunctionSpace(mesh0,'Lagrange',1)
mesh0.bounding_box_tree().build(mesh0)
###############################################################


###############################################################
# Nullspace of rigid motions
###############################################################
d = mesh.geometry().dim()
if d==3:
    #Translation in 3D
    Z_transl = [Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0, 1))]

    #Rotations 3D
    Z_rot = [Expression(('0', 'x[2]', '-x[1]')),
            Expression(('-x[2]', '0', 'x[0]')),
            Expression(('x[1]', '-x[0]', '0'))]
elif d==2:
    #Translation 2D.
    Z_transl = [Constant((1, 0)), Constant((0, 1))]

    #Rotations 2D.
    Z_rot = [Expression(('-x[1]', 'x[0]'),degree=0)]

else:
    print("Mesh is not set up appropriately!")
    exit()

Z = Z_transl + Z_rot
###############################################################

###############################################################
#Saving the mesh files
###############################################################
File('Results/mesh.pvd')<<mesh
File('Results/Volume.pvd')<<Volume
File('Results/boundary.pvd')<<bnd_mesh
###############################################################

###############################################################
# Build function spaces
###############################################################
#Mechanical problem
P22 = VectorElement("P", mesh.ufl_cell(), 4)
P11 = FiniteElement("P", mesh.ufl_cell(), 1)
P00 = VectorElement("R", mesh.ufl_cell(), 0,dim=4)
TH = MixedElement([P22,P11,P00])
W = FunctionSpace(mesh, TH)

#Biological problem
#Nodal Enrichment is done for more stability
P1 = FiniteElement('P', triangle,1)
PB = FiniteElement('B', triangle,3)
NEE = NodalEnrichedElement(P1, PB)
element = MixedElement([NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE])
Mixed_Space = FunctionSpace(mesh, element)

#Auxillary spaces for projection and plotting
SDG = FunctionSpace(mesh,'DG',0)
S1 = FunctionSpace(mesh,'P',1)
VV = VectorFunctionSpace(mesh,'Lagrange',4)
VV1 = VectorFunctionSpace(mesh,'Lagrange',1)
R = FunctionSpace(mesh,'R',0)
###############################################################

###############################################################
#Defining functions and test functions
###############################################################
U = Function(Mixed_Space)
U_n = Function(Mixed_Space)
Tn, Th, Tc, Tr, Dn, D, M, C, N, H, mu1, mu2, Igamma, Gbeta = split(U)
Tn_n, Th_n, Tc_n, Tr_n, Dn_n, D_n, M_n, C_n, N_n, H_n, mu1_n, mu2_n, Igamma_n, Gbeta_n = split(U_n)
v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14 = TestFunctions(Mixed_Space)
u_n = Function(VV)
UU = Function(W)
u_, p_, lambda_ =split(UU)
u__ = Function(VV1)
displ = Function(VV1)
###############################################################

###############################################################
# Construct integration measures using these markers
###############################################################
ds = Measure('ds', subdomain_data=bnd_mesh)
dx = Measure('dx', subdomain_data=Volume)
###############################################################

###############################################################
#PDE Parameters (dimensional)
###############################################################
#They are all in cm^2/day.
D_cell, D_H, D_cyto =  8.64e-6, 7.92e-2, 1.24e-3
coeff = Constant(1)    #advection constant taken to be one for this problem
##############################################################

###############################################################
#Parameters names
###############################################################
Pars = ['lambda_{T_hD}',  'lambda_{T_hM}',  'lambda_{T_hmu_1}', 'lambda_{T_CT_h}','lambda_{T_CD}', 'lambda_{T_rT_h}', 'lambda_{T_rmu_2}','lambda_{T_rG_beta}',
                                      'lambda_{DH}',   'lambda_{DC}',     'lambda_{Mmu_2}',   'lambda_{MI_gamma}', 'lambda_{MT_h}', 'lambda_{C}',    'lambda_{Cmu_1}', 'alpha_{NC}',
                                      'lambda_{mu_1T_h}','lambda_{mu_1M}',   'lambda_{mu_1D}',   'lambda_{mu_2M}','lambda_{mu_2D}','lambda_{mu_2T_r}',
                                      'lambda_{HN}',   'lambda_{HM}',     'lambda_{HT_h}',    'lambda_{HT_C}', 'lambda_{HT_r}', 'lambda_{I_gammaT_h}', 'lambda_{I_gammaT_C}', 'lambda_{I_gammaM}',
                                      'lambda_{G_betaM}',  'lambda_{G_betaT_r}',   'delta_{T_N}',      'delta_{T_hmu_2}','delta_{T_hT_r}', 'delta_{T_h}',    'delta_{T_Cmu_2}', 'delta_{T_CT_r}',
                                      'delta_{T_C}',    'delta_{T_rmu_1}',   'delta_{T_r}',      'delta_{DH}',   'delta_{DC}',   'delta_{D}',     'delta_{M}',     'delta_{CG_beta}','delta_{CI_gamma}','delta_{CT_C}',
                                      'delta_{C}',     'delta_{N}',       'delta_{mu_1}',     'delta_{mu_2}',  'delta_{H}',     'delta_{I_gamma}',    'delta_{G_beta}',
                                      'A_{T_N}','A_{Dn}','M_0','C_0', 'alpha_{T_NT_h}', 'alpha_{T_NT_C}', 'alpha_{T_NT_r}', 'alpha_{D_ND}']
###############################################################

###############################################################
# Calculating parameter values
###############################################################
Input2 = 0  #The patient's number. In this paper we only used Cluster 1 of patients
clustercells=pd.read_csv('input/input/Large_Tumor_cell_data.csv').to_numpy()
QSP_=QSP.from_cell_data(clustercells[Input2])
params=QSP_.par
pars = {Pars[k]:value for k,value in zip(range(len(Pars)),params)}

answer = input('What type of macrophages? (Pro-Tumor=1, Anti-tumor=2, Mixed=3)')

if int(answer)==1:
    pars['lambda_{G_betaM}'] = 0
    pars['lambda_{I_gammaM}'] = 0
elif int(answer)==2:
    pars['lambda_{mu_1M}'] = 0
elif int(answer)==3:
    print('The original set of parameters used.')
else:
    print('Error: Wrong input!')
    exit()
###############################################################

###############################################################
#Initial conditions
###############################################################
Source1,Source2,Source3,Source4,Source5,Source6,U_n = Source_Loc_rnd(mesh,Mixed_Space,SDG,S1,U_n,Input2)
Source = [Source1,Source2,Source3,Source4,Source5,Source6]
Tn_n, Th_n, Tc_n, Tr_n, Dn_n, D_n, M_n, C_n, N_n, H_n, mu1_n, mu2_n, Igamma_n, Gbeta_n= U_n.split()
##############################################################

##############################################################
#VTK file array for saving plots
##############################################################
vtkfile = xdmffile_creator()
##############################################################

###############################################################
#Sum of RHS
###############################################################
RHS = RHS_sum(U_n,Source,pars)
RHS_MECH_ = project(RHS,S1)
##############################################################

#######################################################################
#Mesh and remeshing info
#######################################################################
numCells = mesh.num_cells()
mesh.smooth(100)
Counter=0
#######################################################################

#######################################################################
#loop parameters
#######################################################################
t = 0.0
j = int(0)
#######################################################################

#######################################################################
#Curvature and Normal vector for the initial domain
#######################################################################
crvt1, NORMAL1 = Curvature(mesh)
#######################################################################


for n in range(num_steps):
     ##############################################################
     #First we plot the ICs and then solve.
     ##############################################################
     if j>=1:
         #############################################################
         if j>=2:
             ##############################################################
             #constructing mechanical problem based on the updated RHS, curvature and Normal vector.
             ##############################################################
             mu = 1
             RHS = RHS_sum(U_n,Source,pars)
             RHS_MECH_ = project(RHS,S1)
             (u1, p1, l1) = TrialFunctions(W)
             (v0, q0, w0) = TestFunctions(W)
             I = Identity(2)
             Sigma = -p1*I + 2*mu*sym(grad(u1))
             stokes1 = inner(Sigma,grad(v0))*dx  + (div(u1)-k*(0.01)*RHS_MECH_)*q0*dx -sum(l1[i]*inner(v0, Z[i])*dx for i in range(len(Z)))-sum(w0[i]*inner(u1, Z[i])*dx for i in range(len(Z)))+0.0864*dot(crvt1*NORMAL1,v0)*ds(1)  #0.0001 in mm^2/s is 0.0864 cm^2/day
             a1 = lhs(stokes1)
             L1 = rhs(stokes1)
             solve(a1==L1,UU)
             u_, p_,lambda_ = UU.split()
             ##############################################################

             ##############################################################
             #Create displacement for mesh movement. Moving happens from the current configuration
             #Also saving displacements for the reference domain to be used in the sensitvity analysis
             ##############################################################
             u__ = project(u_/k,VV)
             displ = project(u_,VV1)
             displ.set_allow_extrapolation(True)
             displ0 = project(displ,VV0)
             path = os.path.realpath(__file__)
             dir = os.path.dirname(path)
             folder = os.path.basename(dir)
             dir = dir.replace(folder, 'Sensitivity/Cancer_and_CancerImmuneRatio_sensitivity')
             os.chdir(dir)
             File("displacement/u%d.xml" %(t))<<displ0
             dir = dir.replace('Sensitivity/Cancer_and_CancerImmuneRatio_sensitivity',folder)
             os.chdir(dir)
             ALE.move(mesh0,displ0)
             ALE.move(mesh,displ)
             mesh0.bounding_box_tree().build(mesh0)
             mesh.bounding_box_tree().build(mesh)
             #############################################################

             ##############################################################
             #Updating the curvature and normal vectors for the current configuration
             ##############################################################
             crvt1, NORMAL1 = Curvature(mesh)
             ##############################################################

         ##############################################################
         #Loop info update and printing
         ##############################################################
         print(t,flush=True)
         t+=dt
         ##############################################################

         #Update biology PDE and solve
         F1 = ((Tn-Tn_n)/k)*v1*dx-(pars['A_{T_N}']-pars['alpha_{T_NT_h}']*(pars['lambda_{T_hD}']*D+pars['lambda_{T_hM}']*M+pars['lambda_{T_hmu_1}']*mu1)*Tn-pars['alpha_{T_NT_C}']*(pars['lambda_{T_CT_h}']*Th+pars['lambda_{T_CD}']*D)*Tn-pars['alpha_{T_NT_r}']*(pars['lambda_{T_rT_h}']*Th+pars['lambda_{T_rmu_2}']*mu2+pars['lambda_{T_rG_beta}']*Gbeta)*Tn-pars['delta_{T_N}']*Tn)*v1*dx\
         + ((Th-Th_n)/k)*v2*dx+D_cell*dot(grad(Th),grad(v2))*dx+coeff*Th*div(u__)*v2*dx-((pars['lambda_{T_hD}']*D+pars['lambda_{T_hM}']*M+pars['lambda_{T_hmu_1}']*mu1)*Tn-(pars['delta_{T_hmu_2}']*mu2+pars['delta_{T_hT_r}']*Tr+pars['delta_{T_h}'])*Th)*v2*dx-Source1*v2*dx\
         + ((Tc-Tc_n)/k)*v3*dx+D_cell*dot(grad(Tc),grad(v3))*dx+coeff*Tc*div(u__)*v3*dx-((pars['lambda_{T_CT_h}']*Th+pars['lambda_{T_CD}']*D)*Tn-(pars['delta_{T_Cmu_2}']*mu2+pars['delta_{T_CT_r}']*Tr+pars['delta_{T_C}'])*Tc)*v3*dx-Source2*v3*dx\
         + ((Tr-Tr_n)/k)*v4*dx+D_cell*dot(grad(Tr),grad(v4))*dx+coeff*Tr*div(u__)*v4*dx-((pars['lambda_{T_rT_h}']*Th+pars['lambda_{T_rmu_2}']*mu2+pars['lambda_{T_rG_beta}']*Gbeta)*Tn-(pars['delta_{T_rmu_1}']*mu1+pars['delta_{T_r}'])*Tr)*v4*dx-Source3*v4*dx\
         + ((Dn-Dn_n)/k)*v5*dx+D_cell*dot(grad(Dn),grad(v5))*dx+coeff*Dn*div(u__)*v5*dx-(pars['A_{Dn}']-pars['alpha_{D_ND}']*(pars['lambda_{DH}']*H+pars['lambda_{DC}']*C)*Dn-(pars['delta_{DH}']*H+pars['delta_{D}'])*Dn)*v5*dx-Source4*v5*dx\
         + ((D-D_n)/k)*v6*dx+D_cell*dot(grad(D),grad(v6))*dx+coeff*D*div(u__)*v6*dx-((pars['lambda_{DH}']*H+pars['lambda_{DC}']*C)*Dn-(pars['delta_{DH}']*H+pars['delta_{DC}']*C+pars['delta_{D}'])*D)*v6*dx-Source5*v6*dx\
         + ((M-M_n)/k)*v7*dx+D_cell*dot(grad(M),grad(v7))*dx+coeff*M*div(u__)*v7*dx-((pars['lambda_{Mmu_2}']*mu2+pars['lambda_{MI_gamma}']*Igamma+pars['lambda_{MT_h}']*Th)*(pars['M_0']-M)-pars['delta_{M}']*M)*v7*dx-Source6*v7*dx\
         + ((C-C_n)/k)*v8*dx+D_cell*dot(grad(C),grad(v8))*dx+coeff*C*div(u__)*v8*dx-((pars['lambda_{C}']+pars['lambda_{Cmu_1}']*mu1)*C*(Constant(1)-C/pars['C_0'])-(pars['delta_{CG_beta}']*Gbeta+pars['delta_{CI_gamma}']*Igamma+pars['delta_{CT_C}']*Tc+pars['delta_{C}'])*C)*v8*dx\
         + ((N-N_n)/k)*v9*dx+D_cell*dot(grad(N),grad(v9))*dx+coeff*N*div(u__)*v9*dx-(pars['alpha_{NC}']*(pars['delta_{CG_beta}']*Gbeta+pars['delta_{CI_gamma}']*Igamma+pars['delta_{CT_C}']*Tc+pars['delta_{C}'])*C-pars['delta_{N}']*N)*v9*dx\
         + ((H-H_n)/k)*v10*dx+D_H*dot(grad(H),grad(v10))*dx-(pars['lambda_{HN}']*N+pars['lambda_{HM}']*M+pars['lambda_{HT_h}']*Th+pars['lambda_{HT_C}']*Tc+pars['lambda_{HT_r}']*Tr-pars['delta_{H}']*H)*v10*dx\
         + ((mu1-mu1_n)/k)*v11*dx+D_cyto*dot(grad(mu1),grad(v11))*dx-(pars['lambda_{mu_1T_h}']*Th+pars['lambda_{mu_1M}']*M+pars['lambda_{mu_1D}']*D-pars['delta_{mu_1}']*mu1)*v11*dx\
         + ((mu2-mu2_n)/k)*v12*dx+D_cyto*dot(grad(mu2),grad(v12))*dx-(pars['lambda_{mu_2M}']*M+pars['lambda_{mu_2D}']*D+pars['lambda_{mu_2T_r}']*Tr-pars['delta_{mu_2}']*mu2)*v12*dx\
         + ((Igamma-Igamma_n)/k)*v13*dx+D_cyto*dot(grad(Igamma),grad(v13))*dx-(pars['lambda_{I_gammaT_h}']*Th+pars['lambda_{I_gammaT_C}']*Tc+pars['lambda_{I_gammaM}']*M-pars['delta_{I_gamma}']*Igamma)*v13*dx\
         + ((Gbeta-Gbeta_n)/k)*v14*dx+D_cyto*dot(grad(Gbeta),grad(v14))*dx-(pars['lambda_{G_betaM}']*M+pars['lambda_{G_betaT_r}']*Tr-pars['delta_{G_beta}']*Gbeta)*v14*dx

         bc = []
         solve(F1==0,U,bc)
         ##############################################################

         ##############################################################
         #Making a copy of subfunctions
         ##############################################################
         Tn_, Th_, Tc_, Tr_, Dn_, D_, M_, C_, N_, H_, mu1_, mu2_, Igamma_, Gbeta_= U.split()
         ##############################################################

         ##############################################################
         #Saving info of the previous time step
         ##############################################################
         U_n.assign(U)
         Tn_n, Th_n, Tc_n, Tr_n, Dn_n, D_n, M_n, C_n, N_n, H_n, mu1_n, mu2_n, Igamma_n, Gbeta_n= U_n.split()
         #######################################################################


     #######################################################################
     #Plotting the dynamics every 10 steps and at the beginning
     #######################################################################
     if j%10==0 or j==1:

           #######################################################################
           #Renaming is crucial for animations in Paraview
           #######################################################################
           Tn_n.rename('Tn_n','Tn_n')
           Th_n.rename('Th_n','Th_n')
           Tc_n.rename('Tc_n','Tc_n')
           Tr_n.rename('Tr_n','Tr_n')
           Dn_n.rename('Dn_n','Dn_n')
           D_n.rename('D_n','D_n')
           M_n.rename('M_n','M_n')
           C_n.rename('C_n','C_n')
           N_n.rename('N_n','N_n')
           H_n.rename('H_n','H_n')
           mu1_n.rename('mu1_n','mu1_n')
           mu2_n.rename('mu2_n','mu2_n')
           Igamma_n.rename('Igamma_n','Igamma_n')
           Gbeta_n.rename('Gbeta_n','Gbeta_n')
           u__.rename('u__','u__')
           Total_Cells_ = project(Th_n+Tc_n+Tr_n+Dn_n+D_n+M_n+C_n+N_n,S1)
           Total_Cells_.rename('Total_Cells_','Total_Cells_')
           RHS_MECH_.rename('RHS_MECH','RHS_MECH')
           CRVT = project(crvt1,S1)
           NORM = project(NORMAL1,VV1)
           CRVT.rename('CRVT','CRVT')
           NORM.rename('NORM','NORM')
           #######################################################################

           #######################################################################
           #Writting
           #######################################################################
           vtkfile[0].write(Tn_n,t)
           vtkfile[1].write(Th_n,t)
           vtkfile[2].write(Tc_n,t)
           vtkfile[3].write(Tr_n,t)
           vtkfile[4].write(Dn_n,t)
           vtkfile[5].write(D_n,t)
           vtkfile[6].write(M_n,t)
           vtkfile[7].write(C_n,t)
           vtkfile[8].write(N_n,t)
           vtkfile[9].write(H_n,t)
           vtkfile[10].write(mu1_n,t)
           vtkfile[11].write(mu2_n,t)
           vtkfile[12].write(Igamma_n,t)
           vtkfile[13].write(Gbeta_n,t)
           vtkfile[14].write(Total_Cells_,t)
           vtkfile[15].write(RHS_MECH_,t)
           vtkfile[25].write(CRVT,t)
           vtkfile[26].write(NORM,t)
           vtkfile[27].write(u__,t)
           ##############################################################

           ##############################################################
           #Plotting the integrals every 10 steps
           ##############################################################
           if j%10==0:
                Tn_total =assemble(Tn_n*dx)
                Tn_total_ = project(Tn_total,R)
                try:
                    Th_total =assemble(Th_n*dx)
                    Th_total_ = project(Th_total,R)
                except:
                    Th_total_ = project(Constant(0.0),R)
                try:
                    Tc_total =assemble(Tc_n*dx)
                    Tc_total_ = project(Tc_total,R)
                except:
                    Tc_total_ = project(Constant(0.0),R)
                try:
                    Tr_total =assemble(Tr_n*dx)
                    Tr_total_ = project(Tr_total,R)
                except:
                    Tr_total_ = project(Constant(0.0),R)
                try:
                    Dn_total =assemble(Dn_n*dx)
                    Dn_total_ = project(Dn_total,R)
                except:
                    Dn_total_ = project(Constant(0.0),R)
                try:
                    D_total =assemble(D_n*dx)
                    D_total_ = project(D_total,R)
                except:
                    D_total_ = project(Constant(0.0),R)
                try:
                    M_total =assemble(M_n*dx)
                    M_total_ = project(M_total,R)
                except:
                    M_total_ = project(Constant(0.0),R)
                C_total =assemble(C_n*dx)
                C_total_ = project(C_total,R)
                try:
                    N_total =assemble(N_n*dx)
                    N_total_ = project(N_total,R)
                except:
                    N_total_ = project(Constant(0.0),R)

                Tn_total_.rename('Tn_total_','Tn_total_')
                Th_total_.rename('Th_total_','Th_total_')
                Tc_total_.rename('Tc_total_','Tc_total_')
                Tr_total_.rename('Tr_total_','Tr_total_')
                Dn_total_.rename('Dn_total_','Dn_total_')
                D_total_.rename('D_total_','D_total_')
                M_total_.rename('M_total_','M_total_')
                C_total_.rename('C_total_','C_total_')
                N_total_.rename('N_total_','N_total_')

                vtkfile[16].write(Tn_total_,t)
                vtkfile[17].write(Th_total_,t)
                vtkfile[18].write(Tc_total_,t)
                vtkfile[19].write(Tr_total_,t)
                vtkfile[20].write(Dn_total_,t)
                vtkfile[21].write(D_total_,t)
                vtkfile[22].write(M_total_,t)
                vtkfile[23].write(C_total_,t)
                vtkfile[24].write(N_total_,t)
            ##############################################################

     ##############################################################
     #Update loop info
     ##############################################################
     j+=1
     ##############################################################

     ##############################################################
     #Remeshing
     ##############################################################
     if j%remesh_step==0 or j==1:

         ###############################################################
         #Initializing the meshsize
         ###############################################################
         if j==1:
             MeshSize = Org_size
         ##############################################################

         ##############################################################
         #Mesh loop counter update
         ##############################################################
         Counter+=1
         ##############################################################

         ##############################################################
         #Remeshing
         ##############################################################
         GeoMaker(MeshSize,mesh,'Mesh1',Refine,Counter)
         mesh = Mesh("Mesh1.xml")
         numCells = mesh.num_cells()
         print(numCells)
         ##############################################################

         ##############################################################
         #Remesh until it satisfies the desired range of cells [Min_cellnum,Max_cellnum]
         ##############################################################
         while numCells > int(Max_cellnum):
             MeshSize+= MeshSize/100
             GeoMaker(MeshSize,mesh,'Mesh1',Refine,Counter)
             mesh = Mesh("Mesh1.xml")
             numCells = mesh.num_cells()
             print(numCells)
         while (numCells < int(Min_cellnum)) and (numCells < int(Max_cellnum)):
             MeshSize-= MeshSize/200
             GeoMaker(MeshSize,mesh,'Mesh1',Refine,Counter)
             mesh = Mesh("Mesh1.xml")
             numCells = mesh.num_cells()
             print(numCells)
         ##############################################################

         ##############################################################
         #Make the new mesh compatible for Fenics use
         ##############################################################
         Volume = MeshFunction("size_t", mesh, "Mesh1_physical_region.xml")
         bnd_mesh = MeshFunction("size_t", mesh, "Mesh1_facet_region.xml")
         xdmf = XDMFFile(mesh.mpi_comm(),"Mesh1.xdmf")
         xdmf.write(mesh)
         xdmf.write(Volume)
         xdmf = XDMFFile(mesh.mpi_comm(),"boundaries1.xdmf")
         xdmf.write(bnd_mesh)
         xdmf.close()
         mesh = Mesh()
         xdmf = XDMFFile(mesh.mpi_comm(), "Mesh1.xdmf")
         xdmf.read(mesh)
         mvc = MeshValueCollection("size_t", mesh, 2)
         with XDMFFile("Mesh1.xdmf") as infile:
             infile.read(mvc, "f")
         Volume = cpp.mesh.MeshFunctionSizet(mesh, mvc)
         xdmf.close()
         mvc2 = MeshValueCollection("size_t", mesh, 1)
         with XDMFFile("boundaries1.xdmf") as infile:
             infile.read(mvc2, "f")
         bnd_mesh = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
         ###############################################################

         ##############################################################
         # Build function space on the new mesh
         ##############################################################
         #Mechanical problem
         P22 = VectorElement("P", mesh.ufl_cell(), 2)
         P11 = FiniteElement("P", mesh.ufl_cell(), 1)
         P00 = VectorElement("R", mesh.ufl_cell(), 0,dim=4)
         TH = MixedElement([P22,P11,P00])
         W = FunctionSpace(mesh, TH)
         #Biological problem
         #Nodal Enrichment is done for more stability
         P1 = FiniteElement('P', triangle,1)
         PB = FiniteElement('B', triangle,3)
         NEE = NodalEnrichedElement(P1, PB)
         element = MixedElement([NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE,NEE])
         Mixed_Space = FunctionSpace(mesh, element)
         #Auxillary spaces for projection and plotting
         S1 = FunctionSpace(mesh,'P',1)
         VV = VectorFunctionSpace(mesh,'Lagrange',4)
         VV1 = VectorFunctionSpace(mesh,'Lagrange',1)
         R = FunctionSpace(mesh,'R',0)
         ##############################################################################################

         ###############################################################
         #Defining functions and test functions on the new mesh
         #We interpolate our already acquired results using the new mesh functions
         ###############################################################
         U = Function(Mixed_Space)
         U_n.set_allow_extrapolation(True)
         U_n = interpolate(U_n,Mixed_Space)
         Tn, Th, Tc, Tr, Dn, D, M, C, N, H, mu1, mu2, Igamma, Gbeta = split(U)
         Tn_n, Th_n, Tc_n, Tr_n, Dn_n, D_n, M_n, C_n, N_n, H_n, mu1_n, mu2_n, Igamma_n, Gbeta_n = U_n.split()
         v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14 = TestFunctions(Mixed_Space)
         UU.set_allow_extrapolation(True)
         UU = interpolate(UU,W)
         u_, p_ ,lambda_=split(UU)
         Th_R, Tc_R, Tr_R, Dn_R, D_R, M_R= project(Constant(1),S1),project(Constant(1),S1),project(Constant(1),S1),project(Constant(1),S1),project(Constant(1),S1),project(Constant(1),S1)
         u__ = project(u_/k,VV)
         ###############################################################

         #######################################################################
         #Updating curvature and normal vector for the new mesh
         #######################################################################
         crvt1, NORMAL1 = Curvature(mesh)
         #######################################################################

         #######################################################################
         # Construct integration measures using these markers for the new mesh
         #######################################################################
         ds = Measure('ds', subdomain_data=bnd_mesh)
         dx = Measure('dx', subdomain_data=Volume)
         #######################################################################

         ###############################################################
         #Updating the source locations for the new mesh. Since the sources need to move with the mesh
         ###############################################################
         Source1.set_allow_extrapolation(True)
         Source2.set_allow_extrapolation(True)
         Source3.set_allow_extrapolation(True)
         Source4.set_allow_extrapolation(True)
         Source5.set_allow_extrapolation(True)
         Source6.set_allow_extrapolation(True)

         Source1 = interpolate(Source1,S1)
         i_source1=np.argwhere(Source1.vector().get_local()[:]<=0)  #making negatives zero
         Source1.vector()[i_source1[:,0]] = 1.e-16

         Source2 = interpolate(Source2,S1)
         i_source2=np.argwhere(Source2.vector().get_local()[:]<=0)  #making negatives zero
         Source2.vector()[i_source2[:,0]] = 1.e-16

         Source3 = interpolate(Source3,S1)
         i_source3=np.argwhere(Source3.vector().get_local()[:]<=0)  #making negatives zero
         Source3.vector()[i_source3[:,0]] = 1.e-16

         Source4 = interpolate(Source4,S1)
         i_source4=np.argwhere(Source4.vector().get_local()[:]<=0)  #making negatives zero
         Source4.vector()[i_source4[:,0]] = 1.e-16

         Source5 = interpolate(Source5,S1)
         i_source5=np.argwhere(Source5.vector().get_local()[:]<=0)  #making negatives zero
         Source5.vector()[i_source5[:,0]] = 1.e-16

         Source6 = interpolate(Source6,S1)
         i_source6=np.argwhere(Source6.vector().get_local()[:]<=0)  #making negatives zero
         Source6.vector()[i_source6[:,0]] = 1.e-16
         ###############################################################

         #######################################################################
         print('Remeshing done!')
         #######################################################################

#######################################################################
#Print the time table
#######################################################################
list_timings(TimingClear.clear, [TimingType.wall])
#######################################################################
