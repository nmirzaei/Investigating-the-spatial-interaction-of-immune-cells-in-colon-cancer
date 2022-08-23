'''
Investigating the spatial interaction of immune cells in colon cancer

**This script calculates the Sensitivity of cancer cell total population to our model parameters**

qspmodel: Calculates the ODE parameters.
            Courtesy of Arkadz Kirshtein:
https://github.com/ShahriyariLab/Data-driven-mathematical-model-for-colon-cancer

Curvature: Calculates Curvature and Normal vectors of a given domain
            Courtesy of http://jsdokken.com/

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei

(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from fenics import *
#This has to be installed separately from Fenics
from dolfin_adjoint import *
import logging
import scipy.optimize as op
import pandas as pd
from subprocess import call
from qspmodel import *
from ufl import Min
from Curvature import *
import csv
import time
###############################################################

###############################################################
#Taking user input
###############################################################
answer = input('What is your target functional? (||v||=1, omega=2)')
if int(answer)!=1 or int(answer)!=2:
    print('Error: Wrong input!')
    exit()
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
#Reporting options
###############################################################
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("rothemain.rothe_utils")
logging.getLogger('UFL').setLevel(logging.WARNING)
logging.getLogger('FFC').setLevel(logging.WARNING)
###############################################################

###############################################################
# Nullspace of rigid motions
###############################################################
#Translation in 3D. Comment out if the problem is 2D
#Z_transl = [Constant((1, 0, 0)), Constant((0, 1, 0)), Constant((0, 0, 1))]

#Rotations 3D. Comment out if the problem is 2D
#Z_rot = [Expression(('0', 'x[2]', '-x[1]')),
#         Expression(('-x[2]', '0', 'x[0]')),
#         Expression(('x[1]', '-x[0]', '0'))]

##Translation 2D. Comment out if the problem is 3D
Z_transl = [Constant((1, 0)), Constant((0, 1))]

# Rotations 2D. Comment out if the problem is 3D
Z_rot = [Expression(('-x[1]', 'x[0]'),degree=0)]
# All
Z = Z_transl + Z_rot
###############################################################

###############################################################
#Variables to adjust for easy continuation of the code for more iterations
###############################################################
j = int(0)
idx = int(0)
###############################################################

###############################################################
# Load mesh
#In this code since we are calculating the displacement as well as
#the biology, we save the mesh at the end of each full run, so that the
#next run of the code can start from the deformed mesh of the last run.
#Parameters j and idx should be adjusted to the last run's last iteration number
###############################################################
#Parallel compatible Mesh readings
if j==0:
    mesh= Mesh()
    xdmf = XDMFFile(mesh.mpi_comm(), "Mesh.xdmf")
    xdmf.read(mesh)
    mvc = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile("Mesh.xdmf") as infile:
        infile.read(mvc, "f")
    Volume = cpp.mesh.MeshFunctionSizet(mesh, mvc)
    xdmf.close()
    mvc2 = MeshValueCollection("size_t", mesh, 1)
    with XDMFFile("boundaries.xdmf") as infile:
        infile.read(mvc2, "f")
    bnd_mesh = cpp.mesh.MeshFunctionSizet(mesh, mvc2)
if j>0:
    mesh= Mesh()
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
#######################################################################

# Construct integration measure using these markers
ds = Measure('ds', subdomain_data=bnd_mesh)
dx = Measure('dx', subdomain_data=Volume)
###############################################################

###############################################################
#time step variables
###############################################################
T = 3000            # final
num_steps= 3000  # number of time steps
dt = 1  # time step
eps = 1             # diffusion coefficient
t=0                 # initial time
k = Constant(dt)    # Constant tip step object for weak formulation
###############################################################

###############################################################
#ODE Parameters non-dimensional
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
# reading parameter values from file
###############################################################
Input2 = 0  #The patient's number. In this paper we only used Cluster 1 of patients
clustercells=pd.read_csv('input/input/Large_Tumor_cell_data.csv').to_numpy()
QSP_=QSP.from_cell_data(clustercells[Input2])
params=QSP_.par
pars = {Pars[k]:value for k,value in zip(range(len(Pars)),params)}
###############################################################

###############################################################
#Defining all model parameters as Dolfin constants for sensitivity purposes.
###############################################################
params0 = Constant(pars['A_{T_N}'])
params1 = Constant(pars['alpha_{T_NT_h}'])
params2 = Constant(pars['lambda_{T_hD}'])
params3 = Constant(pars['lambda_{T_hM}'])
params4 = Constant(pars['lambda_{T_hmu_1}'])
params5 = Constant(pars['alpha_{T_NT_C}'])
params6 = Constant(pars['lambda_{T_CT_h}'])
params7 = Constant(pars['lambda_{T_CD}'])
params8 = Constant(pars['alpha_{T_NT_r}'])
params9 = Constant(pars['lambda_{T_rT_h}'])
params10 = Constant(pars['lambda_{T_rmu_2}'])
params11 = Constant(pars['lambda_{T_rG_beta}'])
params12 = Constant(pars['delta_{T_N}'])
params13 = Constant(pars['delta_{T_hmu_2}'])
params14 = Constant(pars['delta_{T_hT_r}'])
params15 = Constant(pars['delta_{T_h}'])
params16 = Constant(pars['delta_{T_Cmu_2}'])
params17 = Constant(pars['delta_{T_CT_r}'])
params18 = Constant(pars['delta_{T_C}'])
params19 = Constant(pars['delta_{T_rmu_1}'])
params20 = Constant(pars['delta_{T_r}'])
params21 = Constant(pars['A_{Dn}'])
params22 = Constant(pars['alpha_{D_ND}'])
params23 = Constant(pars['lambda_{DH}'])
params24 = Constant(pars['lambda_{DC}'])
params25 = Constant(pars['delta_{DH}'])
params26 = Constant(pars['delta_{D}'])
params27 = Constant(pars['delta_{DC}'])
params28 = Constant(pars['lambda_{Mmu_2}'])
params29 = Constant(pars['lambda_{MI_gamma}'])
params30 = Constant(pars['lambda_{MT_h}'])
params31 = Constant(pars['M_0'])
params32 = Constant(pars['delta_{M}'])
params33 = Constant(pars['lambda_{C}'])
params34 = Constant(pars['lambda_{Cmu_1}'])
params35 = Constant(pars['C_0'])
params36 = Constant(pars['delta_{CG_beta}'])
params37 = Constant(pars['delta_{CI_gamma}'])
params38 = Constant(pars['delta_{CT_C}'])
params39 = Constant(pars['delta_{C}'])
params40 = Constant(pars['alpha_{NC}'])
params41 = Constant(pars['delta_{N}'])
params42 = Constant(pars['lambda_{HN}'])
params43 = Constant(pars['lambda_{HM}'])
params44 = Constant(pars['lambda_{HT_h}'])
params45 = Constant(pars['lambda_{HT_C}'])
params46 = Constant(pars['lambda_{HT_r}'])
params47 = Constant(pars['delta_{H}'])
params48 = Constant(pars['lambda_{mu_1T_h}'])
params49 = Constant(pars['lambda_{mu_1M}'])
params50 = Constant(pars['lambda_{mu_1D}'])
params51 = Constant(pars['delta_{mu_1}'])
params52 = Constant(pars['lambda_{mu_2M}'])
params53 = Constant(pars['lambda_{mu_2D}'])
params54 = Constant(pars['lambda_{mu_2T_r}'])
params55 = Constant(pars['delta_{mu_2}'])
params56 = Constant(pars['lambda_{I_gammaT_h}'])
params57 = Constant(pars['lambda_{I_gammaT_C}'])
params58 = Constant(pars['lambda_{I_gammaM}'])
params59 = Constant(pars['delta_{I_gamma}'])
params60 = Constant(pars['lambda_{G_betaM}'])
params61 = Constant(pars['lambda_{G_betaT_r}'])
params62 = Constant(pars['delta_{G_beta}'])

PARAMS = [params0,params1,params2,params3,params4,params5,params6,params7,params8,\
params9,params10,params11,params12,params13,params14,params15,params16,params17,\
params18,params19,params20,params21,params22,params23,params24,params25,params26,\
params27,params28,params29,params30,params31,params32,params33,params34,params35,\
params36,params37,params38,params39,params40,params41,params42,params43,params44,\
params45,params46,params47,params48,params49,params50,params51,params52,params53,\
params54,params55,params56,params57,params58,params59,params60,params61,params62]
###############################################################

###############################################################
#Initial conditions
###############################################################
#Tn is outside of the environment and are modeled as ODEs so uniform IC
#Cytokines are taken to be zero at the begining.
#Immune cells will start at their source locations.
#Cancer cells and Necrotic cells ICs are uniform and their values are
#extracted from the ODE paper
IC=np.array(pd.read_csv('input/input/Initial_Conditions.csv'))
Tn_0 = Expression('Init1', degree=0, Init1=IC[Input2,0])
Th_0 = Expression('Init2', degree=0, Init2=0)
Tc_0 = Expression('Init3', degree=0, Init3=0)
Tr_0 = Expression('Init4', degree=0, Init4=0)
Dn_0 = Expression('Init5', degree=0, Init5=0)
D_0 = Expression('Init6', degree=0, Init6=0)
M_0 = Expression('Init7', degree=0, Init7=0)
C_0 = Expression('Init8', degree=0, Init8=IC[Input2,7])
N_0 = Expression('Init9', degree=0, Init9=IC[Input2,8])
H_0 = Expression('Init10', degree=0, Init10=0)
mu1_0 = Expression('Init11', degree=0, Init11=0)
mu2_0 = Expression('Init12', degree=0, Init12=0)
Igamma_0 = Expression('Init13', degree=0, Init13=0)
Gbeta_0 = Expression('Init14', degree=0, Init14=0)
#Interpolation of the ICs into mixed space subspaces
Tn0 = interpolate(Tn_0, Mixed_Space.sub(0).collapse())
Th0 = interpolate(Th_0, Mixed_Space.sub(1).collapse())
Tc0 = interpolate(Tc_0, Mixed_Space.sub(2).collapse())
Tr0 = interpolate(Tr_0, Mixed_Space.sub(3).collapse())
Dn0 = interpolate(Dn_0, Mixed_Space.sub(4).collapse())
D0 = interpolate(D_0, Mixed_Space.sub(5).collapse())
M0 = interpolate(M_0, Mixed_Space.sub(6).collapse())
C0 = interpolate(C_0, Mixed_Space.sub(7).collapse())
N0 = interpolate(N_0, Mixed_Space.sub(8).collapse())
H0 = interpolate(H_0, Mixed_Space.sub(9).collapse())
mu10 = interpolate(mu1_0, Mixed_Space.sub(10).collapse())
mu20 = interpolate(mu2_0, Mixed_Space.sub(11).collapse())
Igamma0 = interpolate(Igamma_0, Mixed_Space.sub(12).collapse())
Gbeta0 = interpolate(Gbeta_0, Mixed_Space.sub(13).collapse())
###############################################################

###############################################################
#The orientation omega measured by the the polar angle of each mesh point
###############################################################
omega=project(Expression('atan2(x[1],x[0])',degree=0),S1)
###############################################################

###############################################################
#For this code we also read the location of the sources from the last run
#Since the sources displace with the deformation of the mesh
###############################################################
#In case this is the first run, just construct the sources from scratch. Otherwise
#read is from the saved files of the previous run
if j==0:
    ss1= Expression('(x[0]-0.1875)*(x[0]-0.1875) + (x[1]-0.32476)*(x[1]-0.32476)', degree=0)
    source1 = conditional(lt(ss1,0.005),1,0)
    Source1 = project(source1,S1)
    i_source1=np.argwhere(Source1.vector().get_local()[:]<=0)  #making negatives zero
    Source1.vector()[i_source1[:,0]] = 1.e-16
    File('test/source1.pvd')<<Source1

    ss2= Expression('(x[0]+0.1875)*(x[0]+0.1875) + (x[1]-0.32476)*(x[1]-0.32476)', degree=0)
    source2 = conditional(lt(ss2,0.005),1,0)
    Source2 = project(source2,S1)
    i_source2=np.argwhere(Source2.vector().get_local()[:]<=0)  #making negatives zero
    Source2.vector()[i_source2[:,0]] = 1.e-16
    File('test/source2.pvd')<<Source2

    ss3= Expression('(x[0]+0.375)*(x[0]+0.375) + (x[1])*(x[1])', degree=0)
    source3 = conditional(lt(ss3,0.005),1,0)
    Source3 = project(source3,S1)
    i_source3=np.argwhere(Source3.vector().get_local()[:]<=0)  #making negatives zero
    Source3.vector()[i_source3[:,0]] = 1.e-16
    File('test/source3.pvd')<<Source3

    ss4= Expression('(x[0]+0.1875)*(x[0]+0.1875) + (x[1]+0.32476)*(x[1]+0.32476)', degree=0)
    source4 = conditional(lt(ss4,0.005),1,0)
    Source4 = project(source4,S1)
    i_source4=np.argwhere(Source4.vector().get_local()[:]<=0)  #making negatives zero
    Source4.vector()[i_source4[:,0]] = 1.e-16
    File('test/source4.pvd')<<Source4

    ss5= Expression('(x[0]-0.1875)*(x[0]-0.1875) + (x[1]+0.32476)*(x[1]+0.32476)', degree=0)
    source5 = conditional(lt(ss5,0.005),1,0)
    Source5 = project(source5,S1)
    i_source5=np.argwhere(Source5.vector().get_local()[:]<=0)  #making negatives zero
    Source5.vector()[i_source5[:,0]] = 1.e-16
    File('test/source5.pvd')<<Source5

    ss6= Expression('(x[0]-0.375)*(x[0]-0.375) + (x[1])*(x[1])', degree=0)
    source6 = conditional(lt(ss6,0.005),1,0)
    Source6 = project(source6,S1)
    i_source6=np.argwhere(Source6.vector().get_local()[:]<=0)  #making negatives zero
    Source6.vector()[i_source6[:,0]] = 1.e-16
    File('test/source6.pvd')<<Source6
if j>0:
    Source1 = Function(S1,"Biology/source1_%d.xml" %(idx))
    Source2 = Function(S1,"Biology/source2_%d.xml" %(idx))
    Source3 = Function(S1,"Biology/source3_%d.xml" %(idx))
    Source4 = Function(S1,"Biology/source4_%d.xml" %(idx))
    Source5 = Function(S1,"Biology/source5_%d.xml" %(idx))
    Source6 = Function(S1,"Biology/source6_%d.xml" %(idx))
Source = [Source1,Source2,Source3,Source4,Source5,Source6]
###############################################################

###############################################################
#Assigning subspace projected ICs as the initial step of iteration
###############################################################
assign(U_n.sub(0),Tn0)
assign(U_n.sub(1),Th0)
assign(U_n.sub(2),Tc0)
assign(U_n.sub(3),Tr0)
assign(U_n.sub(4),Dn0)
assign(U_n.sub(5),D0)
assign(U_n.sub(6),M0)
assign(U_n.sub(7),C0)
assign(U_n.sub(8),N0)
assign(U_n.sub(9),H0)
assign(U_n.sub(10),mu10)
assign(U_n.sub(11),mu20)
assign(U_n.sub(12),Igamma0)
assign(U_n.sub(13),Gbeta0)

Tn_n, Th_n, Tc_n, Tr_n, Dn_n, D_n, M_n, C_n, N_n, H_n, mu1_n, mu2_n, Igamma_n, Gbeta_n= U_n.split()
##############################################################


###############################################################
#PDE Parameters dimensional. We use Dolfin constant to be able to calculate their sensitivity
###############################################################
#They are all in cm^2/day.
D_Th, D_Tc, D_Tr, D_Dn, D_D, D_M, D_C, D_N, D_H, D_mu1, D_mu2, D_Igamma, D_Gbeta =  Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(7.92e-2), Constant(1.24e-3), Constant(1.24e-3), Constant(1.24e-3), Constant(1.24e-3)
coeff = Constant(1)    #advection constant
S_Th, S_Tc, S_Tr, S_Dn, S_D, S_M = Constant(1),Constant(1),Constant(1),Constant(1),Constant(1),Constant(1)
Source_const = [S_Th, S_Tc, S_Tr, S_Dn, S_D, S_M]
##############################################################


###############################################################
# Create XDMFFile files for cancer to make sure the code is running fine. (optional)
###############################################################
vtkfile = XDMFFile(MPI.comm_world,"reaction_system/C.xdmf")
vtkfile.parameters["flush_output"] = True
##############################################################

##############################################################
#Sum of right hand sides function
##############################################################
def RHS_sum(U,S_C,params):
    Tn, Th, Tc, Tr, Dn, D, M, C, N, H, mu1, mu2, Igamma, Gbeta= U.split()
    RHS_Th = ((params[2]*D+params[3]*M+params[2]*mu1)*Tn-(params[13]*mu2+params[14]*Tr+params[15])*Th)+S_C[0]*Source1
    RHS_Tc = ((params[6]*Th+params[7]*D)*Tn-(params[16]*mu2+params[17]*Tr+params[18])*Tc)+S_C[1]*Source2
    RHS_Tr = ((params[9]*Th+params[10]*mu2+params[11]*Gbeta)*Tn-(params[19]*mu1+params[20])*Tr)+S_C[2]*Source3
    RHS_Dn = (params[21]-params[22]*(params[23]*H+params[24]*C)*Dn-(params[25]*H+params[26])*Dn)+S_C[3]*Source4
    RHS_D = ((params[23]*H+params[24]*C)*Dn-(params[25]*H+params[27]*C+params[26])*D)+S_C[4]*Source5
    RHS_M = ((params[28]*mu2+params[29]*Igamma+params[30]*Th)*(params[31]-M)-params[32]*M)+S_C[5]*Source6
    RHS_C = ((params[33]+params[34]*mu1)*C*(Constant(1)-C/params[35])-(params[36]*Gbeta+params[37] *Igamma+params[38]*Tc+params[39])*C)
    RHS_N = (params[40]*(params[36]*Gbeta+params[37]*Igamma+params[38]*Tc+params[39])*C-params[41]*N)
    return RHS_Th+RHS_Tc+RHS_Tr+RHS_Dn+RHS_D+RHS_M+RHS_C+RHS_N

RHS = RHS_sum(U_n,Source_const,PARAMS)
RHS_MECH_ = project(RHS,S1)
##############################################################




#######################################################################
#Mesh and remeshing related info and
#######################################################################
numCells = mesh.num_cells()
mesh.smooth(100)
Counter=0
#######################################################################

#######################################################################
#loop parameters
#######################################################################
t = 0.0
#######################################################################

#If not the first run of the code, read the biology from the saved files of
#the last run
if j>0:
    U_n = Function(Mixed_Space,"Biology/U%d.xml" %(idx))
#######################################################################
#######################################################################

#Curvature and Normal vector for the initial domain
#######################################################################
crvt1,NORMAL1 = Curvature(mesh)
#######################################################################


#151 steps at a time or the memory will fill up.
#This code saves the last U_n, mesh and sources so you can continure to higher values of n.
#For example once you calculate this loop is done you can immediatley start a
#new code with range(151,301) and so on. Make sure the last U_n,mesh and sources from the previous
#code are saved in the directory of the new code.
for n in range(151):
     ##############################################################
     #First we plot the ICs and then solve. This is why we have this if condition
     ##############################################################
     if j>=1:
         #############################################################
         if j>=2:
             #############################################################
             #Mechanical problem setup and solve
             #############################################################
             mu = 1
             RHS = RHS_sum(U_n,Source_const,PARAMS)
             RHS_MECH_ = project(RHS,S1)
             (u1, p1, l1) = TrialFunctions(W)
             (v0, q0, w0) = TestFunctions(W)
             I = Identity(2)
             Sigma = -p1*I + 2*mu*sym(grad(u1))
             stokes1 = inner(Sigma,grad(v0))*dx  + (div(u1)-k*(0.01)*RHS_MECH_)*q0*dx -sum(l1[i]*inner(v0, Z[i])*dx for i in range(len(Z)))-sum(w0[i]*inner(u1, Z[i])*dx for i in range(len(Z)))+0.0864*dot(crvt1*NORMAL1,v0)*ds(1)  #0.0001 in cm^2/day is 0.0864 mm^2/s
             a1 = lhs(stokes1)
             L1 = rhs(stokes1)
             solve(a1==L1,UU)
             u_, p_,lambda_ = UU.split()
             Sigma = project(-p_*I + 2*mu*sym(grad(u_)),TFS)
             ##############################################################

             #############################################################
             #Create displacement for mesh movement. Moving from current configuration
             ##############################################################
             dis = u_
             displ = project(dis,VV1)
             ALE.move(mesh,displ)
             #############################################################

             ##############################################################
             #Updatung the curvature and normal vectors for the current configuration
             ##############################################################
             crvt1,NORMAL1 = Curvature(mesh)
             ##############################################################

         ##############################################################
         #Loop info update and printing
         ##############################################################
         print(n,flush=True)
         t+=dt
         ##############################################################

         #Update biology PDE and solve
         F1 = ((Tn-Tn_n)/k)*v1*dx-(params0-params1*(params2*D+params3*M+params4*mu1)*Tn-params5 *(params6*Th+params7*D)*Tn-params8*(params9*Th+params10*mu2+params11*Gbeta)*Tn-params12*Tn)*v1*dx\
         + ((Th-Th_n)/k)*v2*dx+D_Th*dot(grad(Th),grad(v2))*dx+coeff*Th*div(u__)*v2*dx-((params2*D+params3*M+params2*mu1)*Tn-(params13*mu2+params14*Tr+params15)*Th)*v2*dx-S_Th*Source1*v2*dx\
         + ((Tc-Tc_n)/k)*v3*dx+D_Tc*dot(grad(Tc),grad(v3))*dx+coeff*Tc*div(u__)*v3*dx-((params6*Th+params7*D)*Tn-(params16*mu2+params17*Tr+params18)*Tc)*v3*dx-S_Tc*Source2*v3*dx\
         + ((Tr-Tr_n)/k)*v4*dx+D_Tr*dot(grad(Tr),grad(v4))*dx+coeff*Tr*div(u__)*v4*dx-((params9*Th+params10*mu2+params11*Gbeta)*Tn-(params19*mu1+params20)*Tr)*v4*dx-S_Tr*Source3*v4*dx\
         + ((Dn-Dn_n)/k)*v5*dx+D_Dn*dot(grad(Dn),grad(v5))*dx+coeff*Dn*div(u__)*v5*dx-(params21-params22*(params23*H+params24*C)*Dn-(params25*H+params26)*Dn)*v5*dx-S_Dn*Source4*v5*dx\
         + ((D-D_n)/k)*v6*dx+D_D*dot(grad(D),grad(v6))*dx+coeff*D*div(u__)*v6*dx-((params23*H+params24*C)*Dn-(params25*H+params27*C+params26)*D)*v6*dx-S_D*Source5*v6*dx\
         + ((M-M_n)/k)*v7*dx+D_M*dot(grad(M),grad(v7))*dx+coeff*M*div(u__)*v7*dx-((params28*mu2+params29*Igamma+params30*Th)*(params31-M)-params32*M)*v7*dx-S_M*Source6*v7*dx\
         + ((C-C_n)/k)*v8*dx+D_C*dot(grad(C),grad(v8))*dx+coeff*C*div(u__)*v8*dx-((params33+params34*mu1)*C*(Constant(1)-C/params35)-(params36*Gbeta+params37 *Igamma+params38*Tc+params39)*C)*v8*dx\
         + ((N-N_n)/k)*v9*dx+D_N*dot(grad(N),grad(v9))*dx+coeff*N*div(u__)*v9*dx-(params40*(params36*Gbeta+params37*Igamma+params38*Tc+params39)*C-params41*N)*v9*dx\
         + ((H-H_n)/k)*v10*dx+D_H*dot(grad(H),grad(v10))*dx-(params42*N+params43*M+params44*Th+params45*Tc+params46*Tr-params47*H)*v10*dx\
         + ((mu1-mu1_n)/k)*v11*dx+D_mu1*dot(grad(mu1),grad(v11))*dx-(params48*Th+params49*M+params50*D-params51*mu1)*v11*dx\
         + ((mu2-mu2_n)/k)*v12*dx+D_mu2*dot(grad(mu2),grad(v12))*dx-(params52*M+params53*D+params54*Tr-params55*mu2)*v12*dx\
         + ((Igamma-Igamma_n)/k)*v13*dx+D_Igamma*dot(grad(Igamma),grad(v13))*dx-(params56*Th+params57*Tc+params58*M-params59*Igamma)*v13*dx\
         + ((Gbeta-Gbeta_n)/k)*v14*dx+D_Gbeta*dot(grad(Gbeta),grad(v14))*dx-(params60*M+params61*Tr-params62*Gbeta)*v14*dx
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
     #Plotting the dynamics every 10 steps
     #######################################################################
     if n%10==0:
           CC.rename('C_n','C_n')
           vtkfile.write(CC,t)
           ##############################################################

     j+=1
     ##############################################################

##############################################################
#Saving files for the next run
##############################################################
File("Biology/U%d.xml" %(n))<<U
File("Biology/source1_%d.xml" %(n))<<Source1
File("Biology/source2_%d.xml" %(n))<<Source2
File("Biology/source3_%d.xml" %(n))<<Source3
File("Biology/source4_%d.xml" %(n))<<Source4
File("Biology/source5_%d.xml" %(n))<<Source5
File("Biology/source6_%d.xml" %(n))<<Source6
##############################################################

##############################################################
#Plotting for quality check. (optional)
##############################################################
vtkfile1 = XDMFFile(MPI.comm_world,"Mesh1.xdmf")
vtkfile1.parameters["flush_output"] = True
vtkfile2 = XDMFFile(MPI.comm_world,"boundaries1.xdmf")
vtkfile2.parameters["flush_output"] = True
vtkfile1.write(Volume)
vtkfile2.write(bnd_mesh)
##############################################################

##############################################################
#Sensitivity Analysis using adjoint method
##############################################################

##############################################################
#Define your functional
##############################################################
if int(answer)==1:
    J1 = assemble(dot(u_,u_)*dx)
elif int(answer)=2:
    J1 = assemble(omega*dx)
##############################################################

##############################################################
print('Assemble and Controls done!')
##############################################################

##############################################################
#Computing sensitivities
##############################################################
print('\n This will take a very long time. On GHPCC took close to 2 days!!')
SS = []
start = time.time()

dJ1dp0 = compute_gradient(J1, Control(params0), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp0,S1).vector()[:]))
print('1 Gradient done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
end = time.time()
print ("Time elapsed:", end - start)
##############################################################
dJ1dp1 = compute_gradient(J1, Control(params1), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp1,S1).vector()[:]))
print('2 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp2 = compute_gradient(J1, Control(params2), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp2,S1).vector()[:]))
print('3 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp3 = compute_gradient(J1, Control(params3), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp3,S1).vector()[:]))
print('4 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp4 = compute_gradient(J1, Control(params4), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp4,S1).vector()[:]))
print('5 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp5 = compute_gradient(J1, Control(params5), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp5,S1).vector()[:]))
print('6 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp6 = compute_gradient(J1, Control(params6), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp6,S1).vector()[:]))
print('7 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp7 = compute_gradient(J1, Control(params7), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp7,S1).vector()[:]))
print('8 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp8 = compute_gradient(J1, Control(params8), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp8,S1).vector()[:]))
print('9 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp9 = compute_gradient(J1, Control(params9), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp9,S1).vector()[:]))
print('10 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp10 = compute_gradient(J1, Control(params10), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp10,S1).vector()[:]))
print('11 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp11 = compute_gradient(J1, Control(params11), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp11,S1).vector()[:]))
print('12 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp12 = compute_gradient(J1, Control(params12), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp12,S1).vector()[:]))
print('13 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp13 = compute_gradient(J1, Control(params13), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp13,S1).vector()[:]))
print('14 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp14 = compute_gradient(J1, Control(params14), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp14,S1).vector()[:]))
print('15 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp15 = compute_gradient(J1, Control(params15), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp15,S1).vector()[:]))
print('16 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp16 = compute_gradient(J1, Control(params16), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp16,S1).vector()[:]))
print('17 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp17 = compute_gradient(J1, Control(params17), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp17,S1).vector()[:]))
print('18 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp18 = compute_gradient(J1, Control(params18), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp18,S1).vector()[:]))
print('19 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp19 = compute_gradient(J1, Control(params19), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp19,S1).vector()[:]))
print('20 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp20 = compute_gradient(J1, Control(params20), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp20,S1).vector()[:]))
print('21 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp21 = compute_gradient(J1, Control(params21), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp21,S1).vector()[:]))
print('22 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp22 = compute_gradient(J1, Control(params22), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp22,S1).vector()[:]))
print('23 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp23 = compute_gradient(J1, Control(params23), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp23,S1).vector()[:]))
print('24 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp24 = compute_gradient(J1, Control(params24), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp24,S1).vector()[:]))
print('25 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp25 = compute_gradient(J1, Control(params25), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp25,S1).vector()[:]))
print('26 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp26 = compute_gradient(J1, Control(params26), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp26,S1).vector()[:]))
print('27 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp27 = compute_gradient(J1, Control(params27), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp27,S1).vector()[:]))
print('28 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp28 = compute_gradient(J1, Control(params28), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp28,S1).vector()[:]))
print('29 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp29 = compute_gradient(J1, Control(params29), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp29,S1).vector()[:]))
print('30 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp30 = compute_gradient(J1, Control(params30), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp30,S1).vector()[:]))
print('31 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp31 = compute_gradient(J1, Control(params31), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp31,S1).vector()[:]))
print('32 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp32 = compute_gradient(J1, Control(params32), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp32,S1).vector()[:]))
print('33 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp33 = compute_gradient(J1, Control(params33), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp33,S1).vector()[:]))
print('34 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp34 = compute_gradient(J1, Control(params34), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp34,S1).vector()[:]))
print('35 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp35 = compute_gradient(J1, Control(params35), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp35,S1).vector()[:]))
print('36 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp36 = compute_gradient(J1, Control(params36), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp36,S1).vector()[:]))
print('37 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp37 = compute_gradient(J1, Control(params37), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp37,S1).vector()[:]))
print('38 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp38 = compute_gradient(J1, Control(params38), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp38,S1).vector()[:]))
print('39 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp39 = compute_gradient(J1, Control(params39), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp39,S1).vector()[:]))
print('40 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp40 = compute_gradient(J1, Control(params40), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp40,S1).vector()[:]))
print('41 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp41 = compute_gradient(J1, Control(params41), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp41,S1).vector()[:]))
print('42 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp42 = compute_gradient(J1, Control(params42), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp42,S1).vector()[:]))
print('43 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp43 = compute_gradient(J1, Control(params43), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp43,S1).vector()[:]))
print('44 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp44 = compute_gradient(J1, Control(params44), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp44,S1).vector()[:]))
print('45 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp45 = compute_gradient(J1, Control(params45), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp45,S1).vector()[:]))
print('46 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp46 = compute_gradient(J1, Control(params46), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp46,S1).vector()[:]))
print('47 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp47 = compute_gradient(J1, Control(params47), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp47,S1).vector()[:]))
print('48 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp48 = compute_gradient(J1, Control(params48), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp48,S1).vector()[:]))
print('49 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp49 = compute_gradient(J1, Control(params49), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp49,S1).vector()[:]))
print('50 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp50= compute_gradient(J1, Control(params50), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp50,S1).vector()[:]))
print('51 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp51= compute_gradient(J1, Control(params51), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp51,S1).vector()[:]))
print('52 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp52= compute_gradient(J1, Control(params52), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp52,S1).vector()[:]))
print('53 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp53= compute_gradient(J1, Control(params53), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp53,S1).vector()[:]))
print('54 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp54= compute_gradient(J1, Control(params54), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp54,S1).vector()[:]))
print('55 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp55= compute_gradient(J1, Control(params55), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp55,S1).vector()[:]))
print('56 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp56= compute_gradient(J1, Control(params56), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp56,S1).vector()[:]))
print('57 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp57= compute_gradient(J1, Control(params57), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp57,S1).vector()[:]))
print('58 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp58= compute_gradient(J1, Control(params58), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp58,S1).vector()[:]))
print('59 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
dJ1dp59= compute_gradient(J1, Control(params59), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp59,S1).vector()[:]))
print('60 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################

dJ1dp60= compute_gradient(J1, Control(params60), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp60,S1).vector()[:]))
print('61 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################

dJ1dp61= compute_gradient(J1, Control(params61), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp61,S1).vector()[:]))
print('62 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################

dJ1dp62= compute_gradient(J1, Control(params62), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dp62,S1).vector()[:]))
print('63 Gradients done!')
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
##############################################################
##############################################################
##############################################################

dJ1dDTh = compute_gradient(J1, Control(D_Th), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDTh,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('1st Diffusion Gradient done!')
##############################################################
dJ1dDTc = compute_gradient(J1, Control(D_Tc), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDTc,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('2nd Diffusion Gradient done!')
##############################################################
dJ1dDTr = compute_gradient(J1, Control(D_Tr), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDTr,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('3rd Diffusion Gradient done!')
##############################################################
dJ1dDDn = compute_gradient(J1, Control(D_Dn), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDDn,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('4th Diffusion Gradient done!')
##############################################################
dJ1dDD = compute_gradient(J1, Control(D_D), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDD,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('5th Diffusion Gradient done!')
##############################################################
dJ1dDM = compute_gradient(J1, Control(D_M), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDM,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('6th Diffusion Gradient done!')
##############################################################
dJ1dDC = compute_gradient(J1, Control(D_C), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDC,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('7th Diffusion Gradient done!')
##############################################################
dJ1dDN = compute_gradient(J1, Control(D_N), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDN,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('8th Diffusion Gradient done!')
##############################################################
dJ1dDH = compute_gradient(J1, Control(D_H), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDH,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('9th Diffusion Gradient done!')
##############################################################
dJ1dDmu1 = compute_gradient(J1, Control(D_mu1), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDmu1,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('10th Diffusion Gradient done!')
##############################################################
dJ1dDmu2 = compute_gradient(J1, Control(D_mu2), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDmu2,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('11th Diffusion Gradient done!')
##############################################################
dJ1dDIgamma = compute_gradient(J1, Control(D_Igamma), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDIgamma,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('12th Diffusion Gradient done!')
##############################################################
dJ1dDGbeta = compute_gradient(J1, Control(D_Gbeta), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dDGbeta,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('13th Diffusion Gradient done!')
##############################################################
##############################################################
##############################################################

dJ1dS_Th = compute_gradient(J1, Control(S_Th), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dS_Th,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('1st Source Gradient done!')
#######################################################################
dJ1dS_Tc = compute_gradient(J1, Control(S_Tc), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dS_Tc,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('2nd Source Gradient done!')
#######################################################################
dJ1dS_Tr = compute_gradient(J1, Control(S_Tr), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dS_Tr,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('3rd Source Gradient done!')
#######################################################################
dJ1dS_Dn = compute_gradient(J1, Control(S_Dn), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dS_Dn,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('4th Source Gradient done!')
#######################################################################
dJ1dS_D = compute_gradient(J1, Control(S_D), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dS_D,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('5th Source Gradient done!')
#######################################################################
dJ1dS_M = compute_gradient(J1, Control(S_M), options={"riesz_representation": "L2"})
SS.append(max(project(dJ1dS_M,S1).vector()[:]))
c=csv.writer(open('output/sensitivities_cancer.csv',"w"))
c.writerow(SS)
del c
print('6th Source Gradient done!')
#######################################################################
#Print the time table
#######################################################################
list_timings(TimingClear.clear, [TimingType.wall])
#######################################################################
