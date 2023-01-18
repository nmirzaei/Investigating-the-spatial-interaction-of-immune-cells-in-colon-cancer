'''
Investigating the spatial interaction of immune cells in colon cancer

**This script calculates the Sensitivity of cancer cell total population to our model parameters**

qspmodel: Calculates the ODE parameters.
            Courtesy of Arkadz Kirshtein:
https://github.com/ShahriyariLab/Data-driven-mathematical-model-for-colon-cancer

IC_source: Creates initial conditions and sources for immune cells

check_dependencies: Checks for dependencies and installs them if missing

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei
(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from fenics import *
#This has to be installed separately from Fenics. This package helps with the sensitivity analysis.
from dolfin_adjoint import *
import logging
import scipy.optimize as op
import pandas as pd
from subprocess import call
from qspmodel import *
from check_dependencies import *
from IC_Source import *
from ufl import Min
import csv
import os
import math as mth
###############################################################

###############################################################
#Check required packages
###############################################################
check_dependencies()
###############################################################

###############################################################
#Checking to see the displacements are available
###############################################################
answer = input('Have you calculated the displacements? ( I do not know=2, Yes=1, No=0)')
if int(answer)==2:
    print('See if the displacement folder is populated with .xml files u1.xml to u3000.xml')
    answer = input('Now, have you calculated the displacements? ( I do not know=2, Yes=1, No=0')
elif int(answer) == 0:
    print('Error: This code needs pre-calculated displacements\n')
    print('**Go to the folder Colon_Cancer_ImmuneCells_Sources_Inside and finish running the Main.py file. This will produce all the displacements!**\n')
    exit()
elif int(answer)==1:
    print('Displacement availability: Checked')
else:
    print('Error: Wrong input!')
    exit()
###############################################################

###############################################################
#Checking to see if this is a continuing run or the first run
#if continuing the code will upload the information from the last run
###############################################################
answer = input('Is this your first run of sensitivity or you are continuing? ( continuing=1, first run=0)')
if int(answer)==1:
    file_name = os.listdir('Biology')
    displ_name = os.path.splitext(file_name[-1])[0]
    it_start = displ_name[1:]
elif int(answer)==0:
    it_start = 0
else:
    print('Error: Wrong input!')
    exit()
###############################################################

###############################################################
#Checking memory availibility
###############################################################
mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
mem_gib = mem_bytes/(1024.**3)
###############################################################

###############################################################
#Taking user input
###############################################################
answer = input('What is your target functional? (Total cancer cells=1, Total cancer/immune ratio=2)')
if int(answer)!=1 and int(answer)!=2:
    print('Error: Wrong input!')
    exit()

answer_1 = input('What percentage of memory do you prefer to use? (Recommended range 20-40)')
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
#Paths
###############################################################
# gives the path of Main.py
path = os.path.realpath(__file__)
# gives the directory where Main.py exists
dir = os.path.dirname(path)
folder = os.path.basename(dir)
# replaces folder name to Meshes in directory
dir = dir.replace(folder, 'Meshes')
# changes the current directory to Meshes folder
os.chdir(dir)
###############################################################

###############################################################
# Load mesh
###############################################################
#Parallel compatible Mesh readings
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
###############################################################

###############################################################
#Paths
###############################################################
# replaces Meshes folder name to the current code directory
dir = dir.replace('Meshes',folder)
# changes the meshes directory to the current code folder
os.chdir(dir)
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

# Construct integration measures using these markers
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
k = Constant(dt)    # Constant time step object for the weak formulation
###############################################################

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
#Parameters ordered by appearance in ODE
###############################################################
Pars_ordered = ['A_{T_N}','alpha_{T_NT_h}','lambda_{T_hD}','lambda_{T_hM}','lambda_{T_hmu_1}','alpha_{T_NT_C}','lambda_{T_CT_h}','lambda_{T_CD}',
                                     'alpha_{T_NT_r}','lambda_{T_rT_h}','lambda_{T_rmu_2}','lambda_{T_rG_beta}','delta_{T_N}','delta_{T_hmu_2}','delta_{T_hT_r}','delta_{T_h}',
                                     'delta_{T_Cmu_2}','delta_{T_CT_r}','delta_{T_C}','delta_{T_rmu_1}','delta_{T_r}','A_{Dn}','alpha_{D_ND}','lambda_{DH}','lambda_{DC}','delta_{DH}',
                                     'delta_{D}','delta_{DC}','lambda_{Mmu_2}','lambda_{MI_gamma}','lambda_{MT_h}','M_0','delta_{M}','lambda_{C}','lambda_{Cmu_1}','C_0','delta_{CG_beta}',
                                     'delta_{CI_gamma}','delta_{CT_C}','delta_{C}','alpha_{NC}','delta_{N}','lambda_{HN}','lambda_{HM}','lambda_{HT_h}','lambda_{HT_C}','lambda_{HT_r}','delta_{H}',
                                     'lambda_{mu_1T_h}','lambda_{mu_1M}','lambda_{mu_1D}','delta_{mu_1}','lambda_{mu_2M}','lambda_{mu_2D}','lambda_{mu_2T_r}','delta_{mu_2}','lambda_{I_gammaT_h}',
                                     'lambda_{I_gammaT_C}','lambda_{I_gammaM}','delta_{I_gamma}','lambda_{G_betaM}','lambda_{G_betaT_r}','delta_{G_beta}']
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
theta=[]
for idx in range(len(Pars_ordered)):
    theta.append(Constant(pars[Pars_ordered[idx]]))
###############################################################

###############################################################
#Initial conditions
###############################################################
Source1,Source2,Source3,Source4,Source5,Source6,U_n = IC_Source(Mixed_Space,S1,U_n,Input2)
Source = [Source1,Source2,Source3,Source4,Source5,Source6]
Tn_n, Th_n, Tc_n, Tr_n, Dn_n, D_n, M_n, C_n, N_n, H_n, mu1_n, mu2_n, Igamma_n, Gbeta_n= U_n.split()
##############################################################


###############################################################
#PDE Parameters (dimensional). We use Dolfin constant to be able to calculate their sensitivity
###############################################################
#They are all in cm^2/day.
D_Th, D_Tc, D_Tr, D_Dn, D_D, D_M, D_C, D_N, D_H, D_mu1, D_mu2, D_Igamma, D_Gbeta =  Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(8.64e-6), Constant(7.92e-2), Constant(1.24e-3), Constant(1.24e-3), Constant(1.24e-3), Constant(1.24e-3)
Dfc = [D_Th, D_Tc, D_Tr, D_Dn, D_D, D_M, D_C, D_N, D_H, D_mu1, D_mu2, D_Igamma, D_Gbeta]
coeff = Constant(1)    #advection constant
S_Th, S_Tc, S_Tr, S_Dn, S_D, S_M = Constant(1),Constant(1),Constant(1),Constant(1),Constant(1),Constant(1)
Src = [S_Th, S_Tc, S_Tr, S_Dn, S_D, S_M]
##############################################################

###############################################################
# Create XDMFFile files for cancer to make sure the code is running fine. (optional)
###############################################################
vtkfile = XDMFFile(MPI.comm_world,"reaction_system/C.xdmf")
vtkfile.parameters["flush_output"] = True
##############################################################


#######################################################################
#Mesh and remeshing related info
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
#Calculating the maximum number of iterations and asking the user to pick their preference
#######################################################################
mem_use = mem_gib*(int(answer_1)/100)*1024
max_itr =  (1/18.1667)*(mem_use-528)+2
print('The suggested maximum number of iterations based on your memory and your usage preference is:', mth.floor(max_itr))
usr_itr = input('Now enter the maximum number of iterations based on the suggested value:')
#######################################################################

#This code saves the last U_n, mesh and sources so you can continue to higher values of n.
#For example once this loop is done you can immediatley start a
#new code with range(int(it_start),int(it_start)+int(usr_itr)) and so on. Make sure the last U_n, mesh and sources from the previous
#code are saved in the directory of the new code.
for n in range(int(it_start),int(it_start)+int(usr_itr)):
     ##############################################################
     #First we plot the ICs and then solve. 
     ##############################################################
     if j>=1:
         #############################################################
         if j>=2:
             ##############################################################
             #Read the save displacements
             ##############################################################
             u__ = Function(VV1,"displacement/u%d.xml" %(n-1))
             dis = u__
             displ = project(dis,VV1)
             ALE.move(mesh,displ)
             #############################################################

         ##############################################################
         #Loop info update and printing
         ##############################################################
         print(n,flush=True)
         ##############################################################

         ##############################################################
         #Reading the last U_n only if it is not the first run
         ##############################################################
         if j==1 and n!=1:
             U_n = Function(Mixed_Space,"Biology/U%d.xml" %(n-1))
             Tn_n, Th_n, Tc_n, Tr_n, Dn_n, D_n, M_n, C_n, N_n, H_n, mu1_n, mu2_n, Igamma_n, Gbeta_n= U_n.split()
             print('Read the U_n for n=%d!' %(n-1))
         ##############################################################

         ##############################################################
         #Update biology PDE and solve
         ##############################################################
         F1 = ((Tn-Tn_n)/k)*v1*dx-(theta[0]-theta[1]*(theta[2]*D+theta[3]*M+theta[4]*mu1)*Tn-theta[5]*(theta[6]*Th+theta[7]*D)*Tn-theta[8]*(theta[9]*Th+theta[10]*mu2+theta[11]*Gbeta)*Tn-theta[12]*Tn)*v1*dx\
         + ((Th-Th_n)/k)*v2*dx+Dfc[0]*dot(grad(Th),grad(v2))*dx+coeff*Th*div(u__)*v2*dx-((theta[2]*D+theta[3]*M+theta[2]*mu1)*Tn-(theta[13]*mu2+theta[14]*Tr+theta[15])*Th)*v2*dx-Src[0]*Source1*v2*dx\
         + ((Tc-Tc_n)/k)*v3*dx+Dfc[1]*dot(grad(Tc),grad(v3))*dx+coeff*Tc*div(u__)*v3*dx-((theta[6]*Th+theta[7]*D)*Tn-(theta[16]*mu2+theta[17]*Tr+theta[18])*Tc)*v3*dx-Src[1]*Source2*v3*dx\
         + ((Tr-Tr_n)/k)*v4*dx+Dfc[2]*dot(grad(Tr),grad(v4))*dx+coeff*Tr*div(u__)*v4*dx-((theta[9]*Th+theta[10]*mu2+theta[11]*Gbeta)*Tn-(theta[19]*mu1+theta[20])*Tr)*v4*dx-Src[2]*Source3*v4*dx\
         + ((Dn-Dn_n)/k)*v5*dx+Dfc[3]*dot(grad(Dn),grad(v5))*dx+coeff*Dn*div(u__)*v5*dx-(theta[21]-theta[22]*(theta[23]*H+theta[24]*C)*Dn-(theta[25]*H+theta[26])*Dn)*v5*dx-Src[3]*Source4*v5*dx\
         + ((D-D_n)/k)*v6*dx+Dfc[4]*dot(grad(D),grad(v6))*dx+coeff*D*div(u__)*v6*dx-((theta[23]*H+theta[24]*C)*Dn-(theta[25]*H+theta[27]*C+theta[26])*D)*v6*dx-Src[4]*Source5*v6*dx\
         + ((M-M_n)/k)*v7*dx+Dfc[5]*dot(grad(M),grad(v7))*dx+coeff*M*div(u__)*v7*dx-((theta[28]*mu2+theta[29]*Igamma+theta[30]*Th)*(theta[31]-M)-theta[32]*M)*v7*dx-Src[5]*Source6*v7*dx\
         + ((C-C_n)/k)*v8*dx+Dfc[6]*dot(grad(C),grad(v8))*dx+coeff*C*div(u__)*v8*dx-((theta[33]+theta[34]*mu1)*C*(Constant(1)-C/theta[35])-(theta[36]*Gbeta+theta[37]*Igamma+theta[38]*Tc+theta[39])*C)*v8*dx\
         + ((N-N_n)/k)*v9*dx+Dfc[7]*dot(grad(N),grad(v9))*dx+coeff*N*div(u__)*v9*dx-(theta[40]*(theta[36]*Gbeta+theta[37]*Igamma+theta[38]*Tc+theta[39])*C-theta[41]*N)*v9*dx\
         + ((H-H_n)/k)*v10*dx+Dfc[8]*dot(grad(H),grad(v10))*dx-(theta[42]*N+theta[43]*M+theta[44]*Th+theta[45]*Tc+theta[46]*Tr-theta[47]*H)*v10*dx\
         + ((mu1-mu1_n)/k)*v11*dx+Dfc[9]*dot(grad(mu1),grad(v11))*dx-(theta[48]*Th+theta[49]*M+theta[50]*D-theta[51]*mu1)*v11*dx\
         + ((mu2-mu2_n)/k)*v12*dx+Dfc[10]*dot(grad(mu2),grad(v12))*dx-(theta[52]*M+theta[53]*D+theta[54]*Tr-theta[55]*mu2)*v12*dx\
         + ((Igamma-Igamma_n)/k)*v13*dx+Dfc[11]*dot(grad(Igamma),grad(v13))*dx-(theta[56]*Th+theta[57]*Tc+theta[58]*M-theta[59]*Igamma)*v13*dx\
         + ((Gbeta-Gbeta_n)/k)*v14*dx+Dfc[12]*dot(grad(Gbeta),grad(v14))*dx-(theta[60]*M+theta[61]*Tr-theta[62]*Gbeta)*v14*dx

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
         tot_im = project(Th_n+Tc_n+Tr_n+Dn_n+D_n+M_n,S1)
         #######################################################################

     #######################################################################
     #Plotting the dynamics for C every 10 steps. (Optional)
     #######################################################################
     if j%10==0:
           ##############################################################
           _1, _2, _3, _4, _5, _6, _7, CC, _8, _9, _10, _11, _12, _13= U_n.split()
           CC.rename('C_n','C_n')
           vtkfile.write(CC,t)
           ##############################################################

     ##############################################################
     j+=1
     ##############################################################

##############################################################
#Saving the last U_n for the next sensitivity range
##############################################################
File("Biology/U%d.xml" %(n))<<U
##############################################################

##############################################################
#Sensitivity Analysis using adjoint method
##############################################################

##############################################################
#Define your functional
##############################################################
if int(answer)==1:
    J1 = assemble(C_n*dx)
elif int(answer)==2:
    J1 = assemble((C_n/tot_im)*dx)
##############################################################

##############################################################
print('Assemble and Controls done!')
##############################################################

##############################################################
#Computing sensitivities
##############################################################
print('\n This will take a very long time!!')

SS = []

for idx in range(len(Pars_ordered)):
    dJdp = compute_gradient(J1, Control(theta[idx]), options={"riesz_representation": "L2"})
    SS.append(max(project(dJdp,S1).vector()[:]))
    print(idx+1, 'Gradient done!')
    c=csv.writer(open('sensitivities_cancer'+it_start+'.csv',"w"))
    c.writerow(SS)
    del c
##############################################################
##############################################################
for idx in range(len(Dfc)):
    dJdD = compute_gradient(J1, Control(Dfc[idx]), options={"riesz_representation": "L2"})
    SS.append(max(project(dJdD,S1).vector()[:]))
    print(idx+1, 'Diffusion Gradient done!')
    c=csv.writer(open('sensitivities_cancer'+it_start+'.csv',"w"))
    c.writerow(SS)
    del c
##############################################################
for idx in range(len(Src)):
    dJdS = compute_gradient(J1, Control(Src[idx]), options={"riesz_representation": "L2"})
    SS.append(max(project(dJdS,S1).vector()[:]))
    print(idx+1, 'Source Gradient done!')
    c=csv.writer(open('sensitivities_cancer'+it_start+'.csv',"w"))
    c.writerow(SS)
    del c
#######################################################################

#######################################################################
#Print the time table
#######################################################################
list_timings(TimingClear.clear, [TimingType.wall])
#######################################################################
