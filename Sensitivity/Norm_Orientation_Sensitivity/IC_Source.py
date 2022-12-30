'''
Investigating the spatial interaction of immune cells in colon cancer

IC_Source: Sets up initial conditions and source terms for the system variables

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei

(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from dolfin import *
from dolfin_adjoint import *
import numpy as np
import pandas as pd
###############################################################

def IC_Source(Mixed_Space,S1,U_n,Input2):
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
  #Creating the sources for the immune cells
  ###############################################################
  ss1= Expression('(x[0]-0.1875)*(x[0]-0.1875) + (x[1]-0.32476)*(x[1]-0.32476)', degree=0)
  source1 = conditional(lt(ss1,0.005),0.5,0)
  Source1 = project(source1,S1)
  i_source1=np.argwhere(Source1.vector().get_local()[:]<=0)  #making negatives zero
  Source1.vector()[i_source1[:,0]] = 1.e-16
  File('test/source1.pvd')<<Source1

  ss2= Expression('(x[0]+0.1875)*(x[0]+0.1875) + (x[1]-0.32476)*(x[1]-0.32476)', degree=0)
  source2 = conditional(lt(ss2,0.005),0.5,0)
  Source2 = project(source2,S1)
  i_source2=np.argwhere(Source2.vector().get_local()[:]<=0)  #making negatives zero
  Source2.vector()[i_source2[:,0]] = 1.e-16
  File('test/source2.pvd')<<Source2

  ss3= Expression('(x[0]+0.375)*(x[0]+0.375) + (x[1])*(x[1])', degree=0)
  source3 = conditional(lt(ss3,0.005),0.5,0)
  Source3 = project(source3,S1)
  i_source3=np.argwhere(Source3.vector().get_local()[:]<=0)  #making negatives zero
  Source3.vector()[i_source3[:,0]] = 1.e-16
  File('test/source3.pvd')<<Source3

  ss4= Expression('(x[0]+0.1875)*(x[0]+0.1875) + (x[1]+0.32476)*(x[1]+0.32476)', degree=0)
  source4 = conditional(lt(ss4,0.005),0.5,0)
  Source4 = project(source4,S1)
  i_source4=np.argwhere(Source4.vector().get_local()[:]<=0)  #making negatives zero
  Source4.vector()[i_source4[:,0]] = 1.e-16
  File('test/source4.pvd')<<Source4

  ss5= Expression('(x[0]-0.1875)*(x[0]-0.1875) + (x[1]+0.32476)*(x[1]+0.32476)', degree=0)
  source5 = conditional(lt(ss5,0.005),0.5,0)
  Source5 = project(source5,S1)
  i_source5=np.argwhere(Source5.vector().get_local()[:]<=0)  #making negatives zero
  Source5.vector()[i_source5[:,0]] = 1.e-16
  File('test/source5.pvd')<<Source5

  ss6= Expression('(x[0]-0.375)*(x[0]-0.375) + (x[1])*(x[1])', degree=0)
  source6 = conditional(lt(ss6,0.005),0.5,0)
  Source6 = project(source6,S1)
  i_source6=np.argwhere(Source6.vector().get_local()[:]<=0)  #making negatives zero
  Source6.vector()[i_source6[:,0]] = 1.e-16
  File('test/source6.pvd')<<Source6

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

  return Source1,Source2,Source3,Source4,Source5,Source6, U_n
