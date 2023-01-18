'''
Investigating the spatial interaction of immune cells in colon cancer

Source_Loc_rnd: Creating random sources for immune cells

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei

(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from fenics import *
import numpy as np
import pandas as pd
import csv
import random as rnd
###############################################################
def Source_Loc_rnd(mesh,Mixed_Space,SDG,S1,U_n,Input2):

    ###############################################################
    #Initial conditions
    ###############################################################
    #Tn is outside of the environment and is modeled as ODE with uniform IC.
    #Cytokines are taken to be zero at the begining.
    #Immune cells will start at their source locations.
    #Cancer and Necrotic cells' ICs are uniform and their values are
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
    #Creating bounding box tree to be able to check what cell belongs to which mesh triangle
    ###############################################################
    X = mesh.coordinates()
    tree = BoundingBoxTree()
    tree.build(mesh)
    Num_cell = mesh.num_cells()
    ############################################################################

    ###############################################################
    #Functions on the fine mesh
    ###############################################################
    Th = Function(SDG)
    Tc = Function(SDG)
    Tr = Function(SDG)
    Dn = Function(SDG)
    D  = Function(SDG)
    M  = Function(SDG)
    ############################################################################

    ###############################################################
    #Creating the random cell sources
    ###############################################################
    id1=10000000000000;
    id2=10000000000000;
    id3=10000000000000;
    id4=10000000000000;
    id5=10000000000000;
    id6=10000000000000;
    for i in range(100):

        while id1 > Num_cell:
            p1 = Point(rnd.uniform(-0.504,0.501),rnd.uniform(-0.496,0.458),0)
            id1 = tree.compute_first_entity_collision(p1)
        Th.vector()[id1] = Th.vector().get_local()[id1]+1
        while id2 > Num_cell:
            p2 = Point(rnd.uniform(-0.504,0.501),rnd.uniform(-0.496,0.458),0)
            id2 = tree.compute_first_entity_collision(p2)
        Tc.vector()[id2] = Tc.vector().get_local()[id2]+1
        while id3 > Num_cell:
            p3 = Point(rnd.uniform(-0.504,0.501),rnd.uniform(-0.496,0.458),0)
            id3 = tree.compute_first_entity_collision(p3)
        Tr.vector()[id3] = Tr.vector().get_local()[id3]+1
        while id4 > Num_cell:
            p4 = Point(rnd.uniform(-0.504,0.501),rnd.uniform(-0.496,0.458),0)
            id4 = tree.compute_first_entity_collision(p4)
        Dn.vector()[id4] = Dn.vector().get_local()[id4]+1
        while id5 > Num_cell:
            p5 = Point(rnd.uniform(-0.504,0.501),rnd.uniform(-0.496,0.458),0)
            id5 = tree.compute_first_entity_collision(p5)
        D.vector()[id5] = D.vector().get_local()[id5]+1
        while id6 > Num_cell:
            p6 = Point(rnd.uniform(-0.504,0.501),rnd.uniform(-0.496,0.458),0)
            id6 = tree.compute_first_entity_collision(p6)
        M.vector()[id6] = M.vector().get_local()[id6]+1
        id1=10000000000000;
        id2=10000000000000;
        id3=10000000000000;
        id4=10000000000000;
        id5=10000000000000;
        id6=10000000000000;
        ############################################################################
    ###############################################################

    ###############################################################
    #Max values
    ###############################################################
    maxTh = max(Th.vector()[:])
    maxTc = max(Tc.vector()[:])
    maxTr = max(Tr.vector()[:])
    maxDn = max(Dn.vector()[:])
    maxD = max(D.vector()[:])
    maxM = max(M.vector()[:])


    Th.vector()[:]/=maxTh
    Tc.vector()[:]/=maxTc
    Tr.vector()[:]/=maxTr
    Dn.vector()[:]/=maxDn
    D.vector()[:]/=maxD
    M.vector()[:]/=maxM
    # ############################################################################

    ###############################################################
    #Save plots
    ###############################################################
    Source1 = project(Th,S1)
    i_source1=np.argwhere(Source1.vector().get_local()[:]<=0)  #making negatives zero
    Source1.vector()[i_source1[:,0]] = 1.e-16
    File('test/source1.pvd')<<Source1


    Source2 = project(Tc,S1)
    i_source2=np.argwhere(Source2.vector().get_local()[:]<=0)  #making negatives zero
    Source2.vector()[i_source2[:,0]] = 1.e-16
    File('test/source2.pvd')<<Source2


    Source3 = project(Tr,S1)
    i_source3=np.argwhere(Source3.vector().get_local()[:]<=0)  #making negatives zero
    Source3.vector()[i_source3[:,0]] = 1.e-16
    File('test/source3.pvd')<<Source3


    Source4 = project(Dn,S1)
    i_source4=np.argwhere(Source4.vector().get_local()[:]<=0)  #making negatives zero
    Source4.vector()[i_source4[:,0]] = 1.e-16
    File('test/source4.pvd')<<Source4


    Source5 = project(D,S1)
    i_source5=np.argwhere(Source5.vector().get_local()[:]<=0)  #making negatives zero
    Source5.vector()[i_source5[:,0]] = 1.e-16
    File('test/source5.pvd')<<Source5


    Source6 = project(M,S1)
    i_source6=np.argwhere(Source6.vector().get_local()[:]<=0)  #making negatives zero
    Source6.vector()[i_source6[:,0]] = 1.e-16
    File('test/source6.pvd')<<Source6
    ############################################################################
    ###############################################################
    #Assigning projected ICs to subspace as the initial step of iteration.
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
    ###############################################################
    return Source1,Source2,Source3,Source4,Source5,Source6,U_n
###############################################################
