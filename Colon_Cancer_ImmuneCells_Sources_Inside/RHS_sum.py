'''
Investigating the spatial interaction of immune cells in colon cancer

RHS_sum: Sets up the system's right hand side for cells

Author: Navid Mohammad Mirzaei https://sites.google.com/view/nmirzaei
                               https://github.com/nmirzaei

(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/
'''
###############################################################
#Importing required functions
###############################################################
from dolfin import *
###############################################################

def RHS_sum(U,Source,pars):
    Tn, Th, Tc, Tr, Dn, D, M, C, N, H, mu1, mu2, Igamma, Gbeta= U.split()
    RHS_Tn = pars['A_{T_N}']-pars['alpha_{T_NT_h}']*(pars['lambda_{T_hD}']*D+pars['lambda_{T_hM}']*M+pars['lambda_{T_hmu_1}']*mu1)*Tn-pars['alpha_{T_NT_C}']*(pars['lambda_{T_CT_h}']*Th+pars['lambda_{T_CD}']*D)*Tn-pars['alpha_{T_NT_r}']*(pars['lambda_{T_rT_h}']*Th+pars['lambda_{T_rmu_2}']*mu2+pars['lambda_{T_rG_beta}']*Gbeta)*Tn-pars['delta_{T_N}']*Tn
    RHS_Th = (pars['lambda_{T_hD}']*D+pars['lambda_{T_hM}']*M+pars['lambda_{T_hmu_1}']*mu1)*Tn-(pars['delta_{T_hmu_2}']*mu2+pars['delta_{T_hT_r}']*Tr+pars['delta_{T_h}'])*Th+Source[0]
    RHS_Tc = (pars['lambda_{T_CT_h}']*Th+pars['lambda_{T_CD}']*D)*Tn-(pars['delta_{T_Cmu_2}']*mu2+pars['delta_{T_CT_r}']*Tr+pars['delta_{T_C}'])*Tc+Source[1]
    RHS_Tr = (pars['lambda_{T_rT_h}']*Th+pars['lambda_{T_rmu_2}']*mu2+pars['lambda_{T_rG_beta}']*Gbeta)*Tn-(pars['delta_{T_rmu_1}']*mu1+pars['delta_{T_r}'])*Tr+Source[2]
    RHS_Dn = pars['A_{Dn}']-pars['alpha_{D_ND}']*(pars['lambda_{DH}']*H+pars['lambda_{DC}']*C)*Dn-(pars['delta_{DH}']*H+pars['delta_{D}'])*Dn+Source[3]
    RHS_D = (pars['lambda_{DH}']*H+pars['lambda_{DC}']*C)*Dn-(pars['delta_{DH}']*H+pars['delta_{DC}']*C+pars['delta_{D}'])*D+Source[4]
    RHS_M = (pars['lambda_{Mmu_2}']*mu2+pars['lambda_{MI_gamma}']*Igamma+pars['lambda_{MT_h}']*Th)*(pars['M_0']-M)-pars['delta_{M}']*M+Source[5]
    RHS_C = (pars['lambda_{C}']+pars['lambda_{Cmu_1}']*mu1)*C*(Constant(1)-C/pars['C_0'])-(pars['delta_{CG_beta}']*Gbeta+pars['delta_{CI_gamma}']*Igamma+pars['delta_{CT_C}']*Tc+pars['delta_{C}'])*C
    RHS_N = pars['alpha_{NC}']*(pars['delta_{CG_beta}']*Gbeta+pars['delta_{CI_gamma}']*Igamma+pars['delta_{CT_C}']*Tc+pars['delta_{C}'])*C-pars['delta_{N}']*N
    return RHS_Th+RHS_Tc+RHS_Tr+RHS_Dn+RHS_D+RHS_M+RHS_C+RHS_N
