'''
qspmodel: classes and methods for analysis of
          Mathematical model of immune response in cancer*
See: https://github.com/ShahriyariLab/Data-driven-mathematical-model-for-colon-cancer
Colon_QSP_Functions: class containing functions and jacobians related to
            Data-driven Mathematical model of immune response in colon cancer
QSP: general class containing methods for analysis of
          Mathematical model of immune response in cancer

Author: Arkadz Kirshtein, https://sites.google.com/site/akirshtein/
(c) Shahriyari Lab https://sites.google.com/site/leilishahriyari/

Conceptualization of sensitivity algorithm by Wenrui Hao, http://personal.psu.edu/wxh64/


*part of https://github.com/ShahriyariLab/Data-driven-mathematical-model-for-colon-cancer
 If using any part of this code please cite
  A. Kirshtein, S. Akbarinejad, W. Hao, T. Le, S. Su, R. Aronow, L. Shahriyari,
  Data driven mathematical model of colon cancer progression, Journal of Clinical Medicine, 2020.
'''

import numpy as np
import scipy.optimize as op
from itertools import permutations
from scipy.integrate import odeint


# time derivative for sensitivity ODE
def Sensitivity_function(t,x,QSP):
    nparam=QSP.qspcore.nparam
    nvar=QSP.qspcore.nvar
    x=x.reshape(nparam+1,nvar)
    dxdt=np.empty(x.shape)
    dxdt[0]=QSP(t,x[0])
    dxdt[1:]=np.dot(QSP.Ju(t,x[0]),x[1:].T).T+ QSP.Jp(t,x[0])
    return dxdt.flatten()

# Object class that defines the functions for the appropriate QSP Model
class Colon_QSP_Functions(object):
    nparam=63
    nvar=14
    variable_names=['Naive T-cells', 'helper T-cells', 'cytotoxic cells', 'Treg-cells', 'Naive Dendritic cells', 'Dendritic cells', 'Macrophages',
                                     'Cancer', 'Necrotic cells', '\mu_1', '\mu_2', 'HMGB1', 'IFN-\gamma', 'TNF-\beta']
    parameter_names=['\lambda_{T_hD}',  '\lambda_{T_hM}',  '\lambda_{T_h\mu_1}', '\lambda_{T_CT_h}','\lambda_{T_CD}', '\lambda_{T_rT_h}', '\lambda_{T_r\mu_2}','\lambda_{T_rG_\beta}',
                                      '\lambda_{DH}',   '\lambda_{DC}',     '\lambda_{M\mu_2}',   '\lambda_{MI_\gamma}', '\lambda_{MT_h}', '\lambda_{C}',    '\lambda_{C\mu_1}', '\alpha_{NC}',
                                      '\lambda_{\mu_1T_h}','\lambda_{\mu_1M}',   '\lambda_{\mu_1D}',   '\lambda_{\mu_2M}','\lambda_{\mu_2D}','\lambda_{\mu_2T_r}',
                                      '\lambda_{HN}',   '\lambda_{HM}',     '\lambda_{HT_h}',    '\lambda_{HT_C}', '\lambda_{HT_r}', '\lambda_{I_\gammaT_h}', '\lambda_{I_\gammaT_C}', '\lambda_{I_\gammaM}',
                                      '\lambda_{G_\betaM}',  '\lambda_{G_\betaT_r}',   '\delta_{T_N}',      '\delta_{T_h\mu_2}','\delta_{T_hT_r}', '\delta_{T_h}',    '\delta_{T_C\mu_2}', '\delta_{T_CT_r}',
                                      '\delta_{T_C}',    '\delta_{T_r\mu_1}',   '\delta_{T_r}',      '\delta_{DH}',   '\delta_{DC}',   '\delta_{D}',     '\delta_{M}',     '\delta_{CG_\beta}','\delta_{CI_\gamma}','\delta_{CT_C}',
                                      '\delta_{C}',     '\delta_{N}',       '\delta_{\mu_1}',     '\delta_{\mu_2}',  '\delta_{H}',     '\delta_{I_\gamma}',    '\delta_{G_\beta}',
                                      'A_{T_N}','A_{Dn}','M_0','C_0', '\alpha_{T_NT_h}', '\alpha_{T_NT_C}', '\alpha_{T_NT_r}', '\alpha_{D_ND}']
    def __call__(self,t,x,par):
        # ODE right-hand side
        return np.array([par[55] - par[32]*x[0] - par[60]*x[0]*(par[3]*x[1] + par[4]*x[5]) - \
                  par[59]*x[0]*(par[0]*x[5] + par[1]*x[6] + par[2]*x[9]) - \
                  par[61]*x[0]*(par[5]*x[1] + par[6]*x[10] + par[7]*x[13]), \
                  x[0]*(par[0]*x[5] + par[1]*x[6] + par[2]*x[9]) - x[1]*(par[35] + par[34]*x[3] + par[33]*x[10]), \
                  x[0]*(par[3]*x[1] + par[4]*x[5]) - x[2]*(par[38] + par[37]*x[3] + par[36]*x[10]), \
                  (-x[3])*(par[40] + par[39]*x[9]) + x[0]*(par[5]*x[1] + par[6]*x[10] + par[7]*x[13]), \
                  par[56] - par[62]*x[4]*(par[9]*x[7] + par[8]*x[11]) - x[4]*(par[43] + par[41]*x[11]), \
                  x[4]*(par[9]*x[7] + par[8]*x[11]) - x[5]*(par[43] + par[42]*x[7] + par[41]*x[11]), \
                  (-par[44])*x[6] + (par[57] - x[6])*(par[12]*x[1] + par[10]*x[10] + par[11]*x[12]), \
                  x[7]*(1 - x[7]/par[58])*(par[13] + par[14]*x[9]) - \
                  x[7]*(par[48] + par[47]*x[2] + par[46]*x[12] + par[45]*x[13]), \
                  (-par[49])*x[8] + par[15]*x[7]*(par[48] + par[47]*x[2] + par[46]*x[12] + par[45]*x[13]), \
                  par[16]*x[1] + par[18]*x[5] + par[17]*x[6] - par[50]*x[9], par[21]*x[3] + par[20]*x[5] + \
                  par[19]*x[6] - par[51]*x[10], par[24]*x[1] + par[25]*x[2] + par[26]*x[3] + par[23]*x[6] + \
                  par[22]*x[8] - par[52]*x[11], par[27]*x[1] + par[28]*x[2] + par[29]*x[6] - par[53]*x[12], \
                  par[31]*x[3] + par[30]*x[6] - par[54]*x[13]])
    def Ju(self,t,x,par):
        # Jacobian with respect to variables
        return np.array([[-par[32] - par[60]*(par[3]*x[1] + par[4]*x[5]) - par[59]*(par[0]*x[5] + par[1]*x[6] + par[2]*x[9]) - \
                       par[61]*(par[5]*x[1] + par[6]*x[10] + par[7]*x[13]), (-par[3])*par[60]*x[0] - par[5]*par[61]*x[0], \
                       0, 0, 0, (-par[0])*par[59]*x[0] - par[4]*par[60]*x[0], (-par[1])*par[59]*x[0], 0, 0, \
                       (-par[2])*par[59]*x[0], (-par[6])*par[61]*x[0], 0, 0, (-par[7])*par[61]*x[0]], \
                       [par[0]*x[5] + par[1]*x[6] + par[2]*x[9], -par[35] - par[34]*x[3] - par[33]*x[10], 0, (-par[34])*x[1], \
                       0, par[0]*x[0], par[1]*x[0], 0, 0, par[2]*x[0], (-par[33])*x[1], 0, 0, 0], \
                       [par[3]*x[1] + par[4]*x[5], par[3]*x[0], -par[38] - par[37]*x[3] - par[36]*x[10], (-par[37])*x[2], 0, \
                       par[4]*x[0], 0, 0, 0, 0, (-par[36])*x[2], 0, 0, 0], [par[5]*x[1] + par[6]*x[10] + par[7]*x[13], \
                       par[5]*x[0], 0, -par[40] - par[39]*x[9], 0, 0, 0, 0, 0, (-par[39])*x[3], par[6]*x[0], 0, 0, \
                       par[7]*x[0]], [0, 0, 0, 0, -par[43] - par[41]*x[11] - par[62]*(par[9]*x[7] + par[8]*x[11]), 0, 0, \
                       (-par[9])*par[62]*x[4], 0, 0, 0, (-par[41])*x[4] - par[8]*par[62]*x[4], 0, 0], \
                       [0, 0, 0, 0, par[9]*x[7] + par[8]*x[11], -par[43] - par[42]*x[7] - par[41]*x[11], 0, \
                       par[9]*x[4] - par[42]*x[5], 0, 0, 0, par[8]*x[4] - par[41]*x[5], 0, 0], \
                       [0, par[12]*(par[57] - x[6]), 0, 0, 0, 0, -par[44] - par[12]*x[1] - par[10]*x[10] - par[11]*x[12], 0, \
                       0, 0, par[10]*(par[57] - x[6]), 0, par[11]*(par[57] - x[6]), 0], \
                       [0, 0, (-par[47])*x[7], 0, 0, 0, 0, -par[48] - par[47]*x[2] - (x[7]*(par[13] + par[14]*x[9]))/\
                       par[58] + (1 - x[7]/par[58])*(par[13] + par[14]*x[9]) - par[46]*x[12] - par[45]*x[13], 0, \
                       par[14]*x[7]*(1 - x[7]/par[58]), 0, 0, (-par[46])*x[7], (-par[45])*x[7]], \
                       [0, 0, par[15]*par[47]*x[7], 0, 0, 0, 0, par[15]*(par[48] + par[47]*x[2] + par[46]*x[12] + \
                       par[45]*x[13]), -par[49], 0, 0, 0, par[15]*par[46]*x[7], par[15]*par[45]*x[7]], \
                       [0, par[16], 0, 0, 0, par[18], par[17], 0, 0, -par[50], 0, 0, 0, 0], \
                       [0, 0, 0, par[21], 0, par[20], par[19], 0, 0, 0, -par[51], 0, 0, 0], \
                       [0, par[24], par[25], par[26], 0, 0, par[23], 0, par[22], 0, 0, -par[52], 0, 0], \
                       [0, par[27], par[28], 0, 0, 0, par[29], 0, 0, 0, 0, 0, -par[53], 0], \
                       [0, 0, 0, par[31], 0, 0, par[30], 0, 0, 0, 0, 0, 0, -par[54]]])
    def Jp(self,t,x,par):
        # Jacobian with respect to the parameters
        return np.array([[(-par[59])*x[0]*x[5], x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [(-par[59])*x[0]*x[6], x[0]*x[6], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [(-par[59])*x[0]*x[9], x[0]*x[9], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [(-par[60])*x[0]*x[1], 0, x[0]*x[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [(-par[60])*x[0]*x[5], 0, x[0]*x[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [(-par[61])*x[0]*x[1], 0, 0, x[0]*x[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [(-par[61])*x[0]*x[10], 0, 0, x[0]*x[10], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [(-par[61])*x[0]*x[13], 0, 0, x[0]*x[13], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, (-par[62])*x[4]*x[11], x[4]*x[11], 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, (-par[62])*x[4]*x[7], x[4]*x[7], 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, (par[57] - x[6])*x[10], 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, (par[57] - x[6])*x[12], 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, x[1]*(par[57] - x[6]), 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, x[7]*(1 - x[7]/par[58]), 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, x[7]*(1 - x[7]/par[58])*x[9], 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, x[7]*(par[48] + par[47]*x[2] + par[46]*x[12] + par[45]*x[13]), 0, 0, 0, 0, \
                      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, x[6], 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[6], 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[5], 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[3], 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[8], 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[6], 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[3], 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[1], 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[2], 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[6], 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[6]], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, x[3]], \
                      [-x[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, (-x[1])*x[10], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                      0], [0, (-x[1])*x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, -x[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
                      0, 0], [0, 0, (-x[2])*x[10], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, (-x[2])*x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, -x[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, (-x[3])*x[9], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -x[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, (-x[4])*x[11], (-x[5])*x[11], 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, (-x[5])*x[7], 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -x[4], -x[5], 0, 0, 0, 0, 0, 0, 0, \
                      0], [0, 0, 0, 0, 0, 0, -x[6], 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, (-x[7])*x[13], \
                      par[15]*x[7]*x[13], 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, (-x[7])*x[12], par[15]*x[7]*x[12], 0, 0, 0, \
                      0, 0], [0, 0, 0, 0, 0, 0, 0, (-x[2])*x[7], par[15]*x[2]*x[7], 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, -x[7], par[15]*x[7], 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, -x[8], 0, 0, 0, 0, \
                      0], [0, 0, 0, 0, 0, 0, 0, 0, 0, -x[9], 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[10], 0, 0, 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[11], 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[12], 0], \
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -x[13]], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, par[12]*x[1] + par[10]*x[10] + \
                      par[11]*x[12], 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, (x[7]**2*(par[13] + par[14]*x[9]))/\
                      par[58]**2, 0, 0, 0, 0, 0, 0], [(-x[0])*(par[0]*x[5] + par[1]*x[6] + par[2]*x[9]), 0, 0, 0, 0, 0, 0, \
                      0, 0, 0, 0, 0, 0, 0], [(-x[0])*(par[3]*x[1] + par[4]*x[5]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [(-x[0])*(par[5]*x[1] + par[6]*x[10] + par[7]*x[13]), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
                      [0, 0, 0, 0, (-x[4])*(par[9]*x[7] + par[8]*x[11]), 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    def SS_system(self,par,frac):
        # compute the system and restrictions with non-dimensional steady states at 1
        # pre-defined rates, extreme and global mean values are hardcoded here
        # cell fractions are given as [Tn Th Tc Tr Dn D M M0 C N]
        x=np.ones(self.nvar);
        # rates acquired from bio research [delTn delTh delTc delTr delD delM lamgCmax lamgCmin delmu1 delmu2 delmu3 delH delIg delGb]
        globvals=np.array([9.4951e-4, 0.231, 0.406, 0.231, 0.277, 0.02, 7.5e-3, 6.7152e-04, 1.07, 4.62, 58.7, 33.27, 499])
        # maximal values of each variable across all patients [Tn Th Tc Tr Dn D M C N mu1 mu2 H Ig Gb]
        extremevals=np.array([2.67309712e+04, 1.23113649e+04, 1.91073391e+04, 7.21023854e+03,
                              3.41730382e+03, 4.32746936e+03, 2.31601361e+04, 3.24717521e+05,
                              1.62358760e+05, 1.05768591e+03, 1.39828734e+04, 2.46968870e+04,
                              1.02214368e+02, 1.63407223e+05])
        # average values of each variable across all patients [Tc mu1 Ig Gb]
        meanvals=np.array([2920.305826, 132.316278, 6.603464, 20017.990343])



        return np.array([par[55] - par[32]*x[0] - par[60]*x[0]*(par[3]*x[1] + par[4]*x[5]) -
                         par[59]*x[0]*(par[0]*x[5] + par[1]*x[6] + par[2]*x[9]) -
                         par[61]*x[0]*(par[5]*x[1] + par[6]*x[10] + par[7]*x[13]),
                         x[0]*(par[0]*x[5] + par[1]*x[6] + par[2]*x[9]) -
                         x[1]*(par[35] + par[34]*x[3] + par[33]*x[10]),
                         x[0]*(par[3]*x[1] + par[4]*x[5]) -
                         x[2]*(par[38] + par[37]*x[3] + par[36]*x[10]),
                         (-x[3])*(par[40] + par[39]*x[9]) + x[0]*(par[5]*x[1] + par[6]*x[10] +
                         par[7]*x[13]), par[56] - par[62]*x[4]*(par[9]*x[7] + par[8]*x[11]) -
                         x[4]*(par[43] + par[41]*x[11]), x[4]*(par[9]*x[7] + par[8]*x[11]) -
                         x[5]*(par[43] + par[42]*x[7] + par[41]*x[11]),
                         (-par[44])*x[6] + (par[57] - x[6])*(par[12]*x[1] + par[10]*x[10] +
                         par[11]*x[12]), x[7]*(1 - x[7]/par[58])*(par[13] + par[14]*x[9]) -
                         x[7]*(par[48] + par[47]*x[2] + par[46]*x[12] + par[45]*x[13]),
                         (-par[49])*x[8] + par[15]*x[7]*(par[48] + par[47]*x[2] + par[46]*x[12] +
                         par[45]*x[13]), par[16]*x[1] + par[18]*x[5] + par[17]*x[6] -
                         par[50]*x[9], par[21]*x[3] + par[20]*x[5] + par[19]*x[6] - par[51]*x[10],
                         par[24]*x[1] + par[25]*x[2] + par[26]*x[3] + par[23]*x[6] + par[22]*x[8] -
                         par[52]*x[11], par[27]*x[1] + par[28]*x[2] + par[29]*x[6] -
                         par[53]*x[12], par[31]*x[3] + par[30]*x[6] - par[54]*x[13],
                         (extremevals[5]*par[0])/frac[5] - (200*extremevals[6]*par[1])/frac[6],
                         (extremevals[5]*par[0])/frac[5] - (200*extremevals[9]*par[2])/frac[10],
                         (extremevals[10]*par[33])/frac[11] - 20*par[35],
                         (extremevals[3]*par[34])/frac[3] - 20*par[35],
                         (extremevals[1]*par[3])/frac[1] - (2*extremevals[5]*par[4])/frac[5],
                         (extremevals[10]*par[36])/frac[11] - 20*par[38],
                         (extremevals[3]*par[37])/frac[3] - 20*par[38],
                         (extremevals[10]*par[6])/frac[11] - (extremevals[13]*par[7])/frac[14],
                         (extremevals[1]*par[5])/frac[1] - (4*extremevals[10]*par[6])/frac[11],
                         (extremevals[9]*par[39])/frac[10] - 20*par[40],
                         (extremevals[11]*par[8])/frac[12] - (2*extremevals[7]*par[9])/frac[8],
                         (extremevals[11]*par[41])/frac[12] - (0.5*extremevals[7]*par[42])/frac[8],
                         (extremevals[11]*par[41])/frac[12] + (extremevals[7]*par[42])/frac[8] -
                         par[43], (extremevals[10]*par[10])/frac[11] -
                         (extremevals[12]*par[11])/frac[13],
                         (10*extremevals[10]*par[10])/frac[11] - (extremevals[1]*par[12])/frac[1],
                         (extremevals[13]*par[45])/frac[14] - 10*par[48],
                         (extremevals[13]*par[45])/frac[14] - (extremevals[12]*par[46])/frac[13],
                         -((2*extremevals[13]*par[45])/frac[14]) + (extremevals[2]*par[47])/
                         frac[2], -((4*par[16])/frac[1]) + par[17]/frac[6],
                         par[16]/frac[1] - (2*par[18])/frac[5], -(par[19]/frac[6]) +
                         par[20]/frac[5], par[19]/frac[6] - par[21]/frac[3],
                         par[22]/frac[9] - (10*par[23])/frac[6], par[23]/frac[6] -
                         (2*par[24])/frac[1], par[23]/frac[6] - (2*par[25])/frac[2],
                         par[23]/frac[6] - (2*par[26])/frac[3], -((4*par[27])/frac[1]) +
                         par[28]/frac[2], par[27]/frac[1] - (5*par[29])/frac[6],
                         par[30]/frac[6] - par[31]/frac[3], -globvals[0] + par[32],
                         -globvals[1] + par[35], -globvals[2] + par[38], -globvals[3] + par[40],
                         -globvals[4] + par[43], -globvals[5] + par[44],
                         -globvals[6] + par[13] + (meanvals[1]*par[14])/frac[10] - par[48],
                         -globvals[7] + par[13] - (meanvals[3]*par[45])/frac[14] -
                         (meanvals[2]*par[46])/frac[13] - (meanvals[0]*par[47])/frac[2] - par[48],
                         -globvals[8] + par[50], -globvals[9] + par[51], -globvals[10] + par[52],
                         -globvals[11] + par[53], -globvals[12] + par[54],
                         -((0.75*frac[8])/frac[9]) + par[15], -2 + par[58],
                         -(frac[7]/frac[6]) + par[57], -(frac[1]/frac[0]) + par[59],
                         -(frac[2]/frac[0]) + par[60], -(frac[3]/frac[0]) + par[61],
                         -(frac[5]/frac[4]) + par[62]])


class QSP:
    def __init__(self,parameters,qspcore=Colon_QSP_Functions()):
        self.qspcore=qspcore
        self.par=parameters;
    def set_parameters(self,parameters):
        self.par=parameters;
    def steady_state(self):
        # compute steady state with current parameters
        IC=np.ones(self.qspcore.nvar);
        return op.fsolve((lambda x: self.qspcore(0,x,self.par)),IC,fprime=(lambda x: self.qspcore.Ju(0,x,self.par)),xtol=1e-7,maxfev=10000)
        # return op.root((lambda x,QSP: QSP(0,x,self.par)),IC,args=(self.qspcore,), method='hybr')
    def Sensitivity(self,method='steady',t=0,IC=None):
        if IC is None:
            IC=np.ones(self.qspcore.nvar)
        if method=='time':
            nparam=self.qspcore.nparam
            nvar=self.qspcore.nvar
            initial=np.zeros((nparam+1,nvar));
            initial[0]=IC
            return np.mean(odeint(Sensitivity_function, initial.flatten(), t, args=(self, ), tfirst=True) ,axis=0).reshape(nparam+1,nvar)[1:]
        else:
            u=self.steady_state()
            return -np.dot(self.qspcore.Jp(0,u,self.par),np.linalg.inv(self.qspcore.Ju(0,u,self.par).T))
    def __call__(self,t,x):
        return self.qspcore(t,x,self.par)
    def Ju(self,t,x):
        return self.qspcore.Ju(t,x,self.par)
    def Jp(self,t,x):
        return self.qspcore.Jp(t,x,self.par)
    def variable_names(self):return self.qspcore.variable_names
    def parameter_names(self):return self.qspcore.parameter_names
    def solve_ode(self, t, IC, method='default'):
        # Solve ode system with either default 1e4 time steps or given time discretization
        # t - time: for 'default' needs start and end time
        #           for 'given' needs full array of time discretization points
        # IC - initial conditions
        # method: 'default' - given interval divided by 10000 time steps
        #         'given' - given time discretization
        if method=='given':
            return odeint((lambda t,x: self.qspcore(t,x,self.par)), IC, t,
                            Dfun=(lambda t,x: self.qspcore.Ju(t,x,self.par)), tfirst=True), t
        else:
            return odeint((lambda t,x: self.qspcore(t,x,self.par)), IC, np.linspace(min(t), max(t), 10001),
                            Dfun=(lambda t,x: self.qspcore.Ju(t,x,self.par)), tfirst=True), np.linspace(min(t), max(t), 10001)
    @classmethod
    def from_cell_data(class_object, fracs, qspcore=Colon_QSP_Functions()):
        params=op.fsolve((lambda par,frac: qspcore.SS_system(par,frac)),np.ones(qspcore.nparam),args=(fracs,))
        return class_object(params,qspcore)
