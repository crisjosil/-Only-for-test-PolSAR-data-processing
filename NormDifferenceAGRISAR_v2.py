# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 14:25:03 2016

@author: am37653
"""
import sys
sys.path.insert(0, 'C:\\OULocal\\Programs\\SAR\\')

import SAR_Utilities as sar

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as ln
#from scipy import signal
#import spectral.io.envi as envi


plt.close('all')

#####################  First image ####################################
path1 = "C:\\OULocal\\Data\\AGRISAR\\2006_04_19\\"
fileHH1 = "i06agrsar0209x1_ch1_t20_slc.dat.final"
fileVV1 = "i06agrsar0209x1_ch3_t20_slc.dat.final"
fileHV1 = "i06agrsar0209x1_ch2_t20_slc.dat.final"
hhFull1 = sar.OpenBin(path1 + fileHH1)
vvFull1 = sar.OpenBin(path1 + fileVV1)
hvFull1 = sar.OpenBin(path1 + fileHV1)

#####################  Second image ####################################
path2 = "C:\\OULocal\\Data\\AGRISAR\\2006_06_13\\"
fileHH2 = "i06agrsar1011x1_ch1_t20_slc.dat.final"
fileVV2 = "i06agrsar1011x1_ch3_t20_slc.dat.final"
fileHV2 = "i06agrsar1011x1_ch2_t20_slc.dat.final"
hhFull2 = sar.OpenBin(path2 + fileHH2)
vvFull2 = sar.OpenBin(path2 + fileVV2)
hvFull2 = sar.OpenBin(path2 + fileHV2)


################## Considering a crop of the image #####################
dr = 100    # paper 1000
da = 100    # paper 1000
offr = 500 
offa = 1500
figName = 'test2'

hh1 = hhFull1[offr-dr/2:offr+dr/2-1, offa-da/2:offa+da/2-1]
vv1 = vvFull1[offr-dr/2:offr+dr/2-1, offa-da/2:offa+da/2-1]
hv1 = hvFull1[offr-dr/2:offr+dr/2-1, offa-da/2:offa+da/2-1]
hhFull1 = 0 
vvFull1 = 0 
hvFull1 = 0

hh2 = hhFull2[offr-dr/2:offr+dr/2-1, offa-da/2:offa+da/2-1]
vv2 = vvFull2[offr-dr/2:offr+dr/2-1, offa-da/2:offa+da/2-1]
hv2 = hvFull2[offr-dr/2:offr+dr/2-1, offa-da/2:offa+da/2-1]
hhFull2 = 0 
vvFull2 = 0 
hvFull2 = 0


###################### Coherency matrices #########################
win = 9

[Taa1, Tbb1, Tcc1, Tab1, Tac1, Tbc1] = sar.Tmatrix(hh1, hv1, vv1, win)
[Taa2, Tbb2, Tcc2, Tab2, Tac2, Tbc2] = sar.Tmatrix(hh2, hv2, vv2, win)

#sar.visRGB(Taa1, Tbb1, Tcc1, title = 'RGB image: First')
#sar.visRGB(Taa2, Tbb2, Tcc2, title = 'RGB image: Second')

Trace1 = (Taa1 + Tbb1 + Tcc1)/3
sar.vis2RGB(Tbb1, Tcc1, Taa1, Tbb2, Tcc2, Taa2,
            scalea1 = np.abs(Trace1).mean()*1.5,
            scaleb1 = np.abs(Trace1).mean()*1.5,
            scalec1 = np.abs(Trace1).mean()*1.5,
            scalea2 = np.abs(Trace1).mean()*1.5,
            scaleb2 = np.abs(Trace1).mean()*1.5,
            scalec2 = np.abs(Trace1).mean()*1.5,
            flag = 0,
            outall = 'C:\\OULocal\\Conferences\\2017\\POLinSAR17\\Images\\RGB_'+figName+'.png')

#dddd

################# Change detection #####################         
# Initialisations
T11 = np.matrix(np.zeros([3,3],dtype=np.complex64)) 
T22 = np.matrix(np.zeros([3,3],dtype=np.complex64))
lam1 = np.zeros([dr,da]) 
lam2 = np.zeros([dr,da])
lam3 = np.zeros([dr,da])
sig1 = np.zeros([dr,da]) 
sig2 = np.zeros([dr,da])
sig3 = np.zeros([dr,da])
alpha1 = np.zeros([dr,da]) 
alpha2 = np.zeros([dr,da])
alpha3 = np.zeros([dr,da])
beta1 = np.zeros([dr,da]) 
beta2 = np.zeros([dr,da])
beta3 = np.zeros([dr,da])
a1 = np.zeros([dr,da]) 
a2 = np.zeros([dr,da])
a3 = np.zeros([dr,da])
b1 = np.zeros([dr,da]) 
b2 = np.zeros([dr,da])
b3 = np.zeros([dr,da])
indep = np.zeros([dr,da])

for m in range(0, np.int(dr)-1):
    for n in range(0, np.int(da)-1):
        
        T11[0,0] = Taa1[m,n]
        T11[0,1] = Tab1[m,n] 
        T11[0,2] = Tac1[m,n]
        T11[1,0] = np.conj(Tab1[m,n])
        T11[1,1] = Tbb1[m,n]
        T11[1,2] = Tbc1[m,n]
        T11[2,0] = np.conj(Tac1[m,n])
        T11[2,1] = np.conj(Tbc1[m,n])
        T11[2,2] = Tcc1[m,n]

        T22[0,0] = Taa2[m,n]
        T22[0,1] = Tab2[m,n]
        T22[0,2] = Tac2[m,n]
        T22[1,0] = np.conj(Tab2[m,n])
        T22[1,1] = Tbb2[m,n]
        T22[1,2] = Tbc2[m,n]
        T22[2,0] = np.conj(Tac2[m,n])
        T22[2,1] = np.conj(Tbc2[m,n])
        T22[2,2] = Tcc2[m,n]

        # Optimisation for Normalised Difference
        Tc = (T22 - T11)/(T11.trace()[0,0]+T22.trace()[0,0])
        [d1, v1] = ln.eigh(Tc)
        
#        # Optimisation for Power Ratio
#        invT22 = ln.inv(T22)
#        A = np.dot(invT22, T11)
#        [d2, v2] = ln.eig(A)
        
        # Eigenvalues for Normalised Difference        
        ind = np.argsort(d1)
        lam1[m,n] = d1[ind[2]]
        lam2[m,n] = d1[ind[1]]
        lam3[m,n] = d1[ind[0]]

        # Dependence
        Te = np.dot(Tc.getH(), T11)
        ind = Te.trace()/(Tc.trace()*T11.trace())
        indep[m,n] = np.abs(ind)
        
#        # Eigenvalues for Power Ratio
#        ind = np.argsort(np.abs(d2))
#        sig1[m,n] = np.abs(d2[ind[2]])
#        sig2[m,n] = np.abs(d2[ind[1]])
#        sig3[m,n] = np.abs(d2[ind[0]])

        # Alpha and beta for Normalised Difference
        alpha1[m,n] = np.arccos(np.abs(v1[0,2]))
        alpha2[m,n] = np.arccos(np.abs(v1[0,1]))
        alpha3[m,n] = np.arccos(np.abs(v1[0,0]))
               
        beta1[m,n] = np.arccos(np.abs(v1[1,2])/np.sin(alpha1[m,n]))
        beta2[m,n] = np.arccos(np.abs(v1[1,1])/np.sin(alpha2[m,n]))
        beta3[m,n] = np.arccos(np.abs(v1[1,0])/np.sin(alpha3[m,n]))

#        # Alpha and beta for Power Ratio
#        a1[m,n] = np.arccos(np.abs(v2[0,2]))
#        a2[m,n] = np.arccos(np.abs(v2[0,1]))
#        a3[m,n] = np.arccos(np.abs(v2[0,0]))
#               
#        b1[m,n] = np.arccos(np.abs(v2[1,2])/np.sin(a1[m,n]))
#        b2[m,n] = np.arccos(np.abs(v2[1,1])/np.sin(a2[m,n]))
#        b3[m,n] = np.arccos(np.abs(v2[1,0])/np.sin(a3[m,n]))
        
        
           
#        P = ln.sqrtm(A.getH()*A)
#        R = A*ln.inv(P)   
#        Hpart = (A+A.getH())/2
        
#        normR[m,n] = ln.norm(R)
#        Diff[m,n] = np.abs((R - I).trace()[0,0])
        
    if np.remainder(m,10) == 0:
        print(np.int(dr)-m)




#sig1[sig1 > 100*np.mean(sig1)] = np.mean(sig1)
#sig2[sig2 > 100*np.mean(sig2)] = np.mean(sig2)
#sig3[sig3 > 100*np.mean(sig3)] = np.mean(sig3)

sar.vis4(lam1, lam2, lam3, indep,
#sar.vis4(lam1, lam2, lam3, normR,
         title1 = 'Largest eigenvalue: DIFFERENCE', 
         title2 = 'Middle eigenvalue: DIFFERENCE',
         title3 = 'Smallest eigenvalue: DIFFERENCE',
         title4 = 'Independence',
         scale1 = [np.min(lam1), np.max(lam1)],
         scale2 = [np.min(lam2), np.max(lam2)],
         scale3 = [np.min(lam3), np.max(lam3)],
         scale4 = [-1, 1],
         colormap = 'jet',
         flag = 0, 
         outall = 'C:\\OULocal\\Conferences\\2017\\POLinSAR17\\Images\\Dif_eigenvalues_'+figName+'.png')

sar.vis4(alpha1, alpha2, alpha3, np.zeros(np.shape(hh1)),
#sar.vis4(lam1, lam2, lam3, normR,
         title1 = 'Dominant alpha: DIFFERENCE', 
         title2 = 'Middle alpha: DIFFERENCE',
         title3 = 'Weakest alpha: DIFFERENCE',
         title4 = '',
         scale1 = [0, np.pi/2],
         scale2 = [0, np.pi/2],
         scale3 = [0, np.pi/2],
         colormap = 'jet',
         flag = 0, 
         outall = 'C:\\OULocal\\Conferences\\2017\\POLinSAR17\\Images\\Dif_alpha_'+figName+'.png')

sar.vis4(beta1, beta2, beta3, np.zeros(np.shape(hh1)),
#sar.vis4(lam1, lam2, lam3, normR,
         title1 = 'Dominant beta: DIFFERENCE', 
         title2 = 'Middle beta: DIFFERENCE',
         title3 = 'Weakest beta: DIFFERENCE',
         title4 = '',
         scale1 = [-np.pi, np.pi],
         scale2 = [-np.pi, np.pi],
         scale3 = [-np.pi, np.pi],
         colormap = 'jet',
         flag = 0, 
         outall = 'C:\\OULocal\\Conferences\\2017\\POLinSAR17\\Images\\Dif_beta_'+figName+'.png')
         
#sar.vis4(sig1, sig2, sig3, np.zeros(np.shape(hh1)),
##sar.vis4(lam1, lam2, lam3, normR,
#         title1 = 'Largest eigenvalue: RATIO', 
#         title2 = 'Middle eigenvalue: RATIO',
#         title3 = 'Smallest eigenvalue: RATIO',
#         title4 = '',
#         colormap = 'jet',
#         flag = 1, 
#         outall = 'C:\\OULocal\\Conferences\\2017\\POLinSAR17\\Images\\Ratio_eigenvalues_'+figName+'.png')
#         
#sar.vis4(a1, a2, a3, np.zeros(np.shape(hh1)),
##sar.vis4(lam1, lam2, lam3, normR,
#         title1 = 'Dominant alpha: RATIO', 
#         title2 = 'Middle alpha: RATIO',
#         title3 = 'Weakest alpha: RATIO',
#         title4 = '',
#         scale1 = [0, np.pi/2],
#         scale2 = [0, np.pi/2],
#         scale3 = [0, np.pi/2],
#         colormap = 'jet',
#         flag = 1, 
#         outall = 'C:\\OULocal\\Conferences\\2017\\POLinSAR17\\Images\\Ratio_alpha_'+figName+'.png')
#
#sar.vis4(b1, b2, b3, np.zeros(np.shape(hh1)),
##sar.vis4(lam1, lam2, lam3, normR,
#         title1 = 'Dominant beta: RATIO', 
#         title2 = 'Middle beta: RATIO',
#         title3 = 'Weakest beta: RATIO',
#         title4 = '',
#         scale1 = [-np.pi, np.pi],
#         scale2 = [-np.pi, np.pi],
#         scale3 = [-np.pi, np.pi],
#         colormap = 'jet',
#         flag = 1, 
#         outall = 'C:\\OULocal\\Conferences\\2017\\POLinSAR17\\Images\\Ratio_beta_'+figName+'.png')
         
         
         
         
######################### Test that it hermitian
#sigma = np.matrix(np.zeros([3,3],dtype=np.complex64))
##vec = np.matrix(vec)
#sigma[0,0] = np.abs(d[ind[2]])
#sigma[1,1] = np.abs(d[ind[1]])
#sigma[2,2] = np.abs(d[ind[0]])
#
#Arec = vec*sigma*vec.getH()



