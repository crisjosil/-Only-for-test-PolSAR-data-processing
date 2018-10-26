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
dr = 300 
da = 300
offr = 500 
offa = 500

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

#sar.vis4(np.abs(hh1), np.abs(vv1), np.abs(hv1), np.zeros(np.shape(hh1)),
#         title1 = 'Magnitude HH: First', 
#         title2 = 'Magnitude VV: First',
#         title3 = 'Magnitude HV: First')
#
#sar.vis4(np.abs(hh2), np.abs(vv2), np.abs(hv2), np.zeros(np.shape(hh2)),
#         title1 = 'Magnitude HH: Second', 
#         title2 = 'Magnitude VV: Second',
#         title3 = 'Magnitude HV: Second')


###################### Coherency matrices #########################
win = 9

[Taa1, Tbb1, Tcc1, Tab1, Tac1, Tbc1] = sar.Tmatrix(hh1, hv1, vv1, win)
[Taa2, Tbb2, Tcc2, Tab2, Tac2, Tbc2] = sar.Tmatrix(hh2, hv2, vv2, win)


#sar.visRGB(Taa1, Tbb1, Tcc1, title = 'RGB image: First')
#sar.visRGB(Taa2, Tbb2, Tcc2, title = 'RGB image: Second')

sar.vis2RGB(Taa1, Tbb1, Tcc1, Taa2, Tbb2, Tcc2)


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
#Diff = np.zeros([dr,da])
#normR = np.zeros([dr,da])

vec  = np.matrix(np.zeros([3,3],dtype=np.complex64))

I = np.matrix(np.identity(3))

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

        T = (T11 - T22)/(T11.trace()[0,0]+T22.trace()[0,0])
        invT22 = ln.inv(T22)
        A = np.dot(invT22, T11)
        [d1, v1] = ln.eigh(T)
        [d2, v2] = ln.eig(A)
        
#        [w,d]=eigs(A);
##         wmax(:,m,n) = w(:,1); 
        ind = np.argsort(d1)
        lam1[m,n] = d1[ind[2]]
        lam2[m,n] = d1[ind[1]]
        lam3[m,n] = d1[ind[0]]

        ind = np.argsort(np.abs(d2))
        sig1[m,n] = np.abs(d2[ind[2]])
        sig2[m,n] = np.abs(d2[ind[1]])
        sig3[m,n] = np.abs(d2[ind[0]])

#        vec[:,0] = v[:,ind[2]]        
#        vec[:,1] = v[:,ind[1]]        
#        vec[:,2] = v[:,ind[0]]

#        P = ln.sqrtm(A.getH()*A)
#        R = A*ln.inv(P)   
#        Hpart = (A+A.getH())/2
        
#        normR[m,n] = ln.norm(R)
#        Diff[m,n] = np.abs((R - I).trace()[0,0])
        
    if np.remainder(m,10) == 0:
        print(np.int(dr)-m)


#lam1[lam1 > 100*np.mean(lam1)] = np.mean(lam1)
#lam2[lam2 > 100*np.mean(lam2)] = np.mean(lam2)
#lam3[lam3 > 100*np.mean(lam3)] = np.mean(lam3)

sig1[sig1 > 100*np.mean(sig1)] = np.mean(sig1)
sig2[sig2 > 100*np.mean(sig2)] = np.mean(sig2)
sig3[sig3 > 100*np.mean(sig3)] = np.mean(sig3)

sar.vis4(lam1, lam2, lam3, np.zeros(np.shape(hh1)),
#sar.vis4(lam1, lam2, lam3, normR,
         title1 = 'Largest eigenvalue: DIFFERENCE', 
         title2 = 'Middle eigenvalue: DIFFERENCE',
         title3 = 'Smallest eigenvalue: DIFFERENCE',
         title4 = 'Diff or R from identity',
         scale1 = [-1, 1],
         scale2 = [np.min(lam2), np.max(lam2)],
         scale3 = [np.min(lam3), np.max(lam3)],
         colormap = 'jet')

sar.vis4(sig1, sig2, sig3, np.zeros(np.shape(hh1)),
#sar.vis4(lam1, lam2, lam3, normR,
         title1 = 'Largest eigenvalue: RATIO', 
         title2 = 'Middle eigenvalue: RATIO',
         title3 = 'Smallest eigenvalue: RATIO',
         title4 = 'Diff or R from identity: RATIO')
         
         
######################### Test that it hermitian
#sigma = np.matrix(np.zeros([3,3],dtype=np.complex64))
##vec = np.matrix(vec)
#sigma[0,0] = np.abs(d[ind[2]])
#sigma[1,1] = np.abs(d[ind[1]])
#sigma[2,2] = np.abs(d[ind[0]])
#
#Arec = vec*sigma*vec.getH()



