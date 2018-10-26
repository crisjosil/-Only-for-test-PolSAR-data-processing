# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 12:53:08 2018

@author: crisj
"""
import sys
sys.path.insert(0, 'D:\\Juanma\\')
#import SAR_Utilities as sar
import Library_for_Donana as lib 
import numpy as np
import math as math
#import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
#import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
#import colorsys
#from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from pandas import Series, DataFrame, Panel
from skimage import color
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as ln

path1    = "D:\\Juanma\\FQ19W\\2014-06-05.rds2\\"      
path2    = "D:\\Juanma\\FQ19W\\2014-06-29.rds2\\"  
 
############################# Determine new coordinate system based on the ROI #################################################################
#subset the image to the 4 points given by the clicks

y1=3500
y3=5000
x1=4000
x3=6000

y2=y1
y4=y3
x2=x1
x4=x2
y=np.array([y1,y2,y3,y4]).astype(int) 
x=np.array([x1,x2,x3,x4]).astype(int)
   
y_min=min(y)
y_max=max(y)
x_min=min(x)
x_max=max(x)
y=y-y_min
x=x-x_min
Img_size_rows = (y_max-y_min)
Img_size_columns = (x_max-x_min)
############################# Open RGB, select irregular ROI and get coordinates #################################################################
nameC11  = "\\C11.bin"
nameC12R = "\\C12_real.bin"
nameC12I = "\\C12_imag.bin"
nameC13R = "\\C13_real.bin"
nameC13I = "\\C13_imag.bin"
nameC22  = "\\C22.bin"
nameC23R = "\\C23_real.bin"
nameC23I = "\\C23_imag.bin"
nameC33  = "\\C33.bin"

def color_bar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    #ax.tick_params(labelsize=10)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)   
    return fig.colorbar(mappable, cax=cax,orientation="horizontal") 

def plot_descriptor(descriptor,vmin,vmax,title):
    fig, (ax10) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax10.set_title(title)
    im10=ax10.imshow(descriptor, cmap = 'jet', vmin=vmin, vmax=vmax)
    ax10.axis('off')
    color_bar(im10)
    plt.tight_layout()
    return

def plot_2D_Image_array(beam,Acquisition_1,Acquisition_2,New_T1,vmin, vmax,title,save,x,y,Parcela,L):
    """
    ...
    """     
    #New_T1 =mask_element(New_T1,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    #New_T2 =mask_element(New_T2,x,y)
    #New_T3 =mask_element(New_T3,x,y)
    
    fig, (ax3) = plt.subplots(figsize=(8, 9))
    ax3.set_title(title)
    im3=ax3.imshow(New_T1, cmap = 'jet', vmin=vmin, vmax=vmax)
    lib.color_bar(im3)
    ax3.axis('off')
    plt.tight_layout()
    if save == 1:
        outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\'+L+' from '+str(Acquisition_1)+' to '+str(Acquisition_2)+'.png'
        fig.savefig(outall, bbox_inches='tight') 
    plt.close()
    return

#input_coordinates=input("Type ROI to select region of interest or coord to type image coordinates: ")
#if input_coordinates=='ROI':
#    print ("Opening RGB Image ... ")
#    x1,x2,x3,x4,y1,y2,y3,y4 = lib.get_Single_Parcel(pathA,nameC11,nameC22,nameC33)
#else:
#    x1=input("Type X1 in image coordinates: ")
#    x3=x1
#    x2=input("Type X2 in image coordinates: ")
#    x4=x2
#    y1=input("Type Y1 in image coordinates: ")
#    y2=y1
#    y3=input("Type Y3 in image coordinates: ")
#    y4=y3
#    x1=input("Type X1 in image coordinates: ")
#    y1=input("Type Y1 in image coordinates: ")
#    x2=input("Type X2 in image coordinates: ")
#    y2=input("Type Y2 in image coordinates: ")
#    x3=input("Type X3 in image coordinates: ")
#    y3=input("Type Y3 in image coordinates: ")
#    x4=input("Type X4 in image coordinates: ")
#    y4=input("Type Y4 in image coordinates: ")
#
def visRGB_from_T(img1, img2, img3,
           title,
           scale1 = [],
           scale2 = [],
           scale3 = []):
    """
    Visualise the RGB of a single acquisition
    """           
    if scale1 == []:
       scale1 = (0, np.abs(img1).mean()*2)
    if scale2 == []:
       scale2 = (0, np.abs(img2).mean()*2)
    if scale3 == []:
       scale3 = (0, np.abs(img3).mean()*2)
    size = np.shape(img1)           
    iRGB = np.zeros([size[0],size[1],3])
    iRGB[:,:,0] = np.abs(img1)/(np.abs(img1).mean()*2.5)
    iRGB[:,:,1] = np.abs(img2)/(np.abs(img2).mean()*2.5)
    iRGB[:,:,2] = np.abs(img3)/(np.abs(img3).mean()*2.5)
    iRGB[np.abs(iRGB) > 1] = 1
            
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()
    return  

def scattering_mechanism(L,alpha,beta,title):
    R=(np.sqrt(np.abs(L)))*(np.sin(alpha))*(np.cos(beta)) # R
    G=(np.sqrt(np.abs(L)))*(np.sin(alpha))*(np.sin(beta)) # G
    B=(np.sqrt(np.abs(L)))*(np.cos(alpha))                    # B
    
    RGB[:,:,0] = np.abs(R)/(np.abs(R).mean()*2)
    RGB[:,:,1] = np.abs(G)/(np.abs(G).mean()*2)
    RGB[:,:,2] = np.abs(B)/(np.abs(B).mean()*2)
    RGB[np.abs(RGB) > 1] = 1    
     
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(RGB))
    plt.axis("off")
    plt.tight_layout() 
    
def scattering_mechanism_total(L1,L2,L3,alpha1,alpha2,alpha3,beta1,beta2,beta3,title):
    R1=(np.sqrt(np.abs(L1)))*(np.sin(alpha1))*(np.cos(beta1)) # R
    G1=(np.sqrt(np.abs(L1)))*(np.sin(alpha1))*(np.sin(beta1)) # G
    B1=(np.sqrt(np.abs(L1)))*(np.cos(alpha1))                    # B

    R2=(np.sqrt(np.abs(L2)))*(np.sin(alpha2))*(np.cos(beta2)) # R
    G2=(np.sqrt(np.abs(L2)))*(np.sin(alpha2))*(np.sin(beta2)) # G
    B2=(np.sqrt(np.abs(L2)))*(np.cos(alpha2))

    R3=(np.sqrt(np.abs(L3)))*(np.sin(alpha3))*(np.cos(beta3)) # R
    G3=(np.sqrt(np.abs(L3)))*(np.sin(alpha3))*(np.sin(beta3)) # G
    B3=(np.sqrt(np.abs(L3)))*(np.cos(alpha3))                    # B                    # B
    
    R=R1+R2+R3
    G=G1+G2+G3
    B=B1+B2+B3
    
    RGB[:,:,0] = np.abs(R)/(np.abs(R).mean()*8)
    RGB[:,:,1] = np.abs(G)/(np.abs(G).mean()*8)
    RGB[:,:,2] = np.abs(B)/(np.abs(B).mean()*8)
    RGB[np.abs(RGB) > 1] = 1    
     
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(RGB))
    plt.axis("off")
    plt.tight_layout() 

############################## Create empty arrays #############################################################################################  
C = np.zeros([3,3],dtype=np.complex64)
C0 = np.zeros([3,3],dtype=np.complex64)
C_I2 = np.zeros([3,3],dtype=np.complex64)
U=np.matrix([[1, 0, 1], [1, 0, -1],[0, math.sqrt(2), 0]])
U=(1/math.sqrt(2))* np.matrix(U)     #U matrix to transform Covariance to coherency matrix
#   Uinv=np.array([[0.707107, 0.707107, 0], [0, 0, 1],[0, -0.707107, 0]])
Uinv=U.getH()                #U inverse matrix to transform Covariance to coherency matrix
T = np.zeros([3,3],dtype=np.complex64) 
T_I2 = np.zeros([3,3],dtype=np.complex64)
L1 = np.zeros([Img_size_rows,Img_size_columns])
L2 = np.zeros([Img_size_rows,Img_size_columns])
L3 = np.zeros([Img_size_rows,Img_size_columns])
L1a = np.zeros([Img_size_rows,Img_size_columns])
L1r = np.zeros([Img_size_rows,Img_size_columns])
L2a = np.zeros([Img_size_rows,Img_size_columns])
L2r = np.zeros([Img_size_rows,Img_size_columns])
L3a = np.zeros([Img_size_rows,Img_size_columns])
L3r = np.zeros([Img_size_rows,Img_size_columns])
U_11 = np.zeros([Img_size_rows,Img_size_columns])
U_21 = np.zeros([Img_size_rows,Img_size_columns])
U_31 = np.zeros([Img_size_rows,Img_size_columns])
U_12 = np.zeros([Img_size_rows,Img_size_columns])
U_22 = np.zeros([Img_size_rows,Img_size_columns])
U_32 = np.zeros([Img_size_rows,Img_size_columns])
U_13 = np.zeros([Img_size_rows,Img_size_columns])
U_23 = np.zeros([Img_size_rows,Img_size_columns])
U_33 = np.zeros([Img_size_rows,Img_size_columns])
Anisotropy = np.zeros([Img_size_rows,Img_size_columns])
Ent_Matrix = np.zeros([Img_size_rows,Img_size_columns])
entropy1 = np.zeros([Img_size_rows,Img_size_columns])
entropy2 = np.zeros([Img_size_rows,Img_size_columns])
entropy3 = np.zeros([Img_size_rows,Img_size_columns])
alpha1 = np.zeros([Img_size_rows,Img_size_columns])
beta1 = np.zeros([Img_size_rows,Img_size_columns])
alpha2 = np.zeros([Img_size_rows,Img_size_columns])
beta2 = np.zeros([Img_size_rows,Img_size_columns])
alpha3 = np.zeros([Img_size_rows,Img_size_columns])
beta3 = np.zeros([Img_size_rows,Img_size_columns])
alpha_avg = np.zeros([Img_size_rows,Img_size_columns])
beta_avg = np.zeros([Img_size_rows,Img_size_columns])   
La_avg = np.zeros([Img_size_rows,Img_size_columns])   
alpha_avg_a=np.zeros([Img_size_rows,Img_size_columns])   
beta_avg_a=np.zeros([Img_size_rows,Img_size_columns])   
Lr_avg = np.zeros([Img_size_rows,Img_size_columns])   
alpha_avg_r=np.zeros([Img_size_rows,Img_size_columns])   
beta_avg_r=np.zeros([Img_size_rows,Img_size_columns]) 
T_I2 = np.zeros([3,3],dtype=np.complex64)
T_of_change1 = np.zeros([Img_size_rows,Img_size_columns])  
T_of_change2 = np.zeros([Img_size_rows,Img_size_columns])    
T_of_change3 = np.zeros([Img_size_rows,Img_size_columns])      
TA1 = np.zeros([Img_size_rows,Img_size_columns])  
TA2 = np.zeros([Img_size_rows,Img_size_columns])    
TA3 = np.zeros([Img_size_rows,Img_size_columns])   
TB1 = np.zeros([Img_size_rows,Img_size_columns])  
TB2 = np.zeros([Img_size_rows,Img_size_columns])    
TB3 = np.zeros([Img_size_rows,Img_size_columns])    
T_of_changeA2= np.zeros([Img_size_rows,Img_size_columns])  
T_of_changeA3= np.zeros([Img_size_rows,Img_size_columns])  
T_of_changeA1= np.zeros([Img_size_rows,Img_size_columns]) 
T_of_changeB2= np.zeros([Img_size_rows,Img_size_columns])  
T_of_changeB3= np.zeros([Img_size_rows,Img_size_columns])  
T_of_changeB1= np.zeros([Img_size_rows,Img_size_columns])
RGB = np.zeros([Img_size_rows,Img_size_columns,3])
 
 ################################ Open image 1 ########################## 
   # Open the covariance matrix elements
print ("Opening Image A date ...")
C11 = lib.Open_C_diag_element(path1 + nameC11)
C12 = lib.Open_C_element(path1 + nameC12R, path1 + nameC12I)
C13 = lib.Open_C_element(path1 + nameC13R, path1 + nameC13I)
C22 = lib.Open_C_diag_element(path1 + nameC22)
C23 = lib.Open_C_element(path1 + nameC23R, path1 + nameC23I)
C33 = lib.Open_C_diag_element(path1 + nameC33)
    
C11 = C11[y_min:y_max, x_min:x_max]  
C12 = C12[y_min:y_max, x_min:x_max] 
C13 = C13[y_min:y_max, x_min:x_max]     
C22 = C22[y_min:y_max, x_min:x_max] 
C23 = C23[y_min:y_max, x_min:x_max]  
C33 = C33[y_min:y_max, x_min:x_max] 

################################ Open image 2 ##########################    
print ("Opening Image B date...")
C11_I2 = lib.Open_C_diag_element(path2 + nameC11)
C12_I2 = lib.Open_C_element(path2 + nameC12R, path1 + nameC12I)
C13_I2 = lib.Open_C_element(path2 + nameC13R, path1 + nameC13I)
C22_I2 = lib.Open_C_diag_element(path2 + nameC22)
C23_I2 = lib.Open_C_element(path2 + nameC23R, path1 + nameC23I)
C33_I2 = lib.Open_C_diag_element(path2 + nameC33)

# Crop the image to only leave the ROI
C11_I2 = C11_I2[y_min:y_max, x_min:x_max] 
C12_I2 = C12_I2[y_min:y_max, x_min:x_max] 
C13_I2 = C13_I2[y_min:y_max, x_min:x_max] 
C22_I2 = C22_I2[y_min:y_max, x_min:x_max] 
C23_I2 = C23_I2[y_min:y_max, x_min:x_max] 
C33_I2 = C33_I2[y_min:y_max, x_min:x_max]

for i in range(Img_size_rows-1):
    print (np.round((i*100/Img_size_rows),1),"%")      
    for j in range(Img_size_columns-1):

        C[0][0]=C11[i][j]
        C[0][1]=C12[i][j]
        C[0][2]=C13[i][j]
        C[1][0]=np.conj(C12[i][j])
        C[1][1]=C22[i][j]
        C[1][2]=C23[i][j]
        C[2][0]=np.conj(C13[i][j])
        C[2][1]=np.conj(C23[i][j])
        C[2][2]=C33[i][j]
        #print "cycle y=",i,j
        T=U.dot(C).dot(Uinv)
        
        TA1[i][j]=T[0,0]
        TA2[i][j]=T[1,1]
        TA3[i][j]=T[2,2]
        
        C_I2[0][0]=C11_I2[i][j]
        C_I2[0][1]=C12_I2[i][j]
        C_I2[0][2]=C13_I2[i][j]
        C_I2[1][0]=np.conj(C12_I2[i][j])
        C_I2[1][1]=C22_I2[i][j]
        C_I2[1][2]=C23_I2[i][j]
        C_I2[2][0]=np.conj(C13_I2[i][j])
        C_I2[2][1]=np.conj(C23_I2[i][j])
        C_I2[2][2]=C33_I2[i][j]
        
        T_I2=U.dot(C_I2).dot(Uinv)
        
        TB1[i][j]=T_I2[0,0]
        TB2[i][j]=T_I2[1,1]
        TB3[i][j]=T_I2[2,2]
              
        # Optimisation for Normalised Difference
        Tc = (T_I2 - T)/(np.trace(T)+np.trace(T_I2))
        #Tc = np.nan_to_num(Tc)
        T_of_changeA1[i][j]=Tc[0,0]
        T_of_changeA2[i][j]=Tc[1,1]
        T_of_changeA3[i][j]=Tc[2,2]
        
        Tc12 = (T - T_I2)
        Tc12 = np.nan_to_num(Tc12)
        T_of_changeB1[i][j]=Tc12[0,0]
        T_of_changeB2[i][j]=Tc12[1,1]
        T_of_changeB3[i][j]=Tc12[2,2]
        #invT11 = ln.inv(T)
        #A = np.dot(T_I2, invT11)
#            eigenvalues, eigenvectors = np.linalg.eigh(A)
        #A = np.nan_to_num(A)
        eigenvalues, eigenvectors = np.linalg.eig(Tc)
#            [d1, v1] = ln.eigh(T)
#            [d2, v2] = ln.eig(A)
        
        ind = np.argsort(eigenvalues)# organize eigenvectors from higher to lower value
                    
        L1[i][j] = eigenvalues[ind[2]]
        L2[i][j] = eigenvalues[ind[1]]
        L3[i][j] = eigenvalues[ind[0]]
            
        U_11[i][j]=np.abs(eigenvectors[0,2]) 
        U_21[i][j]=np.abs(eigenvectors[1,2])
        U_31[i][j]=np.abs(eigenvectors[2,2])
 
        U_12[i][j]=np.abs(eigenvectors[0,1])
        U_22[i][j]=np.abs(eigenvectors[1,1])
        U_32[i][j]=np.abs(eigenvectors[2,1])
        
        U_13[i][j]=np.abs(eigenvectors[0,0])
        U_23[i][j]=np.abs(eigenvectors[1,0])
        U_33[i][j]=np.abs(eigenvectors[2,0]) 
        
        # Only Positive eigenvalues from L1
#        if L1>1:
#            L1a[i][j]=L1
#        else: 
#            L1a[i][j]=0    
#     
## Only Negative eigenvalues from L1
#        if L1<1:
#            L1r[i][j]=L1
#        else: 
#            L1r[i][j]=0 
## Only Positive eigenvalues from L2
#        if L2>1:
#            L2a[i][j]=L2
#        else: 
#            L2a[i][j]=0    
## Only Negative eigenvalues from L2
#        if L2<1:
#            L2r[i][j]=L2
#        else: 
#            L2r[i][j]=0 
# # Only Positive eigenvalues from L3
#        if L3>1:
#            L3a[i][j]=L3
#        else: 
#            L3a[i][j]=0    
## Only Negative eigenvalues from L3
#        if L3<1:
#            L3r[i][j]=L3
#        else: 
#            L3r[i][j]=0 

visRGB_from_T(TA2, TA3, TA1, "RGB from [T1]")
visRGB_from_T(TB2, TB3, TB1, "RGB from [T2]")

T_of_change2[T_of_changeA2 < 0] = 0 
T_of_change3[T_of_changeA3 < 0] = 0 
T_of_change1[T_of_changeA1 < 0] = 0  
visRGB_from_T(T_of_changeA2, T_of_changeA3, T_of_changeA1, "RGB Added - from [Tc 2-1]")

T_of_changeB2[T_of_changeB2 < 0] = 0 
T_of_changeB3[T_of_changeB3 < 0] = 0 
T_of_changeB1[T_of_changeB1 < 0] = 0  
visRGB_from_T(T_of_changeB2, T_of_changeB3, T_of_changeB1, "RGB Removed - from [Tc 1-2]")

alpha1=np.arccos(np.abs(U_11))
alpha2=np.arccos(np.abs(U_12))
alpha3=np.arccos(np.abs(U_13))

beta1=np.arccos(np.abs(U_21)/np.sin(alpha1))
beta2=np.arccos(np.abs(U_22)/np.sin(alpha2))
beta3=np.arccos(np.abs(U_23)/np.sin(alpha3))

vmin=-1
vmax=1
title="Lamda 1"
plot_descriptor(L1,vmin,vmax,title)

title="Lamda 2" 
plot_descriptor(L2,vmin,vmax,title)

title="Lamda 3"
plot_descriptor(L3,vmin,vmax,title) 

vmin=0
vmax=np.pi/2
title="Alpha 1 - rad"
plot_descriptor(alpha1,vmin,vmax,title)

title="Alpha 2 - rad - Rad" 
plot_descriptor(alpha2,vmin,vmax,title)

title="Alpha 3 - rad"
plot_descriptor(alpha3,vmin,vmax,title) 

#vmin=-np.pi
#vmax=np.pi
title="Beta 1 - rad"
plot_descriptor(beta1,vmin,vmax,title)

title="Beta 2 - rad - Rad" 
plot_descriptor(beta2,vmin,vmax,title)

title="Beta 3 - rad"
plot_descriptor(beta3,vmin,vmax,title) 

L1a = L1*(L1 > 0)# Separating positive and negative eigenvalues
L1r = L1*(L1 < 0)

title="Removed First scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(L1r,alpha1,beta1,title)

title="Added First scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(L1a,alpha1,beta1,title)

L2a = L2*(L2 > 0)# Separating positive and negative eigenvalues
L2r = L3*(L2 < 0)

title="Removed Second scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(L2r,alpha2,beta2,title)

title="Added Second scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(L2a,alpha2,beta2,title)

L3a = L3*(L3 > 0)# Separating positive and negative eigenvalues
L3r = L3*(L3 < 0)

title="Removed Third scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(L3r,alpha3,beta3,title)

title="Added Third scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(L3a,alpha3,beta3,title)

title="Total Added scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism_total(L1a,L2a,L3a,alpha1,alpha2,alpha3,beta1,beta2,beta3,title)

title="Total removed scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism_total(L1r,L2r,L3r,alpha1,alpha2,alpha3,beta1,beta2,beta3,title)

P1a=(L1a/(L1a+L2a+L3a))
P1a = np.nan_to_num(P1a)
P2a=(L2a/(L1a+L2a+L3a))
P2a = np.nan_to_num(P2a)
P3a=(L3a/(L1a+L2a+L3a))
P3a = np.nan_to_num(P3a)

La_avg=(P1a*L1a)+(P2a*L2a)+(P3a*L3a)
alpha_avg_a=(P1a*alpha1)+(P2a*alpha2)+(P3a*alpha3)
beta_avg_a=(P1a*beta1)+(P2a*beta2)+(P3a*beta3)

title="Total Added scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(La_avg,alpha_avg_a,beta_avg_a,title)
#            

P1r=(L1r/(L1r+L2r+L3r))
P1r = np.nan_to_num(P1r)
P2r=(L2r/(L1r+L2r+L3r))
P2r = np.nan_to_num(P2r)
P3r=(L3r/(L1r+L2r+L3r))
P3r = np.nan_to_num(P3r)

Lr_avg=(P1r*L1r)+(P2r*L2r)+(P3r*L3r)
alpha_avg_r=(P1r*alpha1)+(P2r*alpha2)+(P3r*alpha3)
beta_avg_r=(P1r*beta1)+(P2r*beta2)+(P3r*beta3)

title="Total removed scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(Lr_avg,alpha_avg_r,beta_avg_r,title)
#Title='Parcel '+str(Parcela)+': Added SM from '+str(Acquisition_2)+' to '+str(Acquisition_1)
#y=np.array([y1,y2,y3,y4]).astype(int) 
#x=np.array([x1,x2,x3,x4]).astype(int)
SM='Added'
beam=""
Acquisition_1=""
Acquisition_2=""
Parcela=""
title="Total Added scattering mechanism provided by the Alberto's way"
iRGB=lib.visRGB_L_Contrast(U_11, U_21, U_31,U_12, U_22, U_32, U_13, U_23, U_33,title,L1a,L2a,L3a,x,y,beam,Acquisition_1,Acquisition_2,Parcela,SM)
##iRGBi=lib.visRGB_L_Contrast_log10(U_11, U_21, U_31,U_12, U_22, U_32, U_13, U_23, U_33,Title,L1a,L2a,L3a,x,y,beam,Acquisition_1,Acquisition_2,Parcela,SM)
#Title='Parcel '+str(Parcela)+': Removed SM from '+str(Acquisition_2)+' to '+str(Acquisition_1)
SM='Removed'
title="Total removed scattering mechanism provided by the Alberto's way"
iRGBr=lib.visRGB_L_Contrast(U_11, U_21, U_31,U_12, U_22, U_32, U_13, U_23, U_33,title,L1r,L2r,L3r,x,y,beam,Acquisition_1,Acquisition_2,Parcela,SM)

entropy1=((-P1a)*(np.log(P1a)/np.log(3)))
entropy1 = np.nan_to_num(entropy1)
entropy2=((-P2a)*(np.log(P2a)/np.log(3)))
entropy2 = np.nan_to_num(entropy2)
entropy3=((-P3a)*(np.log(P3a)/np.log(3)))
entropy3 = np.nan_to_num(entropy3)
Entropy=entropy1+entropy2+entropy3

vmin=0
vmax=1
title="Entropy of the added change"
plot_descriptor(Entropy,vmin,vmax,title)

entropy1=((-P1r)*(np.log(P1r)/np.log(3))) # using log change basis to do log in base 3
entropy1 = np.nan_to_num(entropy1)
entropy2=((-P2r)*(np.log(P2r)/np.log(3)))
entropy2 = np.nan_to_num(entropy2)
entropy3=((-P3r)*(np.log(P3r)/np.log(3)))
entropy3 = np.nan_to_num(entropy3)
Entropy=entropy1+entropy2+entropy3

vmin=0
vmax=1
title="Entropy of the removed change"
plot_descriptor(Entropy,vmin,vmax,title)

Anisotropy_a=((L2a-L3a)/(L2a+L3a))
Anisotropy_a = np.nan_to_num(Anisotropy_a)
vmin=-1
vmax=1
title="Anisotropy of the added change"
plot_descriptor(Anisotropy_a,vmin,vmax,title)

Anisotropy_r=((L2r-L3r)/(L2r+L3r))
Anisotropy_r = np.nan_to_num(Anisotropy_r)
vmin=-1
vmax=1
title="Anisotropy of the removed change"
plot_descriptor(Anisotropy_r,vmin,vmax,title)