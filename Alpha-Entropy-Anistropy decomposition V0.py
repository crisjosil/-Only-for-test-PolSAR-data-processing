# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 11:18:04 2018

@author: Cristian Silva
"""
import numpy as np
import matplotlib.pyplot as plt 
import scipy.linalg as ln
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import signal  
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import sys
sys.path.insert(0, 'D:\\Juanma\\')
#import Library_for_Donana as lib 
#from scipy import signal

# Image size and characteristics
#dx=7080   # rice
#dy=8000
#dx=1168   # San francisco L band
#dy=2531
dx=900   # San francisco test sample
dy=1024
datatype = 'float32'
header =  np.array([dx, dy])
#folder="D:\\Juanma\\FQ19W\\2014-06-05.rds2\\C3\\"
#folder="D:\\DATASETS\\cm6406_3_Frequency_Polarimetry_San_Fran\\cm6406\\C3\\"
folder="D:\\DATASETS\\AIRSAR_SanFrancisco\\C3\\"
# Function to open an image (backscatter components e.g. HHHH, HVHV, VVVV)
def open_diag(path):
    f = open(path, 'rb')
    img = np.fromfile(f, dtype=datatype, sep="")
    img = img.reshape(header).astype('float32')
    return (img)
# Function to open an image (backscatter components e.g. HHHV, HHVV, HVVV...)
def open_off_diag(path):
    f = open(path, 'rb')
    img = np.fromfile(f, dtype='complex64', sep="")
    img = img.reshape(header).astype('complex64')
    return (img)

# Function to open an image (T11,T22,T33)
def Open_C_diag_element(filename):
    f = open(filename, 'rb')
    img = np.fromfile(f, dtype=datatype, sep="")
    img = img.reshape(header).astype('float32')
    return(img)
# Function to open a COMPLEX image (T12,T23,T23)
def Open_C_element(filename1,filename2):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """
    fR = open(filename1, 'rb')
    imgR = np.fromfile(fR, dtype=datatype, sep="")
    imgR = imgR.reshape(header).astype('float32')

    fI = open(filename2, 'rb') # image data
    imgI = np.fromfile(fI, dtype=datatype, sep="")
    imgIr = imgI.reshape(header).astype('float32')
    imgI = imgIr*1j
    img12=imgR+imgI
    return(img12)
# Function to create an RGB
def visRGB(img1, img2, img3, title):
    size = np.shape(img2)           
    iRGB = np.zeros([size[0],size[1],3])
    R=(img1-img3)
    G=img2
    B=(img1+img3)
    iRGB[:,:,0] = np.abs(R)/(np.abs(R).mean()*2.5)
    iRGB[:,:,1] = np.abs(G)/(np.abs(G).mean()*2.5)
    iRGB[:,:,2] = np.abs(B)/(np.abs(B).mean()*2.5)
    iRGB[np.abs(iRGB) > 1] = 1
#    
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()
    return 
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
    
def plot_backscatter(img,title):
    img=10*np.log10(img)
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(img), cmap = 'gray', vmin=0, vmax=np.abs(img).mean()*2)
    plt.axis("off")
    plt.tight_layout()
    #fig.colorbar(im1,ax=ax1)
    
def obtain_alphas(U_11,U_12,U_13):
    alpha1=np.arccos(np.abs(U_11))
    alpha1=np.degrees(alpha1)
    alpha3=np.arccos(np.abs(U_13))
    alpha3=np.degrees(alpha3)    
    return(alpha1,alpha3)
        
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

def Tmatrix(hh, hv, vv, win):
    """
    This routine generates the elements of the Coherency matrix, given the 
    images in lexicographic basis and the size of the moving window
    """
    # Elements of the Pauli basis
    p1 = hh+vv
    p2 = hh-vv
    p3 = 2*hv
    
#    kernel for averaging
    kernel  = np.ones((win,win),np.float32)/(np.power(win,2))
    
    Taa = signal.convolve2d(np.power(np.abs(p1),2), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tbb = signal.convolve2d(np.power(np.abs(p2),2), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tcc = signal.convolve2d(np.power(np.abs(p3),2), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tab = signal.convolve2d(p1*np.conj(p2), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tac = signal.convolve2d(p1*np.conj(p3), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
    
    Tbc = signal.convolve2d(p2*np.conj(p3), kernel, 
                            mode='full', boundary='fill', fillvalue=0)
                            
    return(Taa, Tbb, Tcc, Tab, Tac, Tbc)
    
def Lee_Filter(img,win):
    """
    This module aplies Lee Filter with a win x win window 
    """    
    img_mean = uniform_filter(img, (win, win))
    img_sqr_mean = uniform_filter(img**2, (win, win))
    img_variance = img_sqr_mean - img_mean**2

    overall_variance =variance(img)

    img_weights = img_variance**2 / (img_variance**2 + overall_variance**2)
    img_output = img_mean + img_weights * (img - img_mean)

    return(img_output)

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
    return
############################################################## Input data from backscatter images ##############################################
# Opening images EMISAR_Foulum-C3
#folder="D:\\DATASETS\EMISAR_Foulum-C3\\"    
#fhhhh=open_diag    (folder+"fl063_m0719_foulumW_l_lhhhh.co")
#fhhhv=open_off_diag(folder+"fl063_m0719_foulumW_l_lhhhv.co")
#fhhvv=open_off_diag(folder+"fl063_m0719_foulumW_l_lhhvv.co")
#fhvhv=open_diag    (folder+"fl063_m0719_foulumW_l_lhvhv.co")
#fhvvv=open_off_diag(folder+"fl063_m0719_foulumW_l_lhvvv.co")
#fvvvv=open_diag    (folder+"fl063_m0719_foulumW_l_lvvvv.co")
#
#fhhhh=fhhhh[:,4:1760]
#fhhhv=fhhhv[:,4:1760]
#fhhvv=fhhvv[:,4:1760]
#fhvhv=fhvhv[:,4:1760]
#fhvvv=fhvvv[:,4:1760]
#fvvvv=fvvvv[:,4:1760]    
    
#plot_backscatter(fhhhh,'HH (log scale)')
#plot_backscatter(fvvvv,'VV (log scale)')
#plot_backscatter(fhvhv,'HV (log scale)')
#visRGB(fhhhh,fhvhv,fvvvv,"RGB EMISAR FOULUM_L_C3")
#Taa1, Tbb1, Tcc1, Tab1, Tac1, Tbc1=Tmatrix(fhhhh, fhvhv, fvvvv, win)
#visRGB_from_T(Tbb1, Tcc1, Taa1, "RGB")
############################################################### Input data from Coherency matrix components #############################################    
# Opening images of E-SAR Oberpfaffenhofen, DE 
#folder="D:\\Datasets\\AIRSAR_SanFrancisco\\T3\\T\\T3\\"    
#folder="D:\DATASETS\JAXA\Peru\ALOS-P1_1__A-ORBIT__ALPSRP149237020.data"
#T11=Open_C_diag_element(folder+"T11.bin")
#T22=Open_C_diag_element(folder+"T22.bin")
#T33=Open_C_diag_element(folder+"T33.bin")
#T12=Open_C_element     (folder+"T12_real.bin",folder+"T12_imag.bin")
#T13=Open_C_element     (folder+"T13_real.bin",folder+"T13_imag.bin")
#T23=Open_C_element     (folder+"T23_real.bin",folder+"T23_imag.bin")
#win=3
#T11=Lee_Filter(T11,win)
#T22=Lee_Filter(T22,win)
#T33=Lee_Filter(T33,win)
#T12=Lee_Filter(T11,win)
#T13=Lee_Filter(T22,win)
#T23=Lee_Filter(T33,win)
#Taa1, Tbb1, Tcc1, Tab1, Tac1, Tbc1=T11,T22,T33,T12,T13,T23
#visRGB_from_T(Tbb1, Tcc1, Taa1, "RGB")
###########################################################################################################################################################
############################################################### Input data from covariance matrix components #############################################    
# Opening images of E-SAR Oberpfaffenhofen, DE 
#folder="D:\\Datasets\\AIRSAR_SanFrancisco\\T3\\T\\T3\\"    
#folder="C:\\Users\\crisj\\Box\\Python\\Datasets\\AIRSAR_Flevoland\\C3\\"
##folder="D:\DATASETS\JAXA\Peru\ALOS-P1_1__A-ORBIT__ALPSRP149237020.data"
C11=Open_C_diag_element(folder+"C11.bin")
C22=Open_C_diag_element(folder+"C22.bin")
C33=Open_C_diag_element(folder+"C33.bin")
#C11[C11 < 0] = 0
#C22[C22 < 0] = 0
#C33[C33 < 0] = 0
C12=Open_C_element     (folder+"C12_real.bin",folder+"C12_imag.bin")
C13=Open_C_element     (folder+"C13_real.bin",folder+"C13_imag.bin")
C23=Open_C_element     (folder+"C23_real.bin",folder+"C23_imag.bin")
#C11=C11[2900:5400,4300:6000] # For seville dataset
#C22=C22[2900:5400,4300:6000]
#C33=C33[2900:5400,4300:6000]  
#C11=C11[500:1500,500:1500] # For san fran
#C22=C22[500:1500,500:1500]
#C33=C33[500:1500,500:1500]
#C12=C12[500:1500,500:1500] # For san fran
#C13=C13[500:1500,500:1500]
#C23=C23[500:1500,500:1500]
win=1
#C11=Lee_Filter(C11,win)
#C22=Lee_Filter(C22,win)
#C33=Lee_Filter(C33,win)
#C12=Lee_Filter(C11,win)
#C13=Lee_Filter(C22,win)
#C23=Lee_Filter(C33,win)
visRGB(C11, C22, C33, "RGB from [C]")
Taa1, Tbb1, Tcc1, Tab1, Tac1, Tbc1=C11,C22,C33,C12,C13,C23
#Taa1, Tbb1, Tcc1, Tab1, Tac1, Tbc1=Tmatrix(C11, C22, C33, win)
#visRGB_from_T(Tbb1, Tcc1, Taa1, "RGB from [T]")
###########################################################################################################################################################
    # Initialisations
dr=Taa1.shape[0]    
da=Taa1.shape[1]    
T = np.matrix(np.zeros([3,3],dtype=np.complex64)) 
C = np.matrix(np.zeros([3,3],dtype=np.complex64)) 
#T22 = np.matrix(np.zeros([3,3],dtype=np.complex64))
lam1 = np.zeros([dr,da]) 
lam2 = np.zeros([dr,da])
lam3 = np.zeros([dr,da])
L1 = np.zeros([dr,da]) 
L2 = np.zeros([dr,da])
L3 = np.zeros([dr,da])
P1 = np.zeros([dr,da]) 
P2 = np.zeros([dr,da])
P3 = np.zeros([dr,da])
alpha_avg= np.zeros([dr,da])
beta_avg= np.zeros([dr,da])
L_avg= np.zeros([dr,da])
alpha1 = np.zeros([dr,da]) 
alpha2 = np.zeros([dr,da])
alpha3 = np.zeros([dr,da])
beta1 = np.zeros([dr,da]) 
beta2 = np.zeros([dr,da])
beta3 = np.zeros([dr,da])
U_11 = np.zeros([dr,da])
U_21 = np.zeros([dr,da])
U_31 = np.zeros([dr,da])
U_12 = np.zeros([dr,da])
U_22 = np.zeros([dr,da])
U_32 = np.zeros([dr,da])
U_13 = np.zeros([dr,da])
U_23 = np.zeros([dr,da])
U_33 = np.zeros([dr,da])
T11 = np.zeros([dr,da])
T22 = np.zeros([dr,da])
T33 = np.zeros([dr,da])


RGB = np.zeros([dr,da,3])
Entropy = np.zeros([dr,da])
entropy1 = np.zeros([dr,da])
entropy2 = np.zeros([dr,da])
entropy3 = np.zeros([dr,da])
Anisotropy= np.zeros([dr,da])
win=3
U=np.matrix([[1, 0, 1], [1, 0, -1],[0, math.sqrt(2), 0]])
U=(1/math.sqrt(2))* np.matrix(U)     #U matrix to transform Covariance to coherency matrix
#   Uinv=np.array([[0.707107, 0.707107, 0], [0, 0, 1],[0, -0.707107, 0]])
Uinv=U.getH() 
epsilon=1e-6
for m in range(dr-1):
    print (np.round((m*100/dr),1),"%")
    for n in range(da-1):
        
        C[0,0] = Taa1[m,n]
        C[0,1] = Tab1[m,n] 
        C[0,2] = Tac1[m,n]
        C[1,0] = np.conj(Tab1[m,n])
        C[1,1] = Tbb1[m,n]
        C[1,2] = Tbc1[m,n]
        C[2,0] = np.conj(Tac1[m,n])
        C[2,1] = np.conj(Tbc1[m,n])
        C[2,2] = Tcc1[m,n]
        T=U.dot(C).dot(Uinv)   
        #
        T11[m][n]=T[0,0]
        T22[m][n]=T[1,1]
        T33[m][n]=T[2,2]
        #T = np.nan_to_num(T)
        eigenvalues, eigenvectors = np.linalg.eigh(T)
        
        ind = np.argsort(eigenvalues)# organize eigenvalues from higher to lower value
        eigenvalues[eigenvalues < 0] = 0    
        
        L1[m][n] = eigenvalues[ind[2]]
        L2[m][n] = eigenvalues[ind[1]]
        L3[m][n] = eigenvalues[ind[0]]
        
        U_11[m][n]=np.abs(eigenvectors[0,2]) 
        U_21[m][n]=np.abs(eigenvectors[1,2])
        U_31[m][n]=np.abs(eigenvectors[2,2])
 
        U_12[m][n]=np.abs(eigenvectors[0,1])
        U_22[m][n]=np.abs(eigenvectors[1,1])
        U_32[m][n]=np.abs(eigenvectors[2,1])
        
        U_13[m][n]=np.abs(eigenvectors[0,0])
        U_23[m][n]=np.abs(eigenvectors[1,0])
        U_33[m][n]=np.abs(eigenvectors[2,0])
        
P1=(L1/(L1+L2+L3))
entropy1=((-P1)*(np.log(P1)/np.log(3)))
P2=(L2/(L1+L2+L3))
entropy2=((-P2)*(np.log(P2)/np.log(3)))
P3=(L3/(L1+L2+L3))
entropy3=((-P3)*(np.log(P3)/np.log(3)))
        
Entropy=entropy1+entropy2+entropy3
Anisotropy=((L2-L3)/(L2+L3))

alpha1=np.arccos(np.abs(U_11))
alpha2=np.arccos(np.abs(U_12))
alpha3=np.arccos(np.abs(U_13))

beta1=np.arccos(np.abs(U_21)/np.sin(alpha1))
beta2=np.arccos(np.abs(U_22)/np.sin(alpha2))
beta3=np.arccos(np.abs(U_23)/np.sin(alpha3))

L_avg=(P1*L1)+(P2*L2)+(P3*L3)
alpha_avg=(P1*alpha1)+(P2*alpha2)+(P3*alpha3)
beta_avg=(P1*beta1)+(P2*beta2)+(P3*beta3)
L_avg = np.nan_to_num(L_avg)
alpha_avg = np.nan_to_num(alpha_avg)
beta_avg = np.nan_to_num(beta_avg)

visRGB_from_T(T22, T33, T11, "RGB from [T]")

vmin=0
vmax=np.pi/2
title="Dominant Alpha angle - rad"
plot_descriptor(alpha1,vmin,vmax,title)

title="Average Alpha angle - Rad" 
plot_descriptor(alpha_avg,vmin,vmax,title)

title="Dominant Beta angle"
plot_descriptor(beta1,vmin,vmax,title) 

title="Average Beta angle - Rad"
plot_descriptor(beta_avg,vmin,vmax,title)

vmax=1
title="Entropy"
plot_descriptor(Entropy,vmin,vmax,title)

title="Anisotropy"
plot_descriptor(Anisotropy,vmin,vmax,title)

title="Lamda 1"
plot_descriptor(L1,vmin,vmax,title)

title="Main scattering mechanism provided by the eigenvector-eigenvalue based decomposition"
scattering_mechanism(L_avg,alpha_avg,beta_avg,title)





