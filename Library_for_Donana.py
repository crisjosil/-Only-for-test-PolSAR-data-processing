# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 05:57:35 2017
Library to open bin images
@author: crisj
"""
import numpy as np
import matplotlib.pyplot as plt
import math as math
from mpl_toolkits.axes_grid1 import make_axes_locatable
#from scipy import signal
#import spectral.io.envi as envi
from mpl_toolkits.mplot3d import Axes3D
import cv2
from skimage import color
#from scipy.ndimage.filters import uniform_filter
#from scipy.ndimage.measurements import variance
import pandas as pd
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
import pydotplus
import graphviz
import math as math
from scipy import stats
from scipy.optimize import curve_fit
import scipy
import scipy.linalg as ln

Site=input("Type Seville or Agrisar ")

if Site == "Seville":
    header =  np.array([7080, 8000]) # Seville rice
elif Site == "Agrisar":
    header =  np.array([1900, 3380])  # Agrisar 2009

datatype = 'float32'


def get_Single_Parcel(path1,nameC11,nameC22,nameC33):
    """
    Shows an RGB image in PAULI basis to select a ROI
    Then shows the ROI to select the specific coordinates of the parcel
"""     
    C11 = Open_C_diag_element(path1 + nameC11)
    C22 = Open_C_diag_element(path1 + nameC22)
    C33 = Open_C_diag_element(path1 + nameC33)
    
   
    iRGB= visRGB(C11,C22,C33,'RGB - Full Image')
        #Get ROI coordinates and plot it
    print('please click four points')
    Coordinates = plt.ginput(4)
    x1,y1=Coordinates[0]
    x2,y2=Coordinates[1]
    x3,y3=Coordinates[2]
    x4,y4=Coordinates[3]
    
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    x3=int(x3)
    x4=int(x4)
    y3=int(y3)
    y4=int(y4)
    #
    #print(x1)
    #print(x2)
    #print(y1)
    #print(y2)
    
    y=np.array([y1,y2,y3,y4])
    x=np.array([x1,x2,x3,x4])
    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
    
    plt.close()           
    iRGB_ROI = iRGB[y_min:y_max,x_min:x_max]
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
    ax1.set_title('Region of interest')
    ax1.imshow(iRGB_ROI)  
    
    #Get parcel coordinates and plot it
    print('please click four points')
    Coordinates = plt.ginput(4)
    x1,y1=Coordinates[0]
    x2,y2=Coordinates[1]
    x3,y3=Coordinates[2]
    x4,y4=Coordinates[3]
    
    x1=int(x1)
    x2=int(x2)
    y1=int(y1)
    y2=int(y2)
    x3=int(x3)
    x4=int(x4)
    y3=int(y3)
    y4=int(y4)
    
    y=np.array([y1+y_min,y2+y_min,y3+y_min,y4+y_min])
    x=np.array([x1+x_min,x2+x_min,x3+x_min,x4+x_min])
    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
           
    # apply a mask to delete the rest of the image and extract the parcel 
    
    mask = np.full_like(iRGB, 0) #create the mask = array of zeros with the original image size
    roi_corners = np.array([[(x[0],y[0]), (x[1],y[1]), (x[2],y[2]),(x[3],y[3])]]) #roi_corners = Array with the Parcel coordinates
    channel_count = iRGB.shape[2]  # Number of channels (3)
    ignore_mask_color = (1.9999,)*channel_count # How many channels has the image for filling them
    cv2.fillPoly(mask, roi_corners, ignore_mask_color) # Fill of ones the polygon with the parcel coordinates (put a mask)
    # use cv2.fillConvexPoly if you know it's convex
    
    # apply the mask
    plt.close()
    masked_iRGB = cv2.bitwise_and(iRGB, mask)# Merge the original image with the mask (Only leave the pixels inside the polygon)
    masked_iRGB = masked_iRGB[y_min:y_max,x_min:x_max]
    fig, (ax10) = plt.subplots(1, sharex=True, sharey=True)
    ax10.set_title('Masked Image')
    ax10.imshow(masked_iRGB)
    
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    y1=y[0]
    y2=y[1]
    y3=y[2]
    y4=y[3]
    
    return(x1,x2,x3,x4,y1,y2,y3,y4)
    
def mask_C_element(img,x,y):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """   
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(x[0],y[0]), (x[1],y[1]), (x[2],y[2]),(x[3],y[3])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = 2  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (1.9999,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    pixels_in_parcel=np.count_nonzero(mask == 1)
    masked_C_element = np.zeros(img.shape, dtype=np.complex64)
    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
    
    for i in range(y_max-y_min):
            for j in range(x_max-x_min):
            
                if mask[i][j] == 1:
                    masked_C_element[i][j]=img[i][j]
                else:
                    masked_C_element[i][j]=0
                
    return(masked_C_element,pixels_in_parcel)

#def get_ROI(iRGB):
#    """
#    It opens SLC images as formatted for ESAR
#    The header can be read from the ENVI .hdr
#    """
#    print('please click four points')
#    Coordinates = plt.ginput(4)
#    x1,y1=Coordinates[0]
#    x2,y2=Coordinates[1]
#    x3,y3=Coordinates[2]
#    x4,y4=Coordinates[3]
#    
#    x1=int(x1)
#    x2=int(x2)
#    y1=int(y1)
#    y2=int(y2)
#    x3=int(x3)
#    x4=int(x4)
#    y3=int(y3)
#    y4=int(y4)
#    #
#    #print(x1)
#    #print(x2)
#    #print(y1)
#    #print(y2)
#    
#    y=np.array([y1,y2,y3,y4])
#    x=np.array([x1,x2,x3,x4])
#    y_min=int(min(y))
#    y_max=int(max(y))
#    x_min=int(min(x))
#    x_max=int(max(x))
#    
#    iRGB_ROI = iRGB[y_min:y_max,x_min:x_max]
#    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
#    ax1.set_title('Region of interest')
#    #ax1.imshow(np.abs(iRGB_ROI))
#    ax1.imshow(iRGB_ROI, cmap = 'gray', vmin=0, vmax=iRGB_ROI.mean()*2.5)
#   
#    return(iRGB_ROI)

def Open_C_diag_element(filename):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """
    f = open(filename, 'rb')
    img = np.fromfile(f, dtype=datatype, sep="")
    img = img.reshape(header).astype('float32')
   
    return(img)
    
 
def Open_C_element(filename1,filename2):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """
    fR = open(filename1, 'rb')
    imgR = np.fromfile(fR, dtype=datatype, sep="")
    imgR = imgR.reshape(header).astype('complex64')

    fI = open(filename2, 'rb') # image data
    imgI = np.fromfile(fI, dtype=datatype, sep="")
    imgI = imgI.reshape(header).astype('complex64')
    imgI = imgI*1j
    img12=imgR+imgI
    #img12=np.abs(img12)
    
    return(img12)
    
#def visRGB(img1, img2, img3, title,beam,Acquisition_1):
def visRGB(img1, img2, img3, title):
    size = np.shape(img2)           
    iRGB = np.zeros([size[0],size[1],3])
    R=0.5*(img1-img3)
    G=2*img2
    B=0.5*(img1+img3)
    iRGB[:,:,0] = np.abs(R)/(np.abs(R).mean()*4.5)
    iRGB[:,:,1] = np.abs(G)/(np.abs(G).mean()*4.5)
    iRGB[:,:,2] = np.abs(B)/(np.abs(B).mean()*4.5)
    iRGB[np.abs(iRGB) > 1] = 1
#    
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()    
    #outall= 'D:\\Juanma\\'+str(beam)+'\\RGBs\\Pauli RGB'+'_'+str(Acquisition_1)+'.png'
    #fig.savefig(outall, bbox_inches='tight', dpi = 1200)                 
    return(iRGB)    

def visRGB_test(img1, img2, img3, title):
    """
    Visualise the RGB of a single acquisition
    """           
    size = np.shape(img2)           
    iRGB = np.zeros([size[0],size[1],3])
    
    R=0.5*(img1-img3)
    G=2*img2
    B=0.5*(img1+img3)
    
    iRGB[:,:,0] = np.abs(R)
    iRGB[:,:,1] = np.abs(G)
    iRGB[:,:,2] = np.abs(B)
    iRGB[np.abs(iRGB) > 1] = 1
#    
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()    
    #outall= 'D:\\Juanma\\'+str(beam)+'\\RGBs\\Pauli RGB'+'_'+str(Acquisition_1)+'.png'
    #fig.savefig(outall, bbox_inches='tight', dpi = 1200)                 
    return(iRGB)    
    
def visRGB_parcel(img1, img2, img3, title,x,y):
    """
    Visualise the RGB of a single acquisition
    """           
    img1 =mask_element(img1,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img2 =mask_element(img2,x,y)
    img3 =mask_element(img3,x,y)
    
    size = np.shape(img2)           
    iRGB = np.zeros([size[0],size[1],3])
    
    R=(img1-img3)
    G=2*img2
    B=(img1+img3)
    
    iRGB[:,:,0] = np.abs(R)/(np.abs(R).mean()*4.5)
    iRGB[:,:,1] = np.abs(G)/(np.abs(G).mean()*4.5)
    iRGB[:,:,2] = np.abs(B)/(np.abs(B).mean()*4.5)
    iRGB[np.abs(iRGB) > 1] = 1
#    
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()    
    return(iRGB)    
    
def visRGB_L_Contrast(img1, img2, img3, img4,img5,img6,img7,img8,img9, title,L1,L2,L3,x,y,beam,Acquisition_1,Acquisition_2,Parcela,SM):
    """
    Visualise the RGB of a single acquisition
    """           
#    img1 =mask_element(img1,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
#    img2 =mask_element(img2,x,y)
#    img3 =mask_element(img3,x,y)
#    img4 =mask_element(img4,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
#    img5 =mask_element(img5,x,y)
#    img6 =mask_element(img6,x,y)
#    img7 =mask_element(img7,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
#    img8 =mask_element(img8,x,y)
#    img9 =mask_element(img9,x,y)
    size = np.shape(img2)           
    iRGB1 = np.zeros([size[0],size[1],3])

#    R1=0.5*(img1-img3) # for components of C matrix. Delete if the components are those of the T matrix
#    B1=2*img2 # or 2*img
#    G1=0.5*(img1+img3)
    
    R1=img1
    G1=img2
    B1=img3

    K=1
    iRGB1[:,:,0] = (np.abs(R1))/(np.abs(R1).mean()*2.5)
    iRGB1[:,:,1] = (np.abs(G1))/(np.abs(G1).mean()*2.5)
    iRGB1[:,:,2] = (np.abs(B1))/(np.abs(B1).mean()*2.5)
    
    iRGB1[:,:,0] = np.multiply(iRGB1[:,:,0],L1)
    iRGB1[:,:,1] = np.multiply(iRGB1[:,:,1],L1)
    iRGB1[:,:,2] = np.multiply(iRGB1[:,:,2],L1)   
    
    iRGB1[:,:,0] = iRGB1[:,:,0]*K
    iRGB1[:,:,1] = iRGB1[:,:,1]*K
    iRGB1[:,:,2] = iRGB1[:,:,2]*K 
    #iRGB1[np.abs(iRGB1) > 1] = 1
    
    iRGB2 = np.zeros([size[0],size[1],3])
#    R2=0.5*(img4-img6) # for components of C matrix. Delete if the components are those of the T matrix
#    B2=2*img5 # or 2*img
#    G2=0.5*(img4+img6)
    
    R2=img4
    G2=img5
    B2=img6
    
    iRGB2[:,:,0] = (np.abs(R2))/(np.abs(R2).mean()*2.5)
    iRGB2[:,:,1] = (np.abs(G2))/(np.abs(G2).mean()*2.5)
    iRGB2[:,:,2] = (np.abs(B2))/(np.abs(B2).mean()*2.5)
    
    iRGB2[:,:,0] = np.multiply(iRGB2[:,:,0],L2)
    iRGB2[:,:,1] = np.multiply(iRGB2[:,:,1],L2)
    iRGB2[:,:,2] = np.multiply(iRGB2[:,:,2],L2)    
    
    iRGB2[:,:,0] = iRGB2[:,:,0]*K
    iRGB2[:,:,1] = iRGB2[:,:,1]*K
    iRGB2[:,:,2] = iRGB2[:,:,2]*K 
    #iRGB2[np.abs(iRGB2) > 1] = 1
    
#    iRGB2[:,:,0] = (np.abs(R1)/(np.abs(R1).max()))
#    iRGB2[:,:,1] = (np.abs(G1)/(np.abs(G1).max()))
#    iRGB2[:,:,2] = (np.abs(B1)/(np.abs(B1).max()))
#    
#    iRGB2[:,:,0] = np.multiply(iRGB2[:,:,0],L2)
#    iRGB2[:,:,1] = np.multiply(iRGB2[:,:,1],L2)
#    iRGB2[:,:,2] = np.multiply(iRGB2[:,:,2],L2)    
#    
#    iRGB2[:,:,0] = iRGB2[:,:,0]*K
#    iRGB2[:,:,1] = iRGB2[:,:,1]*K
#    iRGB2[:,:,2] = iRGB2[:,:,2]*K 
#    iRGB2[np.abs(iRGB2) > 1] = 1

    iRGB3 = np.zeros([size[0],size[1],3])
#    R3=0.5*(img7-img9) # for components of C matrix. Delete if the components are those of the T matrix
#    B3=2*img8 # or 2*img   
#    G3=0.5*(img7+img9)
#   
    R3=img7
    G3=img8
    B3=img9                                                  
    iRGB3[:,:,0] = (np.abs(R3))/(np.abs(R3).mean()*2.5)
    iRGB3[:,:,1] = (np.abs(G3))/(np.abs(G3).mean()*2.5)
    iRGB3[:,:,2] = (np.abs(B3))/(np.abs(B3).mean()*2.5)
    
#    iRGB3[:,:,0] = (np.abs(R1)/(np.abs(R1).max()))
#    iRGB3[:,:,1] = (np.abs(G1)/(np.abs(G1).max()))
#    iRGB3[:,:,2] = (np.abs(B1)/(np.abs(B1).max()))
    
    iRGB3[:,:,0] = np.multiply(iRGB3[:,:,0],L3)
    iRGB3[:,:,1] = np.multiply(iRGB3[:,:,1],L3)
    iRGB3[:,:,2] = np.multiply(iRGB3[:,:,2],L3)    
    
    iRGB3[:,:,0] = iRGB3[:,:,0]*K
    iRGB3[:,:,1] = iRGB3[:,:,1]*K
    iRGB3[:,:,2] = iRGB3[:,:,2]*K 
    #iRGB3[np.abs(iRGB3) > 1] = 1
    #iRGB=iRGB3
    iRGB=iRGB1+iRGB2+iRGB3
    #K=0.25
    iRGB=iRGB*K
    iRGB[np.abs(iRGB) > 1] = 1
    fig, (ax3) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    #title='W with L and with K'
    ax3.set_title(title)
    ax3.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()   
    outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\'+SM+' scattering from '+str(Acquisition_1)+' to '+str(Acquisition_2)+'.png'
    #fig.savefig(outall, bbox_inches='tight') 
    #plt.close()    
    return(iRGB)     

def visRGB_L_Contrast_eigedecomposition(img1, img2, img3, title,L,x,y,beam,Acquisition_1,Parcela,SM):
    """
    Visualise the RGB of a single acquisition
    """           
    img1 =mask_element(img1,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img2 =mask_element(img2,x,y)
    img3 =mask_element(img3,x,y)
    size = np.shape(img2)           
    iRGB1 = np.zeros([size[0],size[1],3])

    #R1=img2 # for components of C matrix. Delete if the components are those of the T matrix
    #G1=img1
    #B1=img3
    
    #R1=0.5*(img1-img3) # for components of C matrix. Delete if the components are those of the T matrix
    #G1=2*img2 # or 2*img
    #B1=0.5*(img1+img3)

    B1=img1
    R1=img2
    G1=img3
    K=1
    iRGB1[:,:,0] = (np.abs(R1))
    iRGB1[:,:,1] = (np.abs(G1))
    iRGB1[:,:,2] = (np.abs(B1))
    
    iRGB1[:,:,0] = np.multiply(iRGB1[:,:,0],(L))
    iRGB1[:,:,1] = np.multiply(iRGB1[:,:,1],(L))
    iRGB1[:,:,2] = np.multiply(iRGB1[:,:,2],(L))   
    
    iRGB1[:,:,0] = iRGB1[:,:,0]*K
    iRGB1[:,:,1] = iRGB1[:,:,1]*K
    iRGB1[:,:,2] = iRGB1[:,:,2]*K 
    iRGB1[np.abs(iRGB1) > 1] = 1
#    
#    iRGB1[:,:,0] = (np.abs(G1)/(np.abs(G1).max()))
#    iRGB1[:,:,1] = (np.abs(B1)/(np.abs(B1).max()))
#    iRGB1[:,:,2] = (np.abs(R1)/(np.abs(R1).max()))  
#    
#    iRGB1[:,:,0] = np.multiply(iRGB1[:,:,0],L)
#    iRGB1[:,:,1] = np.multiply(iRGB1[:,:,1],L)
#    iRGB1[:,:,2] = np.multiply(iRGB1[:,:,2],L)   
#    
#    iRGB1[:,:,0] = iRGB1[:,:,0]*K
#    iRGB1[:,:,1] = iRGB1[:,:,1]*K
#    iRGB1[:,:,2] = iRGB1[:,:,2]*K 
#    iRGB1[np.abs(iRGB1) > 1] = 1
    
    fig, (ax3) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    #title='W with L and with K'
    ax3.set_title(title)
    ax3.imshow(np.abs(iRGB1))
    plt.axis("off")
    plt.tight_layout()   
    outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\Eigendecomposition '+SM+' '+str(Acquisition_1)+'.png'
    fig.savefig(outall, bbox_inches='tight') 
    plt.close()    
    return(iRGB1)    
    
def visRGB_L_Contrast_Agrisar(img1, img2, img3,title,L,Acquisition_1,Acquisition_2,SM,Crop_name,Crop_number):
    """
    Visualise the RGB of a single acquisition
    """               
    size = np.shape(img2)           
    iRGB = np.zeros([size[0],size[1],3])
#    R=(img3-img2) # for components of C matrix. Delete if the components are those of the T matrix
#    B=(img3+img2)
#    G=2*img1
    B=img1
    R=img2
    G=img3

    K=1
    iRGB[:,:,0] = (np.abs(R))
    iRGB[:,:,1] = (np.abs(G))
    iRGB[:,:,2] = (np.abs(B))
    
    iRGB[:,:,0] = np.multiply(iRGB[:,:,0],L)
    iRGB[:,:,1] = np.multiply(iRGB[:,:,1],L)
    iRGB[:,:,2] = np.multiply(iRGB[:,:,2],L)    
    
    iRGB[:,:,0] = iRGB[:,:,0]*K
    iRGB[:,:,1] = iRGB[:,:,1]*K
    iRGB[:,:,2] = iRGB[:,:,2]*K 
    iRGB[np.abs(iRGB) > 1] = 1
    
    fig, (ax3) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    #title='W with L and with K'
    ax3.set_title(title)
    ax3.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()   
    outall= 'D:\\Juanma - Agrisar 2009\\Data-20180809T211615Z-001\\Data\\Crop-Mosaics\\'+Crop_name+'\\Change detector\\'+Crop_number+'//'+SM+' scattering from '+str(Acquisition_1)+' to '+str(Acquisition_2)+'.png'
    fig.savefig(outall, bbox_inches='tight') 
    plt.close()    
    return(iRGB)  

def visRGB_L_Contrast_Agrisar_total(img1, img2, img3,img4,img5,img6,img7,img8,img9,title,L1,L2,L3,Acquisition_1,Acquisition_2,SM,Crop_name,Crop_number):
    """
    Visualise the RGB of a single acquisition
    """               
    size = np.shape(img2)           
    iRGB = np.zeros([size[0],size[1],3])
    iRGB1 = np.zeros([size[0],size[1],3])
#    R=(img3-img2) # for components of C matrix. Delete if the components are those of the T matrix
#    B=(img3+img2)
#    G=2*img1
    B1=img1 # inverse of following the logic of a pauli basis vector
    R1=img2
    G1=img3
    K=1                                                
    iRGB1[:,:,0] = (np.abs(R1))
    iRGB1[:,:,1] = (np.abs(G1))
    iRGB1[:,:,2] = (np.abs(B1))
    
    iRGB1[:,:,0] = np.multiply(iRGB1[:,:,0],L1)
    iRGB1[:,:,1] = np.multiply(iRGB1[:,:,1],L1)
    iRGB1[:,:,2] = np.multiply(iRGB1[:,:,2],L1)    
    
    iRGB1[:,:,0] = iRGB1[:,:,0]*K
    iRGB1[:,:,1] = iRGB1[:,:,1]*K
    iRGB1[:,:,2] = iRGB1[:,:,2]*K 
    iRGB1[np.abs(iRGB1) > 1] = 1
    
    
    iRGB2 = np.zeros([size[0],size[1],3])
#    R2=(img6-img5) # for components of C matrix. Delete if the components are those of the T matrix
#    B2=(img6+img5)
#    G2=2*img4   
    B2=img4
    R2=img5
    G2=img6
                                               
    iRGB2[:,:,0] = (np.abs(R2))
    iRGB2[:,:,1] = (np.abs(G2))
    iRGB2[:,:,2] = (np.abs(B2))
    
    iRGB2[:,:,0] = np.multiply(iRGB2[:,:,0],L2)
    iRGB2[:,:,1] = np.multiply(iRGB2[:,:,1],L2)
    iRGB2[:,:,2] = np.multiply(iRGB2[:,:,2],L2)    
    
    iRGB2[:,:,0] = iRGB2[:,:,0]*K
    iRGB2[:,:,1] = iRGB2[:,:,1]*K
    iRGB2[:,:,2] = iRGB2[:,:,2]*K 
    iRGB2[np.abs(iRGB2) > 1] = 1
    
    iRGB3 = np.zeros([size[0],size[1],3])
#    R3=(img9-img8) # for components of C matrix. Delete if the components are those of the T matrix
#    B3=(img9+img8)
#    G3=2*img7    
    B3=img7
    R3=img8
    G3=img9                                              
    iRGB3[:,:,0] = (np.abs(R3))
    iRGB3[:,:,1] = (np.abs(G3))
    iRGB3[:,:,2] = (np.abs(B3))
    
    iRGB3[:,:,0] = np.multiply(iRGB3[:,:,0],L3)
    iRGB3[:,:,1] = np.multiply(iRGB3[:,:,1],L3)
    iRGB3[:,:,2] = np.multiply(iRGB3[:,:,2],L3)    
    
    iRGB3[:,:,0] = iRGB3[:,:,0]*K
    iRGB3[:,:,1] = iRGB3[:,:,1]*K
    iRGB3[:,:,2] = iRGB3[:,:,2]*K 
    iRGB3[np.abs(iRGB3) > 1] = 1
    
    iRGB=iRGB1+iRGB2+iRGB3
    #iRGB=np.log10(np.abs(iRGB1))+np.log10(np.abs(iRGB2))+np.log10(np.abs(iRGB3))

    fig, (ax3) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    #title='W with L and with K'
    ax3.set_title(title)
    ax3.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()   
    outall= 'D:\\Juanma - Agrisar 2009\\Data-20180809T211615Z-001\\Data\\Crop-Mosaics\\'+Crop_name+'\\Change detector\\'+Crop_number+'//'+SM+' scattering from '+str(Acquisition_1)+' to '+str(Acquisition_2)+'.png'
    fig.savefig(outall, bbox_inches='tight') 
    plt.close()    
    return(iRGB)      
    
    
def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def visHSV(iRGB, title):
    hsv_image = cv2.cvtColor(iRGB, cv2.COLOR_RGB2HSV)
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True)
    ax1.set_title(title)
    ax1.imshow(np.abs(hsv_image))
    hue, sat, val = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
    plt.figure(figsize=(10,8))
    plt.subplot(311)                             #plot in the first cell
    plt.subplots_adjust(hspace=.5)
    plt.title("Hue")
    plt.hist(np.ndarray.flatten(hue), bins=180)
    plt.subplot(312)                             #plot in the second cell
    plt.title("Saturation")
    plt.hist(np.ndarray.flatten(sat), bins=128)
    plt.subplot(313)                             #plot in the third cell
    plt.title("Luminosity Value")
    plt.hist(np.ndarray.flatten(val), bins=128)
    plt.show()    
          
def vis2RGB(ia1, ib1, ic1,
           ia2, ib2, ic2,
           title1 = 'RGB image: First',
           title2 = 'RGB image: Second',
           scalea1 = [],
           scaleb1 = [],
           scalec1 = [],
           scalea2 = [],
           scaleb2 = [],
           scalec2 = [],
           flag = 0, outall = []):
    """
    Visualise the RGB of a single acquisition. The order of images in argument is RGB
    """           
    if scalea1 == []:
       scalea1 = np.abs(ia1).mean()*2.5
    if scaleb1 == []:
       scaleb1 = np.abs(ib1).mean()*2.5
    if scalec1 == []:
       scalec1 = np.abs(ic1).mean()*2.5
    if scalea2 == []:
       scalea2 = np.abs(ia2).mean()*2.5
    if scaleb2 == []:
       scaleb2 = np.abs(ib2).mean()*2.5
    if scalec2 == []:
       scalec2 = np.abs(ic2).mean()*2.5

    size = np.shape(ia1)           
    RGB1 = np.zeros([size[0],size[1],3])
    RGB1[:,:,0] = np.abs(ia1)/scalea1
    RGB1[:,:,1] = np.abs(ib1)/scaleb1
    RGB1[:,:,2] = np.abs(ic1)/scalec1
    RGB1[np.abs(RGB1) > 1] = 1

    RGB2 = np.zeros([size[0],size[1],3])
    RGB2[:,:,0] = np.abs(ia2)/scalea2
    RGB2[:,:,1] = np.abs(ib2)/scaleb2
    RGB2[:,:,2] = np.abs(ic2)/scalec2
    RGB2[np.abs(RGB2) > 1] = 1

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.set_title(title1)
    ax1.imshow(np.abs(RGB1))
    ax2.set_title(title2)
    ax2.imshow(np.abs(RGB2))
    
    if flag == 1:
        fig.savefig(outall, bbox_inches='tight')
#        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#        fig.savefig(out1, bbox_inches=extent)

    return   

def mask_element(img,x,y):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """   
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(x[0],y[0]), (x[1],y[1]), (x[2],y[2]),(x[3],y[3])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = 2  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (1,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    pixels_in_parcel=np.count_nonzero(mask == 1)
    masked_C_element = np.zeros(img.shape, dtype=np.complex)
    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
    
    for i in range(y_max-y_min):
            for j in range(x_max-x_min):
            
                if mask[i][j] == 1:
                    masked_C_element[i][j]=img[i][j]
                else:
                    masked_C_element[i][j]=0
                
    return(masked_C_element)
    
def mask_parcel_4_clicks(img,x,y):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """   
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(x[0],y[0]), (x[1],y[1]), (x[2],y[2]),(x[3],y[3])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = 2  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (1,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_C_element = np.zeros(img.shape, dtype=np.float32)
    
    for i in range(img.shape[0]):
            for j in range(img.shape[1]):
            
                if mask[i][j] == 1:
                    masked_C_element[i][j]=img[i][j]
                else:
                    masked_C_element[i][j]=0
                
    return(masked_C_element)
    
def get_pixel_values(New_HV_img,x,y):
    mask = np.zeros(New_HV_img.shape, dtype=np.uint8)
    roi_corners = np.array([[(x[0],y[0]), (x[1],y[1]), (x[2],y[2]),(x[3],y[3])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = 2  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (1,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    pixels_in_parcel=np.count_nonzero(mask == 1)
    values_in_parcel=np.zeros([pixels_in_parcel])
    k=-1
    for a in range(New_HV_img.shape[0]):
            for b in range(New_HV_img.shape[1]):
                
                if mask[a][b] == 1:
                    k=k+1
                    #values_in_parcel[k]=10*math.log10(New_HV_img[a][b])
                    values_in_parcel[k]=New_HV_img[a][b]
    for a in range(values_in_parcel.shape[0]):
      if values_in_parcel[a] <= 0:
          values_in_parcel[a]=(values_in_parcel[a]+values_in_parcel[(a-1)])/2
    for a in range(values_in_parcel.shape[0]):
        values_in_parcel[a]=10*math.log10(values_in_parcel[a])    
   
    return(values_in_parcel)
    
def mean_and_deviation(img,x,y):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """   
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(x[0],y[0]), (x[1],y[1]), (x[2],y[2]),(x[3],y[3])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = 2  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (1,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    pixels_in_parcel=np.count_nonzero(mask == 1)
    #masked_C_element = np.zeros(img.shape, dtype=np.float32)
    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
    values_in_parcel=np.zeros([pixels_in_parcel])
    k=-1
    for i in range(y_max-y_min):
            for j in range(x_max-x_min):
                
                if mask[i][j] == 1:
                    k=k+1
                    values_in_parcel[k]=img[i][j]
                    
    std_deviation=np.std(values_in_parcel, dtype=np.float64)
    mean=np.mean(values_in_parcel, dtype=np.float64)     
                     
    return(mean, std_deviation)
    
def mean_and_deviation_backscatter(img,x,y):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """   
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(x[0],y[0]), (x[1],y[1]), (x[2],y[2]),(x[3],y[3])]], dtype=np.int32)
    # fill the ROI so it doesn't get wiped out when the mask is applied
    channel_count = 2  # i.e. 3 or 4 depending on your image
    ignore_mask_color = (1,)*channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    pixels_in_parcel=np.count_nonzero(mask == 1)
    #masked_C_element = np.zeros(img.shape, dtype=np.float32)
    y_min=int(min(y))
    y_max=int(max(y))
    x_min=int(min(x))
    x_max=int(max(x))
    values_in_parcel=np.zeros([pixels_in_parcel])
    k=-1
    
    for i in range(y_max-y_min):
            for j in range(x_max-x_min):
                
                if mask[i][j] == 1:
                    k=k+1
                    #values_in_parcel[k]=10*math.log10(img[i][j])
                    values_in_parcel[k]=(img[i][j])
    
    for a in range(values_in_parcel.shape[0]):
      if values_in_parcel[a] <= 0:
          values_in_parcel[a]=(values_in_parcel[a]+values_in_parcel[(a-1)])/2
    for a in range(values_in_parcel.shape[0]):
        values_in_parcel[a]=10*math.log10(values_in_parcel[a])  
    
    std_deviation=np.std(values_in_parcel, dtype=np.float64)
    mean=np.mean(values_in_parcel, dtype=np.float64)     
                     
    return(mean, std_deviation)
    
#params = {
##    'text.latex.preamble': ['\\usepackage{gensymb}'],
##    'image.origin': 'lower',
##    'image.interpolation': 'nearest',
##    'image.cmap': 'gray',
##    'axes.grid': False,
##    'savefig.dpi': 150,  # to adjust notebook inline plot size
#    'axes.labelsize': 10, # fontsize for x and y labels (was 10)
#    'axes.titlesize': 14,
#    'font.size': 14, # was 10
#    'legend.fontsize': 6, # was 10
#    'xtick.labelsize': 8,
#    'ytick.labelsize': 8,
##    'text.usetex': True,
##    'figure.figsize': [100, 100],
#    'font.family': 'normal',
#}
#matplotlib.rcParams.update(params)


def plot_image(beam,Acquisition_1,Image, Title,save):
    """
    ...
    """   
    Title1=Title+' - Beam : '+str(beam)+' - Date: '+str(Acquisition_1)
    fig, (ax100) = plt.subplots(1)
    ax100.set_title(Title1)
    im100=ax100.imshow(Image, cmap = 'gray', vmin=0, vmax=(1))
    fig.colorbar(im100,ax=ax100)
    plt.tight_layout()
    if save == 1:
        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\'+Title+'\\'+Title+'_'+str(Acquisition_1)+'.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)    
    #plt.close()
    return
    
def plot_backscatter(beam,Acquisition_1,C11,C22,C33):
    """
    ...
    """      
    fig, (ax1, ax4, ax5) = plt.subplots(3, sharex=False, sharey=False)
    ax1.set_title('HH - Beam : '+str(beam)+' - Date: '+str(Acquisition_1))
    im1=ax1.imshow(C11, cmap = 'jet', vmin=-30, vmax=0)
    fig.colorbar(im1,ax=ax1)
    ax4.set_title('HV - Beam : '+str(beam)+' - Date: '+str(Acquisition_1))
    im4=ax4.imshow(C22, cmap = 'jet', vmin=-30, vmax=0)
    fig.colorbar(im4,ax=ax4)
    ax5.set_title('VV - Beam : '+str(beam)+' - Date: '+str(Acquisition_1))
    im5=ax5.imshow(C33, cmap = 'jet', vmin=-30, vmax=0)
    fig.colorbar(im5,ax=ax5)
    plt.tight_layout()
    
    return

def color_bar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    #ax.tick_params(labelsize=10)
    cax = divider.append_axes("bottom", size="5%", pad=0.05)   
    return fig.colorbar(mappable, cax=cax,orientation="horizontal")


def plot_eigenvalues(beam,Acquisition_1,Acquisition_2,New_T1,vmin, vmax,title,save,x,y,Parcela,L):
    """
    ...
    """     
    #New_T1 =mask_element(New_T1,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    #New_T2 =mask_element(New_T2,x,y)
    #New_T3 =mask_element(New_T3,x,y)
    
    fig, (ax3) = plt.subplots(figsize=(8, 9))
    ax3.set_title(title)
    im3=ax3.imshow(New_T1, cmap = 'jet', vmin=vmin, vmax=vmax)
    color_bar(im3)
    ax3.axis('off')
    plt.tight_layout()
#    if save == 1:
#        outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\'+L+' scattering from '+str(Acquisition_1)+' to '+str(Acquisition_2)+'.png'
#        fig.savefig(outall, bbox_inches='tight')     
    return  
    

    #matplotlib.rcParams.update({'font.size': 12})
    fig, (ax1) = plt.subplots(figsize=(15, 10))
 #   plt.ylabel('ylabel', fontsize=30)
    #plt.axis('off')
    ax1.set_title('Increase in intensity. Dates: '+str(Acquisition_1))
    im1=ax1.imshow(New_T1, cmap = 'jet', vmin=vmin, vmax=vmax)
    plt.tight_layout()
    color_bar(im1)

    
    if save == 1:
#        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Eigenvalues\\'+Title+'_'+str(Acquisition_1)+'.png'
        #outall= 'D:\\IGARSS\\2018\\Images\\\Eigenvalues\\'+'L1'+'_'+str(Acquisition_1)+'.png'
        outall1=outall+'\\'+'L1'+'_'+str(Acquisition_1)+'.png'
        fig.savefig(outall1, bbox_inches='tight', dpi = 1200)
#    fig.colorbar(im1,ax=ax1)
    return
    
    
    
    fig, (ax4) = plt.subplots(figsize=(11, 9))
#    plt.ylabel('ylabel', fontsize=30)
    #plt.axis('off')
#    ax4.set_title('Lamda 2 - Beam : '+str(beam)+' - Date: '+str(Acquisition_1))
#    cv2.line(New_T2,(4778,5136),(4921,5110),(255,255,255),5)
#    cv2.line(New_T2,(4921,5110),(4965,5128),(255,255,255),5)
#    cv2.line(New_T2,(4965,5128),(4810,5165),(255,255,255),5)
#    cv2.line(New_T2,(4810,5165),(4778,5136),(255,255,255),5)
    im4=ax4.imshow(New_T2, cmap = 'jet', vmin=vmin, vmax=vmax)
    ax4.axis('off')
    color_bar(im4)
    plt.tight_layout()
    if save == 1:
#        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Eigenvalues\\'+Title+'_'+str(Acquisition_1)+'.png'
        outall2=outall+'\\'+'L2'+'_'+str(Acquisition_1)+'.png'
        fig.savefig(outall2, bbox_inches='tight', dpi = 1200)
    
    fig, (ax5) = plt.subplots(figsize=(11, 9))
#    plt.ylabel('ylabel', fontsize=30)
    #plt.axis('off')
#    ax5.set_title('Decrease in intensity. Dates: '+str(Acquisition_1))
#    cv2.line(New_T3,(4778,5136),(4921,5110),(255,255,255),5)
#    cv2.line(New_T3,(4921,5110),(4965,5128),(255,255,255),5)
#    cv2.line(New_T3,(4965,5128),(4810,5165),(255,255,255),5)
#    cv2.line(New_T3,(4810,5165),(4778,5136),(255,255,255),5)
    im5=ax5.imshow(New_T3, cmap = 'jet', vmin=vmin, vmax=vmax)  
    plt.ylabel('ylabel', fontsize=30)
    ax5.axis('off')
    color_bar(im5)
    plt.tight_layout()
    if save == 1:
#        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Eigenvalues\\'+Title+'_'+str(Acquisition_1)+'.png'
        outall3=outall+'\\'+'L3'+'_'+str(Acquisition_1)+'.png'
        fig.savefig(outall3, bbox_inches='tight', dpi = 1200)
       
    #plt.close()
    return

def plot_alphas_or_betas(beam,Acquisition_1,alpha_1,alpha_2,alpha_3,x,y,vmin, vmax,Title,save,outall):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """         
    #alpha_1 =mask_element(alpha_1,x,y) #Mask again to put zeros outside the parcel
    #alpha_2 =mask_element(alpha_2,x,y)
    #alpha_3 =mask_element(alpha_3,x,y)
    

    fig, (ax10) = plt.subplots(1,sharex=False, sharey=False)
#    ax10.set_title(Title+' angle U1 - Beam : '+str(beam)+' - Date: '+str(Acquisition_1))
#    ax10.set_title('Added Scatering mechanism. Dates: '+str(Acquisition_1))
#    cv2.line(alpha_1,(4778,5136),(4921,5110),(255,255,255),5)
#    cv2.line(alpha_1,(4921,5110),(4965,5128),(255,255,255),5)
#    cv2.line(alpha_1,(4965,5128),(4810,5165),(255,255,255),5)
#    cv2.line(alpha_1,(4810,5165),(4778,5136),(255,255,255),5)
    im10=ax10.imshow(alpha_1, cmap = 'jet', vmin=vmin, vmax=vmax)
    ax10.axis('off')
    color_bar(im10)
    plt.tight_layout()
    plt.figsize=(20, 9)
    if save == 1:
#        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Eigenvalues\\'+Title+'_'+str(Acquisition_1)+'.png'
        outall1=outall+'\\'+'Alpha_1'+'_'+str(Acquisition_1)+'.png'
        fig.savefig(outall1, bbox_inches='tight', dpi = 1200)
    #fig.colorbar(im10,ax=ax10)
    fig, (ax40) = plt.subplots(1,sharex=False, sharey=False)
#    ax40.set_title(Title+' angle U2 - Beam : '+str(beam)+' - Date: '+str(Acquisition_1))
#    cv2.line(alpha_2,(4778,5136),(4921,5110),(255,255,255),5)
#    cv2.line(alpha_2,(4921,5110),(4965,5128),(255,255,255),5)
#    cv2.line(alpha_2,(4965,5128),(4810,5165),(255,255,255),5)
#    cv2.line(alpha_2,(4810,5165),(4778,5136),(255,255,255),5)
    im40=ax40.imshow(alpha_2, cmap = 'jet', vmin=vmin, vmax=vmax)
    ax40.axis('off')
    color_bar(im40)
    plt.tight_layout()
    plt.figsize=(20, 9)
    if save == 1:
#        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Eigenvalues\\'+Title+'_'+str(Acquisition_1)+'.png'
        outall2=outall+'\\'+'Alpha_2'+'_'+str(Acquisition_1)+'.png'
        fig.savefig(outall2, bbox_inches='tight', dpi = 1200)
#    fig.colorbar(im40,ax=ax40)
    fig, (ax50) = plt.subplots(1,sharex=False, sharey=False)
#    ax50.set_title('Removed Scatering mechanism. Dates: '+str(Acquisition_1))
#    cv2.line(alpha_3,(4778,5136),(4921,5110),(255,255,255),5)
#    cv2.line(alpha_3,(4921,5110),(4965,5128),(255,255,255),5)
#    cv2.line(alpha_3,(4965,5128),(4810,5165),(255,255,255),5)
#    cv2.line(alpha_3,(4810,5165),(4778,5136),(255,255,255),5)
    im50=ax50.imshow(alpha_3, cmap = 'jet', vmin=vmin, vmax=vmax)
    ax50.axis('off')
    color_bar(im50)
    plt.tight_layout()
    plt.figsize=(20, 9)
    if save == 1:
#        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Eigenvalues\\'+Title+'_'+str(Acquisition_1)+'.png'
        outall3=outall+'\\'+'Alpha_3'+'_'+str(Acquisition_1)+'.png'
        fig.savefig(outall3, bbox_inches='tight', dpi = 1200)    
    #plt.close()
    return

def plot_average(beam,Acquisition_1,alpha_1,x,y,vmin, vmax,Title,save):
    """
    It opens SLC images as formatted for ESAR
    The header can be read from the ENVI .hdr
    """         
    alpha_1 =mask_element(alpha_1,x,y) #Mask again to put zeros outside the parcel    
    fig, (ax10) = plt.subplots(1,sharex=False, sharey=False)
    ax10.set_title(Title+' angle average  - Beam : '+str(beam)+' - Date: '+str(Acquisition_1))
    #plt.ylim(30, 120)
    im10=ax10.imshow(alpha_1, cmap = 'jet', vmin=vmin, vmax=vmax)
    fig.colorbar(im10,ax=ax10)
    plt.tight_layout()
    if save == 1:
        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\'+Title+'\\'+Title+'_average_'+str(Acquisition_1)+'.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    
    plt.close()
    return

#def plot_histogram(Image,x_intial,x_end, Title,y_limit):
#    fig, (ax500) = plt.subplots(1, sharex=True, sharey=True)
#    Image=Image.astype('float32')
#    hist = cv2.calcHist([Image],[0],None,[256],[x_intial,x_end])
#    plt.hist(Image.ravel(),256,[x_intial,x_end])
#    plt.title(Title)
#    plt.ylim(0, y_limit)
#    plt.show()
#    
#    return

def Plot_time_series(dates,mean_Lamda1,mean_alpha1,mean_Entropy,mean_Anisotropy,std_deviation_Lamda1,std_deviation_alpha1,std_deviation_Entropy,std_deviation_Anisotropy,beam,save,App):   
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0,0]
    x = np.arange(dates.size)
    if App == 'CD':
        x = np.arange(dates.size-1)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks)
    plt.ylim(-0.7, 0.7)
    ax.errorbar(x, mean_Lamda1, yerr=std_deviation_Lamda1, fmt='--s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title('Mean Lamda1 and std deviation')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax = axs[0,1]
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks)
    plt.ylim(-0.7, -0.7)
    ax.errorbar(x, mean_alpha1, yerr=std_deviation_alpha1, fmt='--*',ecolor='r', capthick=2, barsabove=True)
    #ax.set_title('Mean Alpha 1 and std deviation')
    ax.set_title('Mean Lamda 2 and std deviation')
    
    ax = axs[1,0]
    ax.errorbar(x, mean_Entropy, yerr=std_deviation_Entropy, fmt='--x',ecolor='m', capthick=2, barsabove=True)
    #ax.set_title('Entropy and std deviation')
    ax.set_title('Mean Lamda 3 or alpha 3 and std deviation')
    plt.ylim(-0.7, 0.7)
    
    ax = axs[1,1]
    ax.errorbar(x, mean_Anisotropy, yerr=std_deviation_Entropy, fmt='o',ecolor='k', capthick=2, barsabove=True)
    ax.set_title('Anisotropy and std deviation')
    #ax.set_title('Mean lamda 1 or alpha average and std deviation')
    plt.ylim(0, 1)
    fig.suptitle('Time series C Matrix difference change detection '+str(beam))
    #fig.suptitle('Time series Cloude-Potier decomposition '+str(beam))
    plt.show()
    outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Time series\\Eigenvalues.png'
    if save == 1:
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    
    return

def Plot_time_series_backscatter(dates,mean_HH,mean_HV,mean_VV,std_deviation_HH,std_deviation_HV,std_deviation_VV,beam,save,App):   
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0,0]
    x = np.arange(dates.size)
    if App == 'CD':
        x = np.arange(dates.size-1)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    #plt.ylim(0, 0.7)
    ax.errorbar(x, mean_HH, yerr=std_deviation_HH, fmt='o',ecolor='g', capthick=2, barsabove=True)
    ax.set_title('Mean HH and std deviation')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax = axs[0,1]
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    plt.ylim(0, 90)
    ax.errorbar(x, mean_HV, yerr=std_deviation_HV, fmt='o',ecolor='r', capthick=2, barsabove=True)
    ax.set_title('Mean HV std deviation')
    
    ax = axs[1,0]
    ax.errorbar(x, mean_VV, yerr=std_deviation_VV, fmt='o',ecolor='m', capthick=2, barsabove=True)
    ax.set_title('Mean VV and std deviation')
    plt.ylim(0, 1)
    
    ax = axs[1,1]
    ratio=mean_HH-mean_VV
    ax.errorbar(x, ratio, yerr=std_deviation_HH, fmt='o',ecolor='k', capthick=2, barsabove=True)
    ax.set_title('Ratio HH/VV and WRONG std deviation')
    plt.ylim(-15, 15)
    fig.suptitle('Time series Backscatter '+str(beam))
    plt.show()
    if save == 1:
        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Time series\\Backscatter.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)

    return

def Plot_time_series_backscatter_sentinel(dates,mean_HV,mean_VV,std_deviation_HV,std_deviation_VV,save):   
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 9), sharex=True)
    ax = axs[0,0]
    x = np.arange(dates.size)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    fig.autofmt_xdate()
    #   plt.ylim(-30, -10)
    ax.errorbar(x, mean_HV, yerr=std_deviation_HV, fmt='--s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title('Mean HV and std deviation (dB)')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax = axs[0,1]
    x = np.arange(dates.size)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    #plt.ylim(-30, -10)
    fig.autofmt_xdate()
    ax.errorbar(x, mean_VV, yerr=std_deviation_VV,fmt='--s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title('Mean VV std deviation (dB)')

        
    ax = axs[1,0]
    plt.xticks(x, my_xticks,rotation='vertical')
    fig.autofmt_xdate()
    mean_ratio_1=mean_HV-mean_VV
    #plt.ylim(0, 3)
    ax.errorbar(x, mean_ratio_1,fmt='--s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title('Mean ratio HV/VV')

    
    ax = axs[1,1]
    mean_ratio_2=mean_VV-mean_HV
    fig.autofmt_xdate()
    ax.errorbar(x, mean_ratio_2, fmt='--s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title('Ratio VV/HV')
    #plt.ylim(0, 3)
    
    fig.suptitle('Time series Backscatter ')
    plt.tight_layout()
    plt.show()
    if save == 1:
        outall= 'D:\\PhD Info\\Trujillo Datasets\\01 05 2018\\Ascending orbit 22\\Subset.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    return

def Plot_time_series_alphas_or_betas(dates,mean_Lamda1,mean_alpha1,mean_Entropy,mean_Anisotropy,std_deviation_Lamda1,std_deviation_alpha1,std_deviation_Entropy,std_deviation_Anisotropy,beam, Title,save,App):   
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0,0]
    x = np.arange(dates.size)
    if App == 'CD':
        x = np.arange(dates.size-1)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks)
    plt.ylim(-90, 90)
    ax.errorbar(x, mean_Lamda1, yerr=std_deviation_Lamda1, fmt='s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title(Title+' 1 and std deviation')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax = axs[0,1]
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks)
    plt.ylim(-90, 90)
    ax.errorbar(x, mean_alpha1, yerr=std_deviation_alpha1, fmt='s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title(Title+' 2 and std deviation')
    
    ax = axs[1,0]
    ax.errorbar(x, mean_Entropy, yerr=std_deviation_Entropy, fmt='s',ecolor='g', capthick=2, barsabove=True)
    #ax.set_title('Entropy and std deviation')
    ax.set_title(Title+' 3 and std deviation')
    plt.ylim(-90, 90)
    
    ax = axs[1,1]
    ax.errorbar(x, mean_Anisotropy, yerr=std_deviation_Anisotropy, fmt='s',ecolor='g', capthick=2, barsabove=True)
    #ax.set_title('Anisotropy and std deviation')
    ax.set_title(Title+' average and std deviation')
    plt.ylim(-90, 90)
    fig.suptitle('Time series '+ Title+'_angle '+str(beam))
    #fig.suptitle('Time series Cloude-Potier decomposition '+str(beam))
    plt.show()
    
    if save == 1:
        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Time series\\'+Title+'.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
        
    return

def Plot_time_series_H_A(dates,mean_Entropy,mean_Anisotropy,std_deviation_Entropy,std_deviation_Anisotropy,beam,save,App):   
    
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
    ax = axs[0,0]
    x = np.arange(dates.size)
    if App == 2:
        x = np.arange(dates.size-1)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks)
    plt.ylim(0, 1)
    ax.errorbar(x, mean_Entropy, yerr=std_deviation_Entropy, fmt='s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title('Entropy and std deviation')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax = axs[0,1]
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks)
    plt.ylim(0, 1)
    ax.errorbar(x, mean_Anisotropy, yerr=std_deviation_Anisotropy, fmt='s',ecolor='g', capthick=2, barsabove=True)
    #ax.set_title('Mean Alpha 1 and std deviation')
    ax.set_title('Anisotropy and std deviation')
    
    ax = axs[1,0]
    ax.errorbar(x, mean_Entropy, yerr=std_deviation_Entropy, fmt='s',ecolor='g', capthick=2, barsabove=True)
    #ax.set_title('Entropy and std deviation')
    ax.set_title('Entropy and std deviation')
    plt.ylim(0, 1)
    
    ax = axs[1,1]
    ax.errorbar(x, mean_Anisotropy, yerr=std_deviation_Anisotropy, fmt='s',ecolor='g', capthick=2, barsabove=True)
    ax.set_title('Anisotropy and std deviation')
    #ax.set_title('Mean lamda 1 or alpha average and std deviation')
    plt.ylim(0, 1)
    fig.suptitle('Time series Entropy and anysotropy '+str(beam))
    #fig.suptitle('Time series Cloude-Potier decomposition '+str(beam))
    plt.show()
    
    if save == 1:
        outall= 'D:\\Juanma\\'+str(beam)+'\\Parcela A Change detector\\Time series\\Entropy and Anistropy.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    
    return

def vis2(img1, img2, 
         title1 = 'Image 1', 
         title2 = 'Image 2', 
         scale1 = [], 
         scale2 = [],  
         flag = (0, 0), 
         outall = [], out1 = []):
    """
    Visualise any two images. We need to tell it the scaling or it uses the 
    default for magnitude images
    """
    if scale1 == []:
       scale1 = (0, np.abs(img1).mean()*3.5)
    if scale2 == []:
       scale2 = (0, np.abs(img2).mean()*3.5)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    ax1.set_title(title1)
    ax1.imshow(np.abs(img1), cmap = 'gray', vmin=scale1[0], vmax=scale1[1])
    ax2.set_title(title2)
    ax2.imshow(np.abs(img2), cmap = 'gray', vmin=scale2[0], vmax=scale2[1])
    if flag[0] == 1:
        fig.savefig(outall)
    if flag[1] == 1:
        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out1, bbox_inches=extent)

    return(fig)
    
def vis3(img1, img2, img3,
         title1 = 'Image 1', 
         title2 = 'Image 2', 
         title3 = 'Image 3',
         scale1 = [], 
         scale2 = [],
         scale3 = [],
         flag = (0, 0, 0), 
         outall = [], out1 = [], out2 = []):
    """
    Visualise any two images. We need to tell it the scaling or it uses the 
    default for magnitude images
    """
    if scale1 == []:
       scale1 = (0, np.abs(img1).mean()*3.5)
    if scale2 == []:
       scale2 = (0, np.abs(img2).mean()*3.5)
    if scale3 == []:
       scale3 = (0, np.abs(img3).mean()*3.5)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 2, 3, sharex=False, sharey=True)
    ax1.set_title(title1)
    ax1.imshow(np.abs(img1), cmap = 'gray', vmin=scale1[0], vmax=scale1[1])
    ax2.set_title(title2)
    ax2.imshow(np.abs(img2), cmap = 'gray', vmin=scale2[0], vmax=scale2[1])
    ax3.set_title(title3)
    ax3.imshow(np.abs(img3), cmap = 'gray', vmin=scale3[0], vmax=scale3[1])
    if flag[0] == 1:
        fig.savefig(outall)
    if flag[1] == 1:
        extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out1, bbox_inches=extent)

    return(fig)
    
def Plot_time_series_backscatter_sentinel_2(dates,mean_HV,mean_VV,std_deviation_HV,std_deviation_VV,mean_HV_1,mean_VV_1,std_deviation_HV_1,std_deviation_VV_1,save):   
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(19, 9), sharex=True)
    ax = axs[0,0]
    x = np.arange(dates.size)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    fig.autofmt_xdate()
    #   plt.ylim(-30, -10)
    ax.errorbar(x, mean_HV, yerr=std_deviation_HV, fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_HV_1, yerr=std_deviation_HV_1, fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.set_title('Mean HV and std deviation (dB)')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax = axs[0,1]
    x = np.arange(dates.size)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    #plt.ylim(-30, -10)
    fig.autofmt_xdate()
    ax.errorbar(x, mean_VV, yerr=std_deviation_VV,fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_VV_1, yerr=std_deviation_VV_1,fmt='r--o',ecolor='r', capthick=2, barsabove=True)

    ax.set_title('Mean VV std deviation (dB)')

        
    ax = axs[1,0]
    plt.xticks(x, my_xticks,rotation='vertical')
    fig.autofmt_xdate()
    mean_ratio=mean_HV-mean_VV
    mean_ratio_1=mean_HV_1-mean_VV_1

    #plt.ylim(0, 3)
    ax.errorbar(x, mean_ratio,fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_ratio_1,fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.set_title('Mean ratio HV/VV')

    
    ax = axs[1,1]
    mean_ratio=mean_VV-mean_HV
    mean_ratio_1=mean_VV_1-mean_HV_1
    fig.autofmt_xdate()
    ax.errorbar(x, mean_ratio, fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_ratio_1, fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.set_title('Ratio VV/HV and WRONG std deviation')
    plt.ylim(0, 3)
    
    fig.suptitle('Time series Backscatter ')
    plt.tight_layout()
    plt.show()
    if save == 1:
        outall= 'D:\\PhD Info\\Trujillo Datasets\\01 05 2018\\Ascending orbit 22\\Subset.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    return

def Plot_time_series_backscatter_sentinel_3(dates,mean_HV,mean_VV,std_deviation_HV,std_deviation_VV,
                                                  mean_HV_1,mean_VV_1,std_deviation_HV_1,std_deviation_VV_1,
                                                  mean_HV_2,mean_VV_2,std_deviation_HV_2,std_deviation_VV_2,save):   
    
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(19, 9), sharex=True)
    ax = axs[0,0]
    x = np.arange(dates.size)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    fig.autofmt_xdate()
    #   plt.ylim(-30, -10)
    ax.errorbar(x, mean_HV, yerr=std_deviation_HV, fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_HV_1, yerr=std_deviation_HV_1, fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.errorbar(x, mean_HV_2, yerr=std_deviation_HV_2, fmt='b--x',ecolor='b', capthick=2, barsabove=True)
    ax.set_title('Mean HV and std deviation (dB)')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax = axs[0,1]
    x = np.arange(dates.size)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    #plt.ylim(-30, -10)
    fig.autofmt_xdate()
    ax.errorbar(x, mean_VV, yerr=std_deviation_VV,fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_VV_1, yerr=std_deviation_VV_1,fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.errorbar(x, mean_VV_2, yerr=std_deviation_VV_2,fmt='b--x',ecolor='b', capthick=2, barsabove=True)
    ax.set_title('Mean VV std deviation (dB)')

        
    ax = axs[1,0]
    plt.xticks(x, my_xticks,rotation='vertical')
    fig.autofmt_xdate()
    mean_ratio=mean_HV-mean_VV
    mean_ratio_1=mean_HV_1-mean_VV_1
    mean_ratio_2=mean_HV_2-mean_VV_2
    #plt.ylim(0, 3)
    ax.errorbar(x, mean_ratio,fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_ratio_1,fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.errorbar(x, mean_ratio_2,fmt='b--x',ecolor='b', capthick=2, barsabove=True)
    ax.set_title('Mean ratio HV/VV')
    
    ax = axs[1,1]
    mean_ratio=mean_VV-mean_HV
    mean_ratio_1=mean_VV_1-mean_HV_1
    mean_ratio_2=mean_VV_2-mean_HV_2
    fig.autofmt_xdate()
    ax.errorbar(x, mean_ratio, fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_ratio_1, fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.errorbar(x, mean_ratio_2, fmt='b--x',ecolor='b', capthick=2, barsabove=True)
    ax.set_title('Ratio VV/HV and WRONG std deviation')
    #plt.ylim(0, 3)
    
    fig.suptitle('Time series Backscatter ')
    plt.tight_layout()
    plt.show()
    if save == 1:
        outall= 'D:\\PhD Info\\Trujillo Datasets\\01 05 2018\\Ascending orbit 22\\Subset.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    return

def Plot_time_series_backscatter_sentinel_3_no_ratio(dates,mean_HV,mean_VV,std_deviation_HV,std_deviation_VV,
                                                  mean_HV_1,mean_VV_1,std_deviation_HV_1,std_deviation_VV_1,
                                                  mean_HV_2,mean_VV_2,std_deviation_HV_2,std_deviation_VV_2,save):   
    
    fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(19, 9), sharex=True)
    ax = axs[0]
    x = np.arange(dates.size)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    fig.autofmt_xdate()
    #   plt.ylim(-30, -10)
    ax.errorbar(x, mean_HV, yerr=std_deviation_HV, fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_HV_1, yerr=std_deviation_HV_1, fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.errorbar(x, mean_HV_2, yerr=std_deviation_HV_2, fmt='b--x',ecolor='b', capthick=2, barsabove=True)
    ax.set_title('Mean HV and std deviation (dB)')
    
    # With 4 subplots, reduce the number of axis ticks to avoid crowding.
    ax.locator_params(nbins=4)
    ax = axs[1]
    x = np.arange(dates.size)
    my_xticks = dates.tolist()
    plt.xticks(x, my_xticks,rotation='vertical')
    #plt.ylim(-30, -10)
    fig.autofmt_xdate()
    ax.errorbar(x, mean_VV, yerr=std_deviation_VV,fmt='g--s',ecolor='g', capthick=2, barsabove=True)
    ax.errorbar(x, mean_VV_1, yerr=std_deviation_VV_1,fmt='r--o',ecolor='r', capthick=2, barsabove=True)
    ax.errorbar(x, mean_VV_2, yerr=std_deviation_VV_2,fmt='b--x',ecolor='b', capthick=2, barsabove=True)
    ax.set_title('Mean VV std deviation (dB)')
    
    fig.suptitle('Time series Backscatter ')
    plt.tight_layout()
    plt.show()
    if save == 1:
        outall= 'D:\\PhD Info\\Trujillo Datasets\\01 05 2018\\Ascending orbit 22\\Subset.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    return
#plt.figure(100)
#plt.subplot(211)
#x = np.array([0,1,2,3,4,5])
#my_xticks = ['20140522','20140615','20140709','20140802','20140826','20140919']
#plt.xticks(x, my_xticks)
#plt.plot(x, Lamda1_pixel,'ro')
#plt.ylabel('Lamda1 pixel value')
#plt.ylim(0, 0.7)
#plt.show()
#
#plt.subplot(212)
#x = np.array([0,1,2,3,4,5])
#my_xticks = ['20140522','20140615','20140709','20140802','20140826','20140919']
#plt.xticks(x, my_xticks)
#plt.plot(x, alpha1_pixel,'ro')
#plt.ylabel('Alpha angle pixel value')
#plt.ylim(0, 2)
#plt.show()
    
#plt.figure(200)
#plt.subplot(211)
#x = np.array([0,1,2,3,4,5])
#my_xticks = ['20140522','20140615','20140709','20140802','20140826','20140919']
#plt.xticks(x, my_xticks)
#plt.plot(x, Entropy1__pixel,'ro')
#plt.ylabel('Entropy pixel value')
#plt.ylim(0, 1)
#plt.show()
#
#plt.subplot(212)
#x = np.array([0,1,2,3,4,5])
#my_xticks = ['20140522','20140615','20140709','20140802','20140826','20140919']
#plt.xticks(x, my_xticks)
#plt.plot(x, Anisotropy1_pixel,'ro')
#plt.ylabel('Anysotropy pixel value')
#plt.ylim(0, 1)
#plt.show()
    
######################################################################################################################
    ##############################################################################################################
        ####################################################################################################
            #########################################################################################
            
def plot_Alpha_H_A(beam,df_a,df_b,df_c,df_d,df_e,df_f):
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Cloude-Potier decomposition: Alpha angles five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Alpha_1'], yerr=df_a['std_deviation_Alpha_1'],fmt='c--o',ecolor='c', capthick=3,label='Alpha 1 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Alpha_1'], yerr=df_b['std_deviation_Alpha_1'],fmt='k--o',ecolor='k', capthick=3,label='Alpha 1 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Alpha_1'], yerr=df_c['std_deviation_Alpha_1'],fmt='r--o',ecolor='r', capthick=3,label='Alpha 1 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Alpha_1'], yerr=df_d['std_deviation_Alpha_1'],fmt='g--o',ecolor='g', capthick=3,label='Alpha 1 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Alpha_1'], yerr=df_e['std_deviation_Alpha_1'],fmt='b--o',ecolor='b', capthick=3,label='Alpha 1 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Alpha_1'], yerr=df_f['std_deviation_Alpha_1'],fmt='y--o',ecolor='y', capthick=3,label='Alpha 1 - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 1)
    Title='Cloude-Potier decomposition: Entropy five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Entropy'], yerr=df_a['std_deviation_Entropy'],fmt='c--o',ecolor='c', capthick=3,label='Entropy - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Entropy'], yerr=df_b['std_deviation_Entropy'],fmt='k--o',ecolor='k', capthick=3,label='Entropy - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Entropy'], yerr=df_c['std_deviation_Entropy'],fmt='r--o',ecolor='r', capthick=3,label='Entropy - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Entropy'], yerr=df_d['std_deviation_Entropy'],fmt='g--o',ecolor='g', capthick=3,label='Entropy - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Entropy'], yerr=df_e['std_deviation_Entropy'],fmt='b--o',ecolor='b', capthick=3,label='Entropy - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Entropy'], yerr=df_f['std_deviation_Entropy'],fmt='y--o',ecolor='y', capthick=3,label='Entropy- Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 1)
    Title='Cloude-Potier decomposition: Anisotropy five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Anisotropy'], yerr=df_a['std_deviation_Anisotropy'],fmt='c--o',ecolor='c', capthick=3,label='Anisotropy - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Anisotropy'], yerr=df_b['std_deviation_Anisotropy'],fmt='k--o',ecolor='k', capthick=3,label='Anisotropy - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Anisotropy'], yerr=df_c['std_deviation_Anisotropy'],fmt='r--o',ecolor='r', capthick=3,label='Anisotropy - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Anisotropy'], yerr=df_d['std_deviation_Anisotropy'],fmt='g--o',ecolor='g', capthick=3,label='Anisotropy - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Anisotropy'], yerr=df_e['std_deviation_Anisotropy'],fmt='b--o',ecolor='b', capthick=3,label='Anisotropy - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Anisotropy'], yerr=df_f['std_deviation_Anisotropy'],fmt='y--o',ecolor='y', capthick=3,label='Anisotropy- Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-30, 0)
    Title='Cloude-Potier decomposition: VV Backscatter five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['VV'], yerr=df_a['std_deviation_VV'],fmt='c--o',ecolor='c', capthick=3,label='VV - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['VV'], yerr=df_b['std_deviation_VV'],fmt='k--o',ecolor='k', capthick=3,label='VV - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['VV'], yerr=df_c['std_deviation_VV'],fmt='r--o',ecolor='r', capthick=3,label='VV - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['VV'], yerr=df_d['std_deviation_VV'],fmt='g--o',ecolor='g', capthick=3,label='VV - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['VV'], yerr=df_e['std_deviation_VV'],fmt='b--o',ecolor='b', capthick=3,label='VV - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['VV'], yerr=df_f['std_deviation_VV'],fmt='y--o',ecolor='y', capthick=3,label='VV - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-30, 0)
    Title='Cloude-Potier decomposition: HV Backscatter five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['HV'], yerr=df_a['std_deviation_HV'],fmt='c--o',ecolor='c', capthick=3,label='HV - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['HV'], yerr=df_b['std_deviation_HV'],fmt='k--o',ecolor='k', capthick=3,label='HV - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['HV'], yerr=df_c['std_deviation_HV'],fmt='r--o',ecolor='r', capthick=3,label='HV - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['HV'], yerr=df_d['std_deviation_HV'],fmt='g--o',ecolor='g', capthick=3,label='HV - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['HV'], yerr=df_e['std_deviation_HV'],fmt='b--o',ecolor='b', capthick=3,label='HV - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['HV'], yerr=df_f['std_deviation_HV'],fmt='y--o',ecolor='y', capthick=3,label='HV - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-30, 0)
    Title='Cloude-Potier decomposition: HH Backscatter five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['HH'], yerr=df_a['std_deviation_HH'],fmt='c--o',ecolor='c', capthick=3,label='HH - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['HH'], yerr=df_b['std_deviation_HH'],fmt='k--o',ecolor='k', capthick=3,label='HH - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['HH'], yerr=df_c['std_deviation_HH'],fmt='r--o',ecolor='r', capthick=3,label='HH - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['HH'], yerr=df_d['std_deviation_HH'],fmt='g--o',ecolor='g', capthick=3,label='HH - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['HH'], yerr=df_e['std_deviation_HH'],fmt='b--o',ecolor='b', capthick=3,label='HH - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['HH'], yerr=df_f['std_deviation_HH'],fmt='y--o',ecolor='y', capthick=3,label='HH - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
def plot_all_beams(beam,df_a,df_b,df_c,df_d,df_e,df_f,df13_a,df13_b,df13_c,df13_d,df13_e,df13_f,df19_a,df19_b,df19_c,df19_d,df19_e,df19_f):
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Cloude-Potier decomposition: Alpha angles five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Alpha_1'], yerr=df_a['std_deviation_Alpha_1'],fmt='r--o',ecolor='r', capthick=3,label='Alpha 1 - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['Alpha_1'], yerr=df_b['std_deviation_Alpha_1'],fmt='r--s',ecolor='r', capthick=3,label='Alpha 1 - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['Alpha_1'], yerr=df_c['std_deviation_Alpha_1'],fmt='r--x',ecolor='r', capthick=3,label='Alpha 1 - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['Alpha_1'], yerr=df_d['std_deviation_Alpha_1'],fmt='r--*',ecolor='r', capthick=3,label='Alpha 1 - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['Alpha_1'], yerr=df_e['std_deviation_Alpha_1'],fmt='r--p',ecolor='r', capthick=3,label='Alpha 1 - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['Alpha_1'], yerr=df_f['std_deviation_Alpha_1'],fmt='r--D',ecolor='r', capthick=3,label='Alpha 1 - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['Alpha_1'], yerr=df13_a['std_deviation_Alpha_1'],fmt='g--o',ecolor='g', capthick=3,label='Alpha 1 - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['Alpha_1'], yerr=df13_b['std_deviation_Alpha_1'],fmt='g--s',ecolor='g', capthick=3,label='Alpha 1 - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['Alpha_1'], yerr=df13_c['std_deviation_Alpha_1'],fmt='g--x',ecolor='g', capthick=3,label='Alpha 1 - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['Alpha_1'], yerr=df13_d['std_deviation_Alpha_1'],fmt='g--*',ecolor='g', capthick=3,label='Alpha 1 - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['Alpha_1'], yerr=df13_e['std_deviation_Alpha_1'],fmt='g--p',ecolor='g', capthick=3,label='Alpha 1 - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['Alpha_1'], yerr=df13_f['std_deviation_Alpha_1'],fmt='g--D',ecolor='g', capthick=3,label='Alpha 1 - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['Alpha_1'], yerr=df19_a['std_deviation_Alpha_1'],fmt='b--o',ecolor='b', capthick=3,label='Alpha 1 - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['Alpha_1'], yerr=df19_b['std_deviation_Alpha_1'],fmt='b--s',ecolor='b', capthick=3,label='Alpha 1 - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['Alpha_1'], yerr=df19_c['std_deviation_Alpha_1'],fmt='b--x',ecolor='b', capthick=3,label='Alpha 1 - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['Alpha_1'], yerr=df19_d['std_deviation_Alpha_1'],fmt='b--*',ecolor='b', capthick=3,label='Alpha 1 - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['Alpha_1'], yerr=df19_e['std_deviation_Alpha_1'],fmt='b--p',ecolor='b', capthick=3,label='Alpha 1 - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['Alpha_1'], yerr=df19_f['std_deviation_Alpha_1'],fmt='b--D',ecolor='b', capthick=3,label='Alpha 1 - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 1)
    Title='Cloude-Potier decomposition: Entropy five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Entropy'], yerr=df_a['std_deviation_Entropy'],fmt='r--o',ecolor='r', capthick=3,label='Entropy - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['Entropy'], yerr=df_b['std_deviation_Entropy'],fmt='r--s',ecolor='r', capthick=3,label='Entropy - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['Entropy'], yerr=df_c['std_deviation_Entropy'],fmt='r--x',ecolor='r', capthick=3,label='Entropy - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['Entropy'], yerr=df_d['std_deviation_Entropy'],fmt='r--*',ecolor='r', capthick=3,label='Entropy - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['Entropy'], yerr=df_e['std_deviation_Entropy'],fmt='r--p',ecolor='r', capthick=3,label='Entropy - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['Entropy'], yerr=df_f['std_deviation_Entropy'],fmt='r--D',ecolor='r', capthick=3,label='Entropy - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['Entropy'], yerr=df13_a['std_deviation_Entropy'],fmt='g--o',ecolor='g', capthick=3,label='Entropy - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['Entropy'], yerr=df13_b['std_deviation_Entropy'],fmt='g--s',ecolor='g', capthick=3,label='Entropy - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['Entropy'], yerr=df13_c['std_deviation_Entropy'],fmt='g--x',ecolor='g', capthick=3,label='Entropy - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['Entropy'], yerr=df13_d['std_deviation_Entropy'],fmt='g--*',ecolor='g', capthick=3,label='Entropy - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['Entropy'], yerr=df13_e['std_deviation_Entropy'],fmt='g--p',ecolor='g', capthick=3,label='Entropy - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['Entropy'], yerr=df13_f['std_deviation_Entropy'],fmt='g--D',ecolor='g', capthick=3,label='Entropy - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['Entropy'], yerr=df19_a['std_deviation_Entropy'],fmt='b--o',ecolor='b', capthick=3,label='Entropy - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['Entropy'], yerr=df19_b['std_deviation_Entropy'],fmt='b--s',ecolor='b', capthick=3,label='Entropy - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['Entropy'], yerr=df19_c['std_deviation_Entropy'],fmt='b--x',ecolor='b', capthick=3,label='Entropy - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['Entropy'], yerr=df19_d['std_deviation_Entropy'],fmt='b--*',ecolor='b', capthick=3,label='Entropy - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['Entropy'], yerr=df19_e['std_deviation_Entropy'],fmt='b--p',ecolor='b', capthick=3,label='Entropy - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['Entropy'], yerr=df19_f['std_deviation_Entropy'],fmt='b--D',ecolor='b', capthick=3,label='Entropy - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 1)
    Title='Cloude-Potier decomposition: Anisotropy five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Anisotropy'], yerr=df_a['std_deviation_Anisotropy'],fmt='r--o',ecolor='r', capthick=3,label='Anisotropy - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['Anisotropy'], yerr=df_b['std_deviation_Anisotropy'],fmt='r--s',ecolor='r', capthick=3,label='Anisotropy - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['Anisotropy'], yerr=df_c['std_deviation_Anisotropy'],fmt='r--x',ecolor='r', capthick=3,label='Anisotropy - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['Anisotropy'], yerr=df_d['std_deviation_Anisotropy'],fmt='r--*',ecolor='r', capthick=3,label='Anisotropy - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['Anisotropy'], yerr=df_e['std_deviation_Anisotropy'],fmt='r--p',ecolor='r', capthick=3,label='Anisotropy - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['Anisotropy'], yerr=df_f['std_deviation_Anisotropy'],fmt='r--D',ecolor='r', capthick=3,label='Anisotropy - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['Anisotropy'], yerr=df13_a['std_deviation_Anisotropy'],fmt='g--o',ecolor='g', capthick=3,label='Anisotropy - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['Anisotropy'], yerr=df13_b['std_deviation_Anisotropy'],fmt='g--s',ecolor='g', capthick=3,label='Anisotropy - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['Anisotropy'], yerr=df13_c['std_deviation_Anisotropy'],fmt='g--x',ecolor='g', capthick=3,label='Anisotropy - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['Anisotropy'], yerr=df13_d['std_deviation_Anisotropy'],fmt='g--*',ecolor='g', capthick=3,label='Anisotropy - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['Anisotropy'], yerr=df13_e['std_deviation_Anisotropy'],fmt='g--p',ecolor='g', capthick=3,label='Anisotropy - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['Anisotropy'], yerr=df13_f['std_deviation_Anisotropy'],fmt='g--D',ecolor='g', capthick=3,label='Anisotropy - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['Anisotropy'], yerr=df19_a['std_deviation_Anisotropy'],fmt='b--o',ecolor='b', capthick=3,label='Anisotropy - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['Anisotropy'], yerr=df19_b['std_deviation_Anisotropy'],fmt='b--s',ecolor='b', capthick=3,label='Anisotropy - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['Anisotropy'], yerr=df19_c['std_deviation_Anisotropy'],fmt='b--x',ecolor='b', capthick=3,label='Anisotropy - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['Anisotropy'], yerr=df19_d['std_deviation_Anisotropy'],fmt='b--*',ecolor='b', capthick=3,label='Anisotropy - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['Anisotropy'], yerr=df19_e['std_deviation_Anisotropy'],fmt='b--p',ecolor='b', capthick=3,label='Anisotropy - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['Anisotropy'], yerr=df19_f['std_deviation_Anisotropy'],fmt='b--D',ecolor='b', capthick=3,label='Anisotropy - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-30, 0)
    Title='Cloude-Potier decomposition: VV Backscatter five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['VV'], yerr=df_a['std_deviation_VV'],fmt='r--o',ecolor='r', capthick=3,label='VV - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['VV'], yerr=df_b['std_deviation_VV'],fmt='r--s',ecolor='r', capthick=3,label='VV - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['VV'], yerr=df_c['std_deviation_VV'],fmt='r--x',ecolor='r', capthick=3,label='VV - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['VV'], yerr=df_d['std_deviation_VV'],fmt='r--*',ecolor='r', capthick=3,label='VV - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['VV'], yerr=df_e['std_deviation_VV'],fmt='r--p',ecolor='r', capthick=3,label='VV - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['VV'], yerr=df_f['std_deviation_VV'],fmt='r--D',ecolor='r', capthick=3,label='VV - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['VV'], yerr=df13_a['std_deviation_VV'],fmt='g--o',ecolor='g', capthick=3,label='VV - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['VV'], yerr=df13_b['std_deviation_VV'],fmt='g--s',ecolor='g', capthick=3,label='VV - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['VV'], yerr=df13_c['std_deviation_VV'],fmt='g--x',ecolor='g', capthick=3,label='VV - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['VV'], yerr=df13_d['std_deviation_VV'],fmt='g--*',ecolor='g', capthick=3,label='VV - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['VV'], yerr=df13_e['std_deviation_VV'],fmt='g--p',ecolor='g', capthick=3,label='VV - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['VV'], yerr=df13_f['std_deviation_VV'],fmt='g--D',ecolor='g', capthick=3,label='VV - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['VV'], yerr=df19_a['std_deviation_VV'],fmt='b--o',ecolor='b', capthick=3,label='VV - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['VV'], yerr=df19_b['std_deviation_VV'],fmt='b--s',ecolor='b', capthick=3,label='VV - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['VV'], yerr=df19_c['std_deviation_VV'],fmt='b--x',ecolor='b', capthick=3,label='VV - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['VV'], yerr=df19_d['std_deviation_VV'],fmt='b--*',ecolor='b', capthick=3,label='VV - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['VV'], yerr=df19_e['std_deviation_VV'],fmt='b--p',ecolor='b', capthick=3,label='VV - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['VV'], yerr=df19_f['std_deviation_VV'],fmt='b--D',ecolor='b', capthick=3,label='VV - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-30, 0)
    Title='Cloude-Potier decomposition: HV Backscatter five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['HV'], yerr=df_a['std_deviation_HV'],fmt='r--o',ecolor='r', capthick=3,label='HV - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['HV'], yerr=df_b['std_deviation_HV'],fmt='r--s',ecolor='r', capthick=3,label='HV - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['HV'], yerr=df_c['std_deviation_HV'],fmt='r--x',ecolor='r', capthick=3,label='HV - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['HV'], yerr=df_d['std_deviation_HV'],fmt='r--*',ecolor='r', capthick=3,label='HV - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['HV'], yerr=df_e['std_deviation_HV'],fmt='r--p',ecolor='r', capthick=3,label='HV - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['HV'], yerr=df_f['std_deviation_HV'],fmt='r--D',ecolor='r', capthick=3,label='HV - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['HV'], yerr=df13_a['std_deviation_HV'],fmt='g--o',ecolor='g', capthick=3,label='HV - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['HV'], yerr=df13_b['std_deviation_HV'],fmt='g--s',ecolor='g', capthick=3,label='HV - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['HV'], yerr=df13_c['std_deviation_HV'],fmt='g--x',ecolor='g', capthick=3,label='HV - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['HV'], yerr=df13_d['std_deviation_HV'],fmt='g--*',ecolor='g', capthick=3,label='HV - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['HV'], yerr=df13_e['std_deviation_HV'],fmt='g--p',ecolor='g', capthick=3,label='HV - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['HV'], yerr=df13_f['std_deviation_HV'],fmt='g--D',ecolor='g', capthick=3,label='HV - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['HV'], yerr=df19_a['std_deviation_HV'],fmt='b--o',ecolor='b', capthick=3,label='HV - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['HV'], yerr=df19_b['std_deviation_HV'],fmt='b--s',ecolor='b', capthick=3,label='HV - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['HV'], yerr=df19_c['std_deviation_HV'],fmt='b--x',ecolor='b', capthick=3,label='HV - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['HV'], yerr=df19_d['std_deviation_HV'],fmt='b--*',ecolor='b', capthick=3,label='HV - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['HV'], yerr=df19_e['std_deviation_HV'],fmt='b--p',ecolor='b', capthick=3,label='HV - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['HV'], yerr=df19_f['std_deviation_HV'],fmt='b--D',ecolor='b', capthick=3,label='HV - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-30, 0)
    Title='Cloude-Potier decomposition: HH Backscatter five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['HH'], yerr=df_a['std_deviation_HH'],fmt='r--o',ecolor='r', capthick=3,label='HH - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['HH'], yerr=df_b['std_deviation_HH'],fmt='r--s',ecolor='r', capthick=3,label='HH - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['HH'], yerr=df_c['std_deviation_HH'],fmt='r--x',ecolor='r', capthick=3,label='HH - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['HH'], yerr=df_d['std_deviation_HH'],fmt='r--*',ecolor='r', capthick=3,label='HH - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['HH'], yerr=df_e['std_deviation_HH'],fmt='r--p',ecolor='r', capthick=3,label='HH - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['HH'], yerr=df_f['std_deviation_HH'],fmt='r--D',ecolor='r', capthick=3,label='HH - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['HH'], yerr=df13_a['std_deviation_HH'],fmt='g--o',ecolor='g', capthick=3,label='HH - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['HH'], yerr=df13_b['std_deviation_HH'],fmt='g--s',ecolor='g', capthick=3,label='HH - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['HH'], yerr=df13_c['std_deviation_HH'],fmt='g--x',ecolor='g', capthick=3,label='HH - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['HH'], yerr=df13_d['std_deviation_HH'],fmt='g--*',ecolor='g', capthick=3,label='HH - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['HH'], yerr=df13_e['std_deviation_HH'],fmt='g--p',ecolor='g', capthick=3,label='HH - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['HH'], yerr=df13_f['std_deviation_HH'],fmt='g--D',ecolor='g', capthick=3,label='HH - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['HH'], yerr=df19_a['std_deviation_HH'],fmt='b--o',ecolor='b', capthick=3,label='HH - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['HH'], yerr=df19_b['std_deviation_HH'],fmt='b--s',ecolor='b', capthick=3,label='HH - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['HH'], yerr=df19_c['std_deviation_HH'],fmt='b--x',ecolor='b', capthick=3,label='HH - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['HH'], yerr=df19_d['std_deviation_HH'],fmt='b--*',ecolor='b', capthick=3,label='HH - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['HH'], yerr=df19_e['std_deviation_HH'],fmt='b--p',ecolor='b', capthick=3,label='HH - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['HH'], yerr=df19_f['std_deviation_HH'],fmt='b--D',ecolor='b', capthick=3,label='HH - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
def Change_detector_plot(beam,df_a,df_b,df_c,df_d,df_e,df_f):
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Change_detector: Alpha_1 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Alpha_1'], yerr=df_a['std_deviation_Alpha_1'],fmt='c--o',ecolor='c', capthick=3,label='Alpha 1 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Alpha_1'], yerr=df_b['std_deviation_Alpha_1'],fmt='k--o',ecolor='k', capthick=3,label='Alpha 1 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Alpha_1'], yerr=df_c['std_deviation_Alpha_1'],fmt='r--o',ecolor='r', capthick=3,label='Alpha 1 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Alpha_1'], yerr=df_d['std_deviation_Alpha_1'],fmt='g--o',ecolor='g', capthick=3,label='Alpha 1 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Alpha_1'], yerr=df_e['std_deviation_Alpha_1'],fmt='b--o',ecolor='b', capthick=3,label='Alpha 1 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Alpha_1'], yerr=df_f['std_deviation_Alpha_1'],fmt='y--o',ecolor='y', capthick=3,label='Alpha 1 - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Change_detector: Alpha_3 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Alpha_3'], yerr=df_a['std_deviation_Alpha_3'],fmt='c--o',ecolor='c', capthick=3,label='Alpha_3 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Alpha_3'], yerr=df_b['std_deviation_Alpha_3'],fmt='k--o',ecolor='k', capthick=3,label='Alpha_3 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Alpha_3'], yerr=df_c['std_deviation_Alpha_3'],fmt='r--o',ecolor='r', capthick=3,label='Alpha_3 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Alpha_3'], yerr=df_d['std_deviation_Alpha_3'],fmt='g--o',ecolor='g', capthick=3,label='Alpha_3 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Alpha_3'], yerr=df_e['std_deviation_Alpha_3'],fmt='b--o',ecolor='b', capthick=3,label='Alpha_3 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Alpha_3'], yerr=df_f['std_deviation_Alpha_3'],fmt='y--o',ecolor='y', capthick=3,label='Alpha_3- Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-1, 1)
    Title='Change_detector: Lamda_1 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['L1'], yerr=df_a['std_deviation_Lamda1'],fmt='c--o',ecolor='c', capthick=3,label='Lamda1 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['L1'], yerr=df_b['std_deviation_Lamda1'],fmt='k--o',ecolor='k', capthick=3,label='Lamda1 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['L1'], yerr=df_c['std_deviation_Lamda1'],fmt='r--o',ecolor='r', capthick=3,label='Lamda1 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['L1'], yerr=df_d['std_deviation_Lamda1'],fmt='g--o',ecolor='g', capthick=3,label='Lamda1 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['L1'], yerr=df_e['std_deviation_Lamda1'],fmt='b--o',ecolor='b', capthick=3,label='Lamda1 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['L1'], yerr=df_f['std_deviation_Lamda1'],fmt='y--o',ecolor='y', capthick=3,label='Lamda1- Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-1, 1)
    Title='Change_detector: Lamda_3 All parcels five parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['L3'], yerr=df_a['std_deviation_Lamda3'],fmt='c--o',ecolor='c', capthick=3,label='Lamda3 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['L3'], yerr=df_b['std_deviation_Lamda3'],fmt='k--o',ecolor='k', capthick=3,label='Lamda3 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['L3'], yerr=df_c['std_deviation_Lamda3'],fmt='r--o',ecolor='r', capthick=3,label='Lamda3 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['L3'], yerr=df_d['std_deviation_Lamda3'],fmt='g--o',ecolor='g', capthick=3,label='Lamda3 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['L3'], yerr=df_e['std_deviation_Lamda3'],fmt='b--o',ecolor='b', capthick=3,label='Lamda3 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['L3'], yerr=df_f['std_deviation_Lamda3'],fmt='y--o',ecolor='y', capthick=3,label='Lamda3 - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()    
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Change_detector: Beta_1 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Beta_1'], yerr=df_a['std_deviation_Beta_1'],fmt='c--o',ecolor='c', capthick=3,label='Beta_1 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Beta_1'], yerr=df_b['std_deviation_Beta_1'],fmt='k--o',ecolor='k', capthick=3,label='Beta_1 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Beta_1'], yerr=df_c['std_deviation_Beta_1'],fmt='r--o',ecolor='r', capthick=3,label='Beta_1 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Beta_1'], yerr=df_d['std_deviation_Beta_1'],fmt='g--o',ecolor='g', capthick=3,label='Beta_1 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Beta_1'], yerr=df_e['std_deviation_Beta_1'],fmt='b--o',ecolor='b', capthick=3,label='Beta_1 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Beta_1'], yerr=df_f['std_deviation_Beta_1'],fmt='y--o',ecolor='y', capthick=3,label='Beta_1 - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()    
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Change_detector: Beta_3 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Beta_3'], yerr=df_a['std_deviation_Beta_3'],fmt='c--o',ecolor='c', capthick=3,label='Beta_3 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Beta_3'], yerr=df_b['std_deviation_Beta_3'],fmt='k--o',ecolor='k', capthick=3,label='Beta_3 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Beta_3'], yerr=df_c['std_deviation_Beta_3'],fmt='r--o',ecolor='r', capthick=3,label='Beta_3 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Beta_3'], yerr=df_d['std_deviation_Beta_3'],fmt='g--o',ecolor='g', capthick=3,label='Beta_3 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Beta_3'], yerr=df_e['std_deviation_Beta_3'],fmt='b--o',ecolor='b', capthick=3,label='Beta_3 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Beta_3'], yerr=df_f['std_deviation_Beta_3'],fmt='y--o',ecolor='y', capthick=3,label='Beta_3 - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
def Change_detector_plot_all_beams(beam,df_a,df_b,df_c,df_d,df_e,df_f,df13_a,df13_b,df13_c,df13_d,df13_e,df13_f,df19_a,df19_b,df19_c,df19_d,df19_e,df19_f):
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Change detector: Alpha_1 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Alpha_1'], yerr=df_a['std_deviation_Alpha_1'],fmt='r--o',ecolor='r', capthick=3,label='Alpha 1 - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['Alpha_1'], yerr=df_b['std_deviation_Alpha_1'],fmt='r--s',ecolor='r', capthick=3,label='Alpha 1 - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['Alpha_1'], yerr=df_c['std_deviation_Alpha_1'],fmt='r--x',ecolor='r', capthick=3,label='Alpha 1 - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['Alpha_1'], yerr=df_d['std_deviation_Alpha_1'],fmt='r--*',ecolor='r', capthick=3,label='Alpha 1 - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['Alpha_1'], yerr=df_e['std_deviation_Alpha_1'],fmt='r--p',ecolor='r', capthick=3,label='Alpha 1 - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['Alpha_1'], yerr=df_f['std_deviation_Alpha_1'],fmt='r--D',ecolor='r', capthick=3,label='Alpha 1 - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['Alpha_1'], yerr=df13_a['std_deviation_Alpha_1'],fmt='g--o',ecolor='g', capthick=3,label='Alpha 1 - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['Alpha_1'], yerr=df13_b['std_deviation_Alpha_1'],fmt='g--s',ecolor='g', capthick=3,label='Alpha 1 - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['Alpha_1'], yerr=df13_c['std_deviation_Alpha_1'],fmt='g--x',ecolor='g', capthick=3,label='Alpha 1 - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['Alpha_1'], yerr=df13_d['std_deviation_Alpha_1'],fmt='g--*',ecolor='g', capthick=3,label='Alpha 1 - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['Alpha_1'], yerr=df13_e['std_deviation_Alpha_1'],fmt='g--p',ecolor='g', capthick=3,label='Alpha 1 - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['Alpha_1'], yerr=df13_f['std_deviation_Alpha_1'],fmt='g--D',ecolor='g', capthick=3,label='Alpha 1 - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['Alpha_1'], yerr=df19_a['std_deviation_Alpha_1'],fmt='b--o',ecolor='b', capthick=3,label='Alpha 1 - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['Alpha_1'], yerr=df19_b['std_deviation_Alpha_1'],fmt='b--s',ecolor='b', capthick=3,label='Alpha 1 - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['Alpha_1'], yerr=df19_c['std_deviation_Alpha_1'],fmt='b--x',ecolor='b', capthick=3,label='Alpha 1 - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['Alpha_1'], yerr=df19_d['std_deviation_Alpha_1'],fmt='b--*',ecolor='b', capthick=3,label='Alpha 1 - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['Alpha_1'], yerr=df19_e['std_deviation_Alpha_1'],fmt='b--p',ecolor='b', capthick=3,label='Alpha 1 - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['Alpha_1'], yerr=df19_f['std_deviation_Alpha_1'],fmt='b--D',ecolor='b', capthick=3,label='Alpha 1 - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Change detector: Alpha_3 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Alpha_3'], yerr=df_a['std_deviation_Alpha_3'],fmt='r--o',ecolor='r', capthick=3,label='Alpha_3 - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['Alpha_3'], yerr=df_b['std_deviation_Alpha_3'],fmt='r--s',ecolor='r', capthick=3,label='Alpha_3 - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['Alpha_3'], yerr=df_c['std_deviation_Alpha_3'],fmt='r--x',ecolor='r', capthick=3,label='Alpha_3 - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['Alpha_3'], yerr=df_d['std_deviation_Alpha_3'],fmt='r--*',ecolor='r', capthick=3,label='Alpha_3 - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['Alpha_3'], yerr=df_e['std_deviation_Alpha_3'],fmt='r--p',ecolor='r', capthick=3,label='Alpha_3 - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['Alpha_3'], yerr=df_f['std_deviation_Alpha_3'],fmt='r--D',ecolor='r', capthick=3,label='Alpha_3 - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['Alpha_3'], yerr=df13_a['std_deviation_Alpha_3'],fmt='g--o',ecolor='g', capthick=3,label='Alpha_3 - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['Alpha_3'], yerr=df13_b['std_deviation_Alpha_3'],fmt='g--s',ecolor='g', capthick=3,label='Alpha_3 - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['Alpha_3'], yerr=df13_c['std_deviation_Alpha_3'],fmt='g--x',ecolor='g', capthick=3,label='Alpha_3 - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['Alpha_3'], yerr=df13_d['std_deviation_Alpha_3'],fmt='g--*',ecolor='g', capthick=3,label='Alpha_3 - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['Alpha_3'], yerr=df13_e['std_deviation_Alpha_3'],fmt='g--p',ecolor='g', capthick=3,label='Alpha_3 - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['Alpha_3'], yerr=df13_f['std_deviation_Alpha_3'],fmt='g--D',ecolor='g', capthick=3,label='Alpha_3 - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['Alpha_3'], yerr=df19_a['std_deviation_Alpha_3'],fmt='b--o',ecolor='b', capthick=3,label='Alpha_3 - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['Alpha_3'], yerr=df19_b['std_deviation_Alpha_3'],fmt='b--s',ecolor='b', capthick=3,label='Alpha_3 - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['Alpha_3'], yerr=df19_c['std_deviation_Alpha_3'],fmt='b--x',ecolor='b', capthick=3,label='Alpha_3 - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['Alpha_3'], yerr=df19_d['std_deviation_Alpha_3'],fmt='b--*',ecolor='b', capthick=3,label='Alpha_3 - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['Alpha_3'], yerr=df19_e['std_deviation_Alpha_3'],fmt='b--p',ecolor='b', capthick=3,label='Alpha_3 - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['Alpha_3'], yerr=df19_f['std_deviation_Alpha_3'],fmt='b--D',ecolor='b', capthick=3,label='Alpha_3 - Parcela F - FQ19W')

    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Change detector: Beta_1 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Beta_1'], yerr=df_a['std_deviation_Beta_1'],fmt='r--o',ecolor='r', capthick=3,label='Beta_1 - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['Beta_1'], yerr=df_b['std_deviation_Beta_1'],fmt='r--s',ecolor='r', capthick=3,label='Beta_1 - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['Beta_1'], yerr=df_c['std_deviation_Beta_1'],fmt='r--x',ecolor='r', capthick=3,label='Beta_1 - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['Beta_1'], yerr=df_d['std_deviation_Beta_1'],fmt='r--*',ecolor='r', capthick=3,label='Beta_1 - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['Beta_1'], yerr=df_e['std_deviation_Beta_1'],fmt='r--p',ecolor='r', capthick=3,label='Beta_1 - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['Beta_1'], yerr=df_f['std_deviation_Beta_1'],fmt='r--D',ecolor='r', capthick=3,label='Beta_1 - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['Beta_1'], yerr=df13_a['std_deviation_Beta_1'],fmt='g--o',ecolor='g', capthick=3,label='Beta_1 - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['Beta_1'], yerr=df13_b['std_deviation_Beta_1'],fmt='g--s',ecolor='g', capthick=3,label='Beta_1 - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['Beta_1'], yerr=df13_c['std_deviation_Beta_1'],fmt='g--x',ecolor='g', capthick=3,label='Beta_1 - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['Beta_1'], yerr=df13_d['std_deviation_Beta_1'],fmt='g--*',ecolor='g', capthick=3,label='Beta_1 - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['Beta_1'], yerr=df13_e['std_deviation_Beta_1'],fmt='g--p',ecolor='g', capthick=3,label='Beta_1 - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['Beta_1'], yerr=df13_f['std_deviation_Beta_1'],fmt='g--D',ecolor='g', capthick=3,label='Beta_1 - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['Beta_1'], yerr=df19_a['std_deviation_Beta_1'],fmt='b--o',ecolor='b', capthick=3,label='Beta_1 - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['Beta_1'], yerr=df19_b['std_deviation_Beta_1'],fmt='b--s',ecolor='b', capthick=3,label='Beta_1 - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['Beta_1'], yerr=df19_c['std_deviation_Beta_1'],fmt='b--x',ecolor='b', capthick=3,label='Beta_1 - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['Beta_1'], yerr=df19_d['std_deviation_Beta_1'],fmt='b--*',ecolor='b', capthick=3,label='Beta_1 - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['Beta_1'], yerr=df19_e['std_deviation_Beta_1'],fmt='b--p',ecolor='b', capthick=3,label='Beta_1 - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['Beta_1'], yerr=df19_f['std_deviation_Beta_1'],fmt='b--D',ecolor='b', capthick=3,label='Beta_1 - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 90)
    Title='Change detector: Beta_3 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Beta_3'], yerr=df_a['std_deviation_Beta_3'],fmt='r--o',ecolor='r', capthick=3,label='Beta_3 - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['Beta_3'], yerr=df_b['std_deviation_Beta_3'],fmt='r--s',ecolor='r', capthick=3,label='Beta_3 - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['Beta_3'], yerr=df_c['std_deviation_Beta_3'],fmt='r--x',ecolor='r', capthick=3,label='Beta_3 - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['Beta_3'], yerr=df_d['std_deviation_Beta_3'],fmt='r--*',ecolor='r', capthick=3,label='Beta_3 - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['Beta_3'], yerr=df_e['std_deviation_Beta_3'],fmt='r--p',ecolor='r', capthick=3,label='Beta_3 - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['Beta_3'], yerr=df_f['std_deviation_Beta_3'],fmt='r--D',ecolor='r', capthick=3,label='Beta_3 - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['Beta_3'], yerr=df13_a['std_deviation_Beta_3'],fmt='g--o',ecolor='g', capthick=3,label='Beta_3 - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['Beta_3'], yerr=df13_b['std_deviation_Beta_3'],fmt='g--s',ecolor='g', capthick=3,label='Beta_3 - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['Beta_3'], yerr=df13_c['std_deviation_Beta_3'],fmt='g--x',ecolor='g', capthick=3,label='Beta_3 - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['Beta_3'], yerr=df13_d['std_deviation_Beta_3'],fmt='g--*',ecolor='g', capthick=3,label='Beta_3 - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['Beta_3'], yerr=df13_e['std_deviation_Beta_3'],fmt='g--p',ecolor='g', capthick=3,label='Beta_3 - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['Beta_3'], yerr=df13_f['std_deviation_Beta_3'],fmt='g--D',ecolor='g', capthick=3,label='Beta_3 - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['Beta_3'], yerr=df19_a['std_deviation_Beta_3'],fmt='b--o',ecolor='b', capthick=3,label='Beta_3 - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['Beta_3'], yerr=df19_b['std_deviation_Beta_3'],fmt='b--s',ecolor='b', capthick=3,label='Beta_3 - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['Beta_3'], yerr=df19_c['std_deviation_Beta_3'],fmt='b--x',ecolor='b', capthick=3,label='Beta_3 - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['Beta_3'], yerr=df19_d['std_deviation_Beta_3'],fmt='b--*',ecolor='b', capthick=3,label='Beta_3 - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['Beta_3'], yerr=df19_e['std_deviation_Beta_3'],fmt='b--p',ecolor='b', capthick=3,label='Beta_3 - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['Beta_3'], yerr=df19_f['std_deviation_Beta_3'],fmt='b--D',ecolor='b', capthick=3,label='Beta_3 - Parcela F - FQ19W')

    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-1, 1)
    Title='Change detector: Lamda 1 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['L1'], yerr=df_a['std_deviation_Lamda1'],fmt='r--o',ecolor='r', capthick=3,label='L1 - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['L1'], yerr=df_b['std_deviation_Lamda1'],fmt='r--s',ecolor='r', capthick=3,label='L1 - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['L1'], yerr=df_c['std_deviation_Lamda1'],fmt='r--x',ecolor='r', capthick=3,label='L1 - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['L1'], yerr=df_d['std_deviation_Lamda1'],fmt='r--*',ecolor='r', capthick=3,label='L1 - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['L1'], yerr=df_e['std_deviation_Lamda1'],fmt='r--p',ecolor='r', capthick=3,label='L1 - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['L1'], yerr=df_f['std_deviation_Lamda1'],fmt='r--D',ecolor='r', capthick=3,label='L1 - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['L1'], yerr=df13_a['std_deviation_Lamda1'],fmt='g--o',ecolor='g', capthick=3,label='L1 - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['L1'], yerr=df13_b['std_deviation_Lamda1'],fmt='g--s',ecolor='g', capthick=3,label='L1 - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['L1'], yerr=df13_c['std_deviation_Lamda1'],fmt='g--x',ecolor='g', capthick=3,label='L1 - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['L1'], yerr=df13_d['std_deviation_Lamda1'],fmt='g--*',ecolor='g', capthick=3,label='L1 - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['L1'], yerr=df13_e['std_deviation_Lamda1'],fmt='g--p',ecolor='g', capthick=3,label='L1 - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['L1'], yerr=df13_f['std_deviation_Lamda1'],fmt='g--D',ecolor='g', capthick=3,label='L1 - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['L1'], yerr=df19_a['std_deviation_Lamda1'],fmt='b--o',ecolor='b', capthick=3,label='L1 - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['L1'], yerr=df19_b['std_deviation_Lamda1'],fmt='b--s',ecolor='b', capthick=3,label='L1 - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['L1'], yerr=df19_c['std_deviation_Lamda1'],fmt='b--x',ecolor='b', capthick=3,label='L1 - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['L1'], yerr=df19_d['std_deviation_Lamda1'],fmt='b--*',ecolor='b', capthick=3,label='L1 - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['L1'], yerr=df19_e['std_deviation_Lamda1'],fmt='b--p',ecolor='b', capthick=3,label='L1 - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['L1'], yerr=df19_f['std_deviation_Lamda1'],fmt='b--D',ecolor='b', capthick=3,label='L1 - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(-1, 1)
    Title='Change detector: Lamda 3 All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['L3'], yerr=df_a['std_deviation_Lamda3'],fmt='r--o',ecolor='r', capthick=3,label='L3 - Parcela A - FQ8W')
    plt.errorbar(df_b['BBCH'], df_b['L3'], yerr=df_b['std_deviation_Lamda3'],fmt='r--s',ecolor='r', capthick=3,label='L3 - Parcela B - FQ8W')
    plt.errorbar(df_c['BBCH'], df_c['L3'], yerr=df_c['std_deviation_Lamda3'],fmt='r--x',ecolor='r', capthick=3,label='L3 - Parcela C - FQ8W')
    plt.errorbar(df_d['BBCH'], df_d['L3'], yerr=df_d['std_deviation_Lamda3'],fmt='r--*',ecolor='r', capthick=3,label='L3 - Parcela D - FQ8W')
    plt.errorbar(df_e['BBCH'], df_e['L3'], yerr=df_e['std_deviation_Lamda3'],fmt='r--p',ecolor='r', capthick=3,label='L3 - Parcela E - FQ8W')
    plt.errorbar(df_f['BBCH'], df_f['L3'], yerr=df_f['std_deviation_Lamda3'],fmt='r--D',ecolor='r', capthick=3,label='L3 - Parcela F - FQ8W')
    plt.errorbar(df13_a['BBCH'], df13_a['L3'], yerr=df13_a['std_deviation_Lamda3'],fmt='g--o',ecolor='g', capthick=3,label='L3 - Parcela A - FQ13W')
    plt.errorbar(df13_b['BBCH'], df13_b['L3'], yerr=df13_b['std_deviation_Lamda3'],fmt='g--s',ecolor='g', capthick=3,label='L3 - Parcela B - FQ13W')
    plt.errorbar(df13_c['BBCH'], df13_c['L3'], yerr=df13_c['std_deviation_Lamda3'],fmt='g--x',ecolor='g', capthick=3,label='L3 - Parcela C - FQ13W')
    plt.errorbar(df13_d['BBCH'], df13_d['L3'], yerr=df13_d['std_deviation_Lamda3'],fmt='g--*',ecolor='g', capthick=3,label='L3 - Parcela D - FQ13W')
    plt.errorbar(df13_e['BBCH'], df13_e['L3'], yerr=df13_e['std_deviation_Lamda3'],fmt='g--p',ecolor='g', capthick=3,label='L3 - Parcela E - FQ13W')
    plt.errorbar(df13_f['BBCH'], df13_f['L3'], yerr=df13_f['std_deviation_Lamda3'],fmt='g--D',ecolor='g', capthick=3,label='L3 - Parcela F - FQ13W')
    plt.errorbar(df19_a['BBCH'], df19_a['L3'], yerr=df19_a['std_deviation_Lamda3'],fmt='b--o',ecolor='b', capthick=3,label='L3 - Parcela A - FQ19W')
    plt.errorbar(df19_b['BBCH'], df19_b['L3'], yerr=df19_b['std_deviation_Lamda3'],fmt='b--s',ecolor='b', capthick=3,label='L3 - Parcela B - FQ19W')
    plt.errorbar(df19_c['BBCH'], df19_c['L3'], yerr=df19_c['std_deviation_Lamda3'],fmt='b--x',ecolor='b', capthick=3,label='L3 - Parcela C - FQ19W')
    plt.errorbar(df19_d['BBCH'], df19_d['L3'], yerr=df19_d['std_deviation_Lamda3'],fmt='b--*',ecolor='b', capthick=3,label='L3 - Parcela D - FQ19W')
    plt.errorbar(df19_e['BBCH'], df19_e['L3'], yerr=df19_e['std_deviation_Lamda3'],fmt='b--p',ecolor='b', capthick=3,label='L3 - Parcela E - FQ19W')
    plt.errorbar(df19_f['BBCH'], df19_f['L3'], yerr=df19_f['std_deviation_Lamda3'],fmt='b--D',ecolor='b', capthick=3,label='L3 - Parcela F - FQ19W')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
#def p_Alpha(beam,df_a,df_c,df_d,df_e,df_f):
def p_Alpha(beam,df_a,df_b,df_c,df_d,df_e,df_f):
    df_a['P1A1']=((np.abs(df_a['L1']))/((np.abs(df_a['L1']))+(np.abs(df_a['L2']))+(np.abs(df_a['L3']))))*df_a['Alpha_1']
    #df_b['P1A1']=((np.abs(df_b['L1']))/((np.abs(df_b['L1']))+(np.abs(df_b['L2']))+(np.abs(df_b['L3']))))*df_b['Alpha_1']
    df_c['P1A1']=((np.abs(df_c['L1']))/((np.abs(df_c['L1']))+(np.abs(df_c['L2']))+(np.abs(df_c['L3']))))*df_c['Alpha_1']
    df_d['P1A1']=((np.abs(df_d['L1']))/((np.abs(df_d['L1']))+(np.abs(df_d['L2']))+(np.abs(df_d['L3']))))*df_d['Alpha_1']
    df_e['P1A1']=((np.abs(df_e['L1']))/((np.abs(df_e['L1']))+(np.abs(df_e['L2']))+(np.abs(df_e['L3']))))*df_e['Alpha_1']
    df_f['P1A1']=((np.abs(df_f['L1']))/((np.abs(df_f['L1']))+(np.abs(df_f['L2']))+(np.abs(df_f['L3']))))*df_f['Alpha_1']
    
    df_a['P2A2']=((np.abs(df_a['L2']))/((np.abs(df_a['L1']))+(np.abs(df_a['L2']))+(np.abs(df_a['L3']))))*df_a['Alpha_2']
    #df_b['P2A2']=((np.abs(df_b['L2']))/((np.abs(df_b['L1']))+(np.abs(df_b['L2']))+(np.abs(df_b['L3']))))*df_b['Alpha_2']
    df_c['P2A2']=((np.abs(df_c['L2']))/((np.abs(df_c['L1']))+(np.abs(df_c['L2']))+(np.abs(df_c['L3']))))*df_c['Alpha_2']
    df_d['P2A2']=((np.abs(df_d['L2']))/((np.abs(df_d['L1']))+(np.abs(df_d['L2']))+(np.abs(df_d['L3']))))*df_d['Alpha_2']
    df_e['P2A2']=((np.abs(df_e['L2']))/((np.abs(df_e['L1']))+(np.abs(df_e['L2']))+(np.abs(df_e['L3']))))*df_e['Alpha_2']
    df_f['P2A2']=((np.abs(df_f['L2']))/((np.abs(df_f['L1']))+(np.abs(df_f['L2']))+(np.abs(df_f['L3']))))*df_f['Alpha_2']
    
    df_a['P3A3']=((np.abs(df_a['L3']))/((np.abs(df_a['L1']))+(np.abs(df_a['L2']))+(np.abs(df_a['L3']))))*df_a['Alpha_3']
    #df_b['P3A3']=((np.abs(df_b['L3']))/((np.abs(df_b['L1']))+(np.abs(df_b['L2']))+(np.abs(df_b['L3']))))*df_b['Alpha_3']
    df_c['P3A3']=((np.abs(df_c['L3']))/((np.abs(df_c['L1']))+(np.abs(df_c['L2']))+(np.abs(df_c['L3']))))*df_c['Alpha_3']
    df_d['P3A3']=((np.abs(df_d['L3']))/((np.abs(df_d['L1']))+(np.abs(df_d['L2']))+(np.abs(df_d['L3']))))*df_d['Alpha_3']
    df_e['P3A3']=((np.abs(df_e['L3']))/((np.abs(df_e['L1']))+(np.abs(df_e['L2']))+(np.abs(df_e['L3']))))*df_e['Alpha_3']
    df_f['P3A3']=((np.abs(df_f['L3']))/((np.abs(df_f['L1']))+(np.abs(df_f['L2']))+(np.abs(df_f['L3']))))*df_f['Alpha_3']
    
#    fig, ax = plt.subplots(figsize=(19, 9))
#    plt.ylim(0, 90)
#    Title='Change_detector: Pi * Alpha - All parcels - '+beam
#    ax.set_title(Title)
#    plt.errorbar(df_a['BBCH'], df_a['P1A1'], fmt='c--o',ecolor='c', capthick=3,label='P1A1 - Parcela A')
#    #plt.errorbar(df_b['BBCH'], df_b['P1A1'], fmt='c--s',ecolor='c', capthick=3,label='P1A1 - Parcela B')
#    plt.errorbar(df_c['BBCH'], df_c['P1A1'], fmt='c--x',ecolor='c', capthick=3,label='P1A1 - Parcela C')
#    plt.errorbar(df_d['BBCH'], df_d['P1A1'], fmt='c--*',ecolor='c', capthick=3,label='P1A1 - Parcela D')
#    plt.errorbar(df_e['BBCH'], df_e['P1A1'], fmt='c--p',ecolor='c', capthick=3,label='P1A1 - Parcela E')
#    plt.errorbar(df_f['BBCH'], df_f['P1A1'], fmt='c--D',ecolor='c', capthick=3,label='P1A1 - Parcela F')
#    plt.errorbar(df_a['BBCH'], df_a['P2A2'], fmt='k--o',ecolor='k', capthick=3,label='P2A2 - Parcela A')
#    #plt.errorbar(df_b['BBCH'], df_b['P2A2'], fmt='k--s',ecolor='k', capthick=3,label='P2A2 - Parcela B')
#    plt.errorbar(df_c['BBCH'], df_c['P2A2'], fmt='k--x',ecolor='k', capthick=3,label='P2A2 - Parcela C')
#    plt.errorbar(df_d['BBCH'], df_d['P2A2'], fmt='k--*',ecolor='k', capthick=3,label='P2A2 - Parcela D')
#    plt.errorbar(df_e['BBCH'], df_e['P2A2'], fmt='k--p',ecolor='k', capthick=3,label='P2A2 - Parcela E')
#    plt.errorbar(df_f['BBCH'], df_f['P2A2'], fmt='k--d',ecolor='k', capthick=3,label='P2A2 - Parcela F')
#    plt.errorbar(df_a['BBCH'], df_a['P3A3'], fmt='r--o',ecolor='r', capthick=3,label='P3A3 - Parcela A')
#    #plt.errorbar(df_b['BBCH'], df_b['P3A3'], fmt='r--s',ecolor='r', capthick=3,label='P3A3 - Parcela B')
#    plt.errorbar(df_c['BBCH'], df_c['P3A3'], fmt='r--x',ecolor='r', capthick=3,label='P3A3 - Parcela C')
#    plt.errorbar(df_d['BBCH'], df_d['P3A3'], fmt='r--*',ecolor='r', capthick=3,label='P3A3 - Parcela D')
#    plt.errorbar(df_e['BBCH'], df_e['P3A3'], fmt='r--p',ecolor='r', capthick=3,label='P3A3 - Parcela E')
#    plt.errorbar(df_f['BBCH'], df_f['P3A3'], fmt='r--d',ecolor='r', capthick=3,label='P3A3 - Parcela F')
#    plt.legend(loc="upper right")
#    plt.tight_layout()
    
#    df_a['P1B1']=((np.abs(df_a['L1']))/((np.abs(df_a['L1']))+(np.abs(df_a['L2']))+(np.abs(df_a['L3']))))*df_a['Beta_1']
#    df_b['P1B1']=((np.abs(df_b['L1']))/((np.abs(df_b['L1']))+(np.abs(df_b['L2']))+(np.abs(df_b['L3']))))*df_b['Beta_1']
#    df_c['P1B1']=((np.abs(df_c['L1']))/((np.abs(df_c['L1']))+(np.abs(df_c['L2']))+(np.abs(df_c['L3']))))*df_c['Beta_1']
#    df_d['P1B1']=((np.abs(df_d['L1']))/((np.abs(df_d['L1']))+(np.abs(df_d['L2']))+(np.abs(df_d['L3']))))*df_d['Beta_1']
#    df_e['P1B1']=((np.abs(df_e['L1']))/((np.abs(df_e['L1']))+(np.abs(df_e['L2']))+(np.abs(df_e['L3']))))*df_e['Beta_1']
#    df_f['P1B1']=((np.abs(df_f['L1']))/((np.abs(df_f['L1']))+(np.abs(df_f['L2']))+(np.abs(df_f['L3']))))*df_f['Beta_1']
#    
#    df_a['P2B2']=((np.abs(df_a['L2']))/((np.abs(df_a['L1']))+(np.abs(df_a['L2']))+(np.abs(df_a['L3']))))*df_a['Beta_2']
#    df_b['P2B2']=((np.abs(df_b['L2']))/((np.abs(df_b['L1']))+(np.abs(df_b['L2']))+(np.abs(df_b['L3']))))*df_b['Beta_2']
#    df_c['P2B2']=((np.abs(df_c['L2']))/((np.abs(df_c['L1']))+(np.abs(df_c['L2']))+(np.abs(df_c['L3']))))*df_c['Beta_2']
#    df_d['P2B2']=((np.abs(df_d['L2']))/((np.abs(df_d['L1']))+(np.abs(df_d['L2']))+(np.abs(df_d['L3']))))*df_d['Beta_2']
#    df_e['P2B2']=((np.abs(df_e['L2']))/((np.abs(df_e['L1']))+(np.abs(df_e['L2']))+(np.abs(df_e['L3']))))*df_e['Beta_2']
#    df_f['P2B2']=((np.abs(df_f['L2']))/((np.abs(df_f['L1']))+(np.abs(df_f['L2']))+(np.abs(df_f['L3']))))*df_f['Beta_2']
#    
#    df_a['P3B3']=((np.abs(df_a['L3']))/((np.abs(df_a['L1']))+(np.abs(df_a['L2']))+(np.abs(df_a['L3']))))*df_a['Beta_3']
#    df_b['P3B3']=((np.abs(df_b['L3']))/((np.abs(df_b['L1']))+(np.abs(df_b['L2']))+(np.abs(df_b['L3']))))*df_b['Beta_3']
#    df_c['P3B3']=((np.abs(df_c['L3']))/((np.abs(df_c['L1']))+(np.abs(df_c['L2']))+(np.abs(df_c['L3']))))*df_c['Beta_3']
#    df_d['P3B3']=((np.abs(df_d['L3']))/((np.abs(df_d['L1']))+(np.abs(df_d['L2']))+(np.abs(df_d['L3']))))*df_d['Beta_3']
#    df_e['P3B3']=((np.abs(df_e['L3']))/((np.abs(df_e['L1']))+(np.abs(df_e['L2']))+(np.abs(df_e['L3']))))*df_e['Beta_3']
#    df_f['P3B3']=((np.abs(df_f['L3']))/((np.abs(df_f['L1']))+(np.abs(df_f['L2']))+(np.abs(df_f['L3']))))*df_f['Beta_3']
    
#    fig, ax = plt.subplots(figsize=(19, 9))
#    plt.ylim(0, 40)
#    Title='Change_detector: Pi * Beta - All parcels - '+beam
#    ax.set_title(Title)
#    plt.errorbar(df_a['BBCH'], df_a['P1B1'], fmt='c--o',ecolor='c', capthick=3,label='P1B1 - Parcela A')
#    plt.errorbar(df_b['BBCH'], df_b['P1B1'], fmt='c--s',ecolor='c', capthick=3,label='P1B1 - Parcela B')
#    plt.errorbar(df_c['BBCH'], df_c['P1B1'], fmt='c--x',ecolor='c', capthick=3,label='P1B1 - Parcela C')
#    plt.errorbar(df_d['BBCH'], df_d['P1B1'], fmt='c--*',ecolor='c', capthick=3,label='P1B1 - Parcela D')
#    plt.errorbar(df_e['BBCH'], df_e['P1B1'], fmt='c--p',ecolor='c', capthick=3,label='P1B1 - Parcela E')
#    plt.errorbar(df_f['BBCH'], df_f['P1B1'], fmt='c--D',ecolor='c', capthick=3,label='P1B1 - Parcela F')
#    plt.errorbar(df_a['BBCH'], df_a['P2B2'], fmt='k--o',ecolor='k', capthick=3,label='P2B2 - Parcela A')
#    plt.errorbar(df_b['BBCH'], df_b['P2B2'], fmt='k--s',ecolor='k', capthick=3,label='P2B2 - Parcela B')
#    plt.errorbar(df_c['BBCH'], df_c['P2B2'], fmt='k--x',ecolor='k', capthick=3,label='P2B2 - Parcela C')
#    plt.errorbar(df_d['BBCH'], df_d['P2B2'], fmt='k--*',ecolor='k', capthick=3,label='P2B2 - Parcela D')
#    plt.errorbar(df_e['BBCH'], df_e['P2B2'], fmt='k--p',ecolor='k', capthick=3,label='P2B2 - Parcela E')
#    plt.errorbar(df_f['BBCH'], df_f['P2B2'], fmt='k--d',ecolor='k', capthick=3,label='P2B2 - Parcela F')
#    plt.errorbar(df_a['BBCH'], df_a['P3B3'], fmt='r--o',ecolor='r', capthick=3,label='P3B3 - Parcela A')
#    plt.errorbar(df_b['BBCH'], df_b['P3B3'], fmt='r--s',ecolor='r', capthick=3,label='P3B3 - Parcela B')
#    plt.errorbar(df_c['BBCH'], df_c['P3B3'], fmt='r--x',ecolor='r', capthick=3,label='P3B3 - Parcela C')
#    plt.errorbar(df_d['BBCH'], df_d['P3B3'], fmt='r--*',ecolor='r', capthick=3,label='P3B3 - Parcela D')
#    plt.errorbar(df_e['BBCH'], df_e['P3B3'], fmt='r--p',ecolor='r', capthick=3,label='P3B3 - Parcela E')
#    plt.errorbar(df_f['BBCH'], df_f['P3B3'], fmt='r--d',ecolor='r', capthick=3,label='P3B3 - Parcela F')
#    plt.legend(loc="upper right")
#    plt.tight_layout()    
    df=df_a.append(df_c).append(df_d).append(df_e).append(df_f)    
    df=df.fillna(0)
    return df
    
def l_Alpha(beam,df_a,df_b,df_c,df_d,df_e,df_f):

    df_a['L1A1']=np.abs(df_a['L1'])*np.abs(df_a['Alpha_1'])
    df_b['L1A1']=np.abs(df_b['L1'])*np.abs(df_b['Alpha_1'])
    df_c['L1A1']=np.abs(df_c['L1'])*np.abs(df_c['Alpha_1'])
    df_d['L1A1']=np.abs(df_d['L1'])*np.abs(df_d['Alpha_1'])
    df_e['L1A1']=np.abs(df_e['L1'])*np.abs(df_e['Alpha_1'])
    df_f['L1A1']=np.abs(df_f['L1'])*np.abs(df_f['Alpha_1'])

    df_a['L2A2']=np.abs(df_a['L2'])*np.abs(df_a['Alpha_2'])
    df_b['L2A2']=np.abs(df_b['L2'])*np.abs(df_b['Alpha_2'])
    df_c['L2A2']=np.abs(df_c['L2'])*np.abs(df_c['Alpha_2'])
    df_d['L2A2']=np.abs(df_d['L2'])*np.abs(df_d['Alpha_2'])
    df_e['L2A2']=np.abs(df_e['L2'])*np.abs(df_e['Alpha_2'])
    df_f['L2A2']=np.abs(df_f['L2'])*np.abs(df_f['Alpha_2'])
    
    df_a['L3A3']=np.abs(df_a['L3'])*np.abs(df_a['Alpha_3'])
    df_b['L3A3']=np.abs(df_b['L3'])*np.abs(df_b['Alpha_3'])
    df_c['L3A3']=np.abs(df_c['L3'])*np.abs(df_c['Alpha_3'])
    df_d['L3A3']=np.abs(df_d['L3'])*np.abs(df_d['Alpha_3'])
    df_e['L3A3']=np.abs(df_e['L3'])*np.abs(df_e['Alpha_3'])
    df_f['L3A3']=np.abs(df_f['L3'])*np.abs(df_f['Alpha_3'])
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(0, 50)
    Title='Change_detector: Li * Alpha - All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['L1A1'], fmt='c--o',ecolor='c', capthick=3,label='L1A1 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['L1A1'], fmt='c--s',ecolor='c', capthick=3,label='L1A1 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['L1A1'], fmt='c--x',ecolor='c', capthick=3,label='L1A1 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['L1A1'], fmt='c--*',ecolor='c', capthick=3,label='L1A1 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['L1A1'], fmt='c--p',ecolor='c', capthick=3,label='L1A1 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['L1A1'], fmt='c--D',ecolor='c', capthick=3,label='L1A1 - Parcela F')
    plt.errorbar(df_a['BBCH'], df_a['L2A2'], fmt='k--o',ecolor='k', capthick=3,label='L2A2 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['L2A2'], fmt='k--s',ecolor='k', capthick=3,label='L2A2 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['L2A2'], fmt='k--x',ecolor='k', capthick=3,label='L2A2 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['L2A2'], fmt='k--*',ecolor='k', capthick=3,label='L2A2 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['L2A2'], fmt='k--p',ecolor='k', capthick=3,label='L2A2 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['L2A2'], fmt='k--d',ecolor='k', capthick=3,label='L2A2 - Parcela F')
    plt.errorbar(df_a['BBCH'], df_a['L3A3'], fmt='r--o',ecolor='r', capthick=3,label='L3A3 - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['L3A3'], fmt='r--s',ecolor='r', capthick=3,label='L3A3 - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['L3A3'], fmt='r--x',ecolor='r', capthick=3,label='L3A3 - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['L3A3'], fmt='r--*',ecolor='r', capthick=3,label='L3A3 - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['L3A3'], fmt='r--p',ecolor='r', capthick=3,label='L3A3 - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['L3A3'], fmt='r--d',ecolor='r', capthick=3,label='L3A3 - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    df=df_a.append(df_c).append(df_d).append(df_e).append(df_f)    
    df=df.fillna(0)
    return df

def Alpha_prima(beam,df_a,df_b,df_c,df_d,df_e,df_f):

    df_a['Alpha1_prima']=(np.arccos((np.cos(df_a['Alpha_1'])*df_a['L1'])))*180/np.pi
    df_b['Alpha1_prima']=(np.arccos((np.cos(df_b['Alpha_1'])*df_b['L1'])))*180/np.pi
    df_c['Alpha1_prima']=(np.arccos((np.cos(df_c['Alpha_1'])*df_c['L1'])))*180/np.pi
    df_d['Alpha1_prima']=(np.arccos((np.cos(df_d['Alpha_1'])*df_d['L1'])))*180/np.pi
    df_e['Alpha1_prima']=(np.arccos((np.cos(df_e['Alpha_1'])*df_e['L1'])))*180/np.pi
    df_f['Alpha1_prima']=(np.arccos((np.cos(df_f['Alpha_1'])*df_f['L1'])))*180/np.pi

    df_a['Alpha2_prima']=(np.arccos((np.cos(df_a['Alpha_2'])*df_a['L2'])))*180/np.pi
    df_b['Alpha2_prima']=(np.arccos((np.cos(df_b['Alpha_2'])*df_b['L2'])))*180/np.pi
    df_c['Alpha2_prima']=(np.arccos((np.cos(df_c['Alpha_2'])*df_c['L2'])))*180/np.pi
    df_d['Alpha2_prima']=(np.arccos((np.cos(df_d['Alpha_2'])*df_d['L2'])))*180/np.pi
    df_e['Alpha2_prima']=(np.arccos((np.cos(df_e['Alpha_2'])*df_e['L2'])))*180/np.pi
    df_f['Alpha2_prima']=(np.arccos((np.cos(df_f['Alpha_2'])*df_f['L2'])))*180/np.pi
    
    df_a['Alpha3_prima']=(np.arccos((np.cos(df_a['Alpha_3'])*df_a['L3'])))*180/np.pi
    df_b['Alpha3_prima']=(np.arccos((np.cos(df_b['Alpha_3'])*df_b['L3'])))*180/np.pi
    df_c['Alpha3_prima']=(np.arccos((np.cos(df_c['Alpha_3'])*df_c['L3'])))*180/np.pi
    df_d['Alpha3_prima']=(np.arccos((np.cos(df_d['Alpha_3'])*df_d['L3'])))*180/np.pi
    df_e['Alpha3_prima']=(np.arccos((np.cos(df_e['Alpha_3'])*df_e['L3'])))*180/np.pi
    df_f['Alpha3_prima']=(np.arccos((np.cos(df_f['Alpha_3'])*df_f['L3'])))*180/np.pi

   
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.ylim(50, 140)
    Title='Change_detector: Alpha_1 prima - All parcels - '+beam
    ax.set_title(Title)
    plt.errorbar(df_a['BBCH'], df_a['Alpha1_prima'], fmt='c--o',ecolor='c', capthick=3,label='Alpha1_prima - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Alpha1_prima'], fmt='c--s',ecolor='c', capthick=3,label='Alpha1_prima - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Alpha1_prima'], fmt='c--x',ecolor='c', capthick=3,label='Alpha1_prima - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Alpha1_prima'], fmt='c--*',ecolor='c', capthick=3,label='Alpha1_prima - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Alpha1_prima'], fmt='c--p',ecolor='c', capthick=3,label='Alpha1_prima - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Alpha1_prima'], fmt='c--D',ecolor='c', capthick=3,label='Alpha1_prima - Parcela F')
    plt.errorbar(df_a['BBCH'], df_a['Alpha2_prima'], fmt='k--o',ecolor='k', capthick=3,label='Alpha2_prima - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Alpha2_prima'], fmt='k--s',ecolor='k', capthick=3,label='Alpha2_prima - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Alpha2_prima'], fmt='k--x',ecolor='k', capthick=3,label='Alpha2_prima - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Alpha2_prima'], fmt='k--*',ecolor='k', capthick=3,label='Alpha2_prima - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Alpha2_prima'], fmt='k--p',ecolor='k', capthick=3,label='Alpha2_prima - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Alpha2_prima'], fmt='k--d',ecolor='k', capthick=3,label='Alpha2_prima - Parcela F')
    plt.errorbar(df_a['BBCH'], df_a['Alpha3_prima'], fmt='r--o',ecolor='r', capthick=3,label='Alpha3_prima - Parcela A')
    plt.errorbar(df_b['BBCH'], df_b['Alpha3_prima'], fmt='r--s',ecolor='r', capthick=3,label='Alpha3_prima - Parcela B')
    plt.errorbar(df_c['BBCH'], df_c['Alpha3_prima'], fmt='r--x',ecolor='r', capthick=3,label='Alpha3_prima - Parcela C')
    plt.errorbar(df_d['BBCH'], df_d['Alpha3_prima'], fmt='r--*',ecolor='r', capthick=3,label='Alpha3_prima - Parcela D')
    plt.errorbar(df_e['BBCH'], df_e['Alpha3_prima'], fmt='r--p',ecolor='r', capthick=3,label='Alpha3_prima - Parcela E')
    plt.errorbar(df_f['BBCH'], df_f['Alpha3_prima'], fmt='r--d',ecolor='r', capthick=3,label='Alpha3_prima - Parcela F')
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    df=df_a.append(df_c).append(df_d).append(df_e).append(df_f)    
    df=df.fillna(0)
    return df

def append_phenology(beam,df_a,df_c,df_d,df_e,df_f):
#def append_phenology(beam,df_a,df_b,df_c,df_d,df_e,df_f):
    df_a.set_index('Date', inplace=True)
    #df_b.set_index('Date', inplace=True)
    df_c.set_index('Date', inplace=True)
    df_d.set_index('Date', inplace=True)
    df_e.set_index('Date', inplace=True)
    df_f.set_index('Date', inplace=True)
    Phenology = pd.read_excel('D:\Juanma\Fenologia de las 5 parcelas.xlsx', sheet_name='BBCH-'+beam)
    # Add classes (phenological stages)
    df_a['Stage']=Phenology['Stage Parcel A']
   # df_b['Stage']=Phenology['Stage Parcel B']
    df_c['Stage']=Phenology['Stage Parcel C']
    df_d['Stage']=Phenology['Stage Parcel D']
    df_e['Stage']=Phenology['Stage Parcel E']
    df_f['Stage']=Phenology['Stage Parcel F']
    
    df_a['BBCH']=Phenology['BBCH Parcel A']
    #df_b['BBCH']=Phenology['BBCH Parcel B']
    df_c['BBCH']=Phenology['BBCH Parcel C']
    df_d['BBCH']=Phenology['BBCH Parcel D']
    df_e['BBCH']=Phenology['BBCH Parcel E']
    df_f['BBCH']=Phenology['BBCH Parcel F']
    #df=df_a.append(df_b).append(df_c).append(df_d).append(df_e).append(df_f)    
    df=df_a.append(df_c).append(df_d).append(df_e).append(df_f)    
    
    return df

def phenology_classification(beam,df):
    # Split-out the dataset with features and classes
    dataset = df[['Alpha_1','Entropy','Anisotropy','Stage']] 
    # scatter plot matrix
    #scatter_matrix(dataset)
    #plt.show()
    global array
    array = dataset.values
    numfeat = len(array[0])
    #Number 3 indicates No. of columns of features available
    X = array[:,0:(numfeat-1)]
    Y = array[:,(numfeat-1)]
    validation_size = 0.20
    seed = 7
    #Function to slipt the dataset into the train and tests parts
    #X_train=features para entrenar el modelo
    #X_validation=features para probar el modelo
    #Y_train=Clases para entrenar el modelo
    #Y_train=Clases para probar el modelo
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # Test options and evaluation metric
    scoring = 'accuracy'
    #Let’s evaluate 6 different algorithms:
    #
    #Logistic Regression (LR)
    #Linear Discriminant Analysis (LDA)
    #K-Nearest Neighbors (KNN).
    #Classification and Regression Trees (CART).
    #Gaussian Naive Bayes (NB).
    #Support Vector Machines (SVM).
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    global results
    global names
    results = []
    results1 = []
    results_dev = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results1.append(cv_results)
        results.append(cv_results.mean())
        results_dev.append(cv_results.std())
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

     #boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison - Phenology classification '+beam)
    ax = fig.add_subplot(111)
    plt.boxplot(results1)
    ax.set_xticklabels(names)
    plt.show()

    CART_clf = tree.DecisionTreeClassifier()
    CART_clf.fit(X_train, Y_train)
    
    ## non-coloured version
    dot_data = tree.export_graphviz(CART_clf, out_file=None)
    graph = graphviz.Source(dot_data) 
    graph.render("dataset")     
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    #
    dot_data = tree.export_graphviz(CART_clf, out_file=None,
                                    feature_names=['Alpha_1','Entropy','Anisotropy'],
                                    class_names=['Early vegetative','Tillering','Stem Elongation','Booting','Maturation'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
#    graph.write_pdf('D:\\Juanma\\FQ8W\\Tree_Report_FQ8W.pdf')
    graph.write_pdf('D:\\Juanma\\'+beam+'\\Tree_Report_3feat.pdf')
    #print('Decision tree report saved in ','D:\\Juanma\\'+beam+'\\Tree_Report_3feat.pdf')
    return X_train,Y_train,X_validation,Y_validation,results,results_dev,names

def CD_classification(beam,df):
    # Split-out the dataset with features and classes
    dataset = df[['L1','L3','Alpha_1','Alpha_3','Stage']] 
    # scatter plot matrix
    #scatter_matrix(dataset)
    #plt.show()
    global array
    array = dataset.values
    numfeat = len(array[0])
    #Number 3 indicates No. of columns of features available
    X = array[:,0:(numfeat-1)]
    Y = array[:,(numfeat-1)]
    validation_size = 0.20
    seed = 7
    #Function to slipt the dataset into the train and tests parts
    #X_train=features para entrenar el modelo
    #X_validation=features para probar el modelo
    #Y_train=Clases para entrenar el modelo
    #Y_train=Clases para probar el modelo
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # Test options and evaluation metric
    scoring = 'accuracy'
    #Let’s evaluate 6 different algorithms:
    #
    #Logistic Regression (LR)
    #Linear Discriminant Analysis (LDA)
    #K-Nearest Neighbors (KNN).
    #Classification and Regression Trees (CART).
    #Gaussian Naive Bayes (NB).
    #Support Vector Machines (SVM).
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    global results
    global names
    results = []
    results1 = []
    results_dev = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results1.append(cv_results)
        results.append(cv_results.mean())
        results_dev.append(cv_results.std())
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

     #boxplot algorithm comparison
#    fig = plt.figure()
#    fig.suptitle('Algorithm Comparison - Change detector classification '+beam)
#    ax = fig.add_subplot(111)
#    plt.boxplot(results1)
#    ax.set_xticklabels(names)
#    plt.show()

    CART_clf = tree.DecisionTreeClassifier()
    CART_clf.fit(X_train, Y_train)
    
    ## non-coloured version
    dot_data = tree.export_graphviz(CART_clf, out_file=None)
    graph = graphviz.Source(dot_data) 
    graph.render("dataset")     
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    #
    dot_data = tree.export_graphviz(CART_clf, out_file=None,
                                    feature_names=['L1','L3','Alpha_1','Alpha_3'],
                                    class_names=['Tillering','Stem Elongation','Booting','Maturation'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
#    graph.write_pdf('D:\\Juanma\\FQ8W\\Tree_Report_FQ8W.pdf')
    graph.write_pdf('D:\\Juanma\\'+beam+'\\Tree_Report_4feat_CD.pdf')
    #print('Decision tree report saved in ','D:\\Juanma\\'+beam+'\\Tree_Report_3feat_CD.pdf')
    return X_train,Y_train,X_validation,Y_validation,results,results_dev,names

def PH_and_CD_classification(beam,df):
    # Split-out the dataset with features and classes
    dataset = df[['L1_CD','L3_CD','Alpha_1_CD','Entropy','Stage']] 
    #print(dataset)
    #scatter plot matrix
    #scatter_matrix(dataset)
    #plt.show()
    # class distribution
    print(dataset.groupby('Stage').size())
    #dataset.groupby('Stage').hist()
    #dataset.groupby('Stage').boxplot()
    #dataset.groupby('Stage').plas.hist(alpha=0.4)
    global array
    array = dataset.values
    numfeat = len(array[0])
    #Number 3 indicates No. of columns of features available
    X = array[:,0:(numfeat-1)]
    Y = array[:,(numfeat-1)]
    validation_size = 0.20
    seed = 7
    #Function to slipt the dataset into the train and tests parts
    #X_train=features para entrenar el modelo
    #X_validation=features para probar el modelo
    #Y_train=Clases para entrenar el modelo
    #Y_train=Clases para probar el modelo
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    # Test options and evaluation metric
    scoring = 'accuracy'
    #Let’s evaluate 6 different algorithms:
    #
    #Logistic Regression (LR)
    #Linear Discriminant Analysis (LDA)
    #K-Nearest Neighbors (KNN).
    #Classification and Regression Trees (CART).
    #Gaussian Naive Bayes (NB).
    #Support Vector Machines (SVM).
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate each model in turn
    global results
    global names
    results = []
    results1 = []
    results_dev = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results1.append(cv_results)
        results.append(cv_results.mean())
        results_dev.append(cv_results.std())
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

     #boxplot algorithm comparison
#    fig = plt.figure()
#    fig.suptitle('Algorithm Comparison - classification PH+CD '+beam)
#    ax = fig.add_subplot(111)
#    plt.boxplot(results1)
#    ax.set_xticklabels(names)
#    plt.show()

    CART_clf = tree.DecisionTreeClassifier()
    CART_clf.fit(X_train, Y_train)
    
    ## non-coloured version
    dot_data = tree.export_graphviz(CART_clf, out_file=None)
    graph = graphviz.Source(dot_data) 
    graph.render("dataset")     
    graph = pydotplus.graph_from_dot_data(dot_data)
    
    #
    dot_data = tree.export_graphviz(CART_clf, out_file=None,
                                    feature_names=['L1_CD','L3_CD','Alpha_1_CD','Entropy'],
                                    class_names=['Tillering','Stem Elongation','Booting','Maturation'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
#    graph.write_pdf('D:\\Juanma\\FQ8W\\Tree_Report_FQ8W.pdf')
    graph.write_pdf('D:\\Juanma\\'+beam+'\\Tree_Report_3feat_CD_PH.pdf')
    #print('Decision tree report saved in ','D:\\Juanma\\'+beam+'\\Tree_Report_3feat_CD_PH.pdf')
    return X_train,Y_train,X_validation,Y_validation,results,results_dev,names


def make_decision_tree_and_knn_predictions(X_train,Y_train,X_validation,Y_validation):
    global accuracy_of_prediction_knn
    global confusion_matrix_knn
    global classification_report_knn
    global accuracy_of_prediction_tree
    global confusion_matrix_tree
    global classification_report_tree
    global accuracy_of_prediction_NB
    global confusion_matrix_NB
    global classification_report_NB
    global accuracy_of_prediction_LDA
    global confusion_matrix_LDA
    global classification_report_LDA
#    confusion_matrix_knn = []
    CART_clf = tree.DecisionTreeClassifier()
    CART_clf.fit(X_train, Y_train)
    predictions = CART_clf.predict(X_validation)
    accuracy_of_prediction_tree=accuracy_score(Y_validation, predictions)
    print('Decision Tree accuracy : ',accuracy_of_prediction_tree)
    print('Y_validation : ',Y_validation)
    print(confusion_matrix(Y_validation, predictions))
    #print(classification_report(Y_validation, predictions))

    # Make predictions on validation dataset with knn
    knn = KNeighborsClassifier()
    knn.fit(X_train, Y_train)
    predictions = knn.predict(X_validation)
    accuracy_of_prediction_knn=accuracy_score(Y_validation, predictions)
    confusion_matrix_knn=confusion_matrix(Y_validation, predictions)
    classification_report_knn=classification_report(Y_validation, predictions)
    print('Knn accuracy : ',accuracy_of_prediction_knn)
    print(confusion_matrix_knn)
    #print(classification_report_knn)
    
    # Make predictions on validation dataset with NB
    NB = GaussianNB()
    NB.fit(X_train, Y_train)
    predictions = NB.predict(X_validation)
    accuracy_of_prediction_NB=accuracy_score(Y_validation, predictions)
    confusion_matrix_NB=confusion_matrix(Y_validation, predictions)
    classification_report_NB=classification_report(Y_validation, predictions)
    print('NB accuracy : ',accuracy_of_prediction_NB)
    print(confusion_matrix_NB)
    #print(classification_report_NB)

    # Make predictions on validation dataset with NB
    LDA = LinearDiscriminantAnalysis()
    LDA.fit(X_train, Y_train)
    predictions = NB.predict(X_validation)
    accuracy_of_prediction_LDA=accuracy_score(Y_validation, predictions)
    confusion_matrix_LDA=confusion_matrix(Y_validation, predictions)
    classification_report_LDA=classification_report(Y_validation, predictions)
    print('LDA accuracy : ',accuracy_of_prediction_LDA)
    print(confusion_matrix_LDA)
    #print(classification_report_LDA)
    
    return accuracy_of_prediction_tree,accuracy_of_prediction_knn,accuracy_of_prediction_NB,accuracy_of_prediction_LDA

def plot_features_by_class(beam,df):
    groups = df.groupby('Stage')
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.xlim(-0.2, 1)
    plt.ylim(0, 1)
    Title='Correlation between Lamda 1 CD and Entropy - '+beam
    ax.set_title(Title)
    for name, group in groups:
        ax.plot(group.L1_CD, group.Entropy, marker='o', linestyle='', ms=12, label=name)
    plt.xlabel('Lamda_1')
    plt.ylabel('Entropy')
    plt.legend(loc="upper right")
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.xlim(-0.20, 1)
    plt.ylim(0, 90)
    Title='Correlation between Lamda 1 CD and Alpha_1 CD - '+beam
    ax.set_title(Title)
    for name, group in groups:
        ax.plot(group.L1_CD, group.Alpha_1_CD, marker='o', linestyle='', ms=12, label=name)
    plt.xlabel('Lamda_1')
    plt.ylabel('Alpha_1_CD')
    plt.legend(loc="upper right")
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.xlim(0, 1)
    plt.ylim(0, 90)
    Title='Correlation between Entropy and Alpha_1 CD - '+beam
    ax.set_title(Title)
    for name, group in groups:
        ax.plot(group.Entropy, group.Alpha_1_CD, marker='o', linestyle='', ms=12, label=name)
    plt.xlabel('Entropy')
    plt.ylabel('Alpha_1_CD')
    plt.legend(loc="upper right")
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.xlim(-0.2, 1)
    plt.ylim(-1, 0.2)
    Title='Correlation between Lamda_1 and Lamda_3 - '+beam
    ax.set_title(Title)
    for name, group in groups:
        ax.plot(group.L1_CD, group.L3_CD, marker='o', linestyle='', ms=12, label=name)
    plt.xlabel('Lamda_1')
    plt.ylabel('Lamda_3')
    plt.legend(loc="upper right")
    plt.tight_layout() 
    plt.show()
    
def plot_features_by_class_Cloude_Potier(beam,df):
    groups = df.groupby('Stage')
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.xlim(0, 1)
    plt.ylim(0, 90)
    Title='Cloude Potier decomp: Correlation Entorpy and Dominant Alpha - '+beam
    ax.set_title(Title)
    for name, group in groups:
        ax.plot(group.Entropy, group.Alpha_1, marker='o', linestyle='', ms=12, label=name)
    plt.xlabel('Entropy')
    plt.ylabel('Dominant Alpha')
    plt.legend(loc="upper right")
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.xlim(0, 1)
    plt.ylim(0, 90)
    Title='Cloude Potier decomp: Correlation between Anisotropy and Alpha_1 - '+beam
    ax.set_title(Title)
    for name, group in groups:
        ax.plot(group.Anisotropy, group.Alpha_1, marker='o', linestyle='', ms=12, label=name)
    plt.xlabel('Anisotropy')
    plt.ylabel('Alpha_1')
    plt.legend(loc="upper right")
    plt.tight_layout() 
    plt.show()
    
    fig, ax = plt.subplots(figsize=(19, 9))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    Title='Cloude Potier decomp: Correlation between Entropy and Anisotropy - '+beam
    ax.set_title(Title)
    for name, group in groups:
        ax.plot(group.Entropy, group.Anisotropy, marker='o', linestyle='', ms=12, label=name)
    plt.xlabel('Entropy')
    plt.ylabel('Anisotropy')
    plt.legend(loc="upper right")
    plt.tight_layout() 
    plt.show()
    
def Change_detector_diff(C11,C12,C13,C22,C23,C33,
                         C11_I2,C12_I2,C13_I2,C22_I2,C23_I2,C33_I2,
                         path1,Img_size_rows,Img_size_columns):    
    ############################## Create empty arrays #############################################################################################  
    C = np.zeros([3,3],dtype=np.complex64)
    C_I2 = np.zeros([3,3],dtype=np.complex64)
    U=np.matrix([[1, 0, 1], [1, 0, -1],[0, math.sqrt(2), 0]])
    U=(1/math.sqrt(2))* np.matrix(U)     #U matrix to transform Covariance to coherency matrix
    #   Uinv=np.array([[0.707107, 0.707107, 0], [0, 0, 1],[0, -0.707107, 0]])
    Uinv=U.getH()                #U inverse matrix to transform Covariance to coherency matrix
    T = np.zeros([3,3],dtype=np.complex64) 
    T_I2 = np.zeros([3,3],dtype=np.complex64)
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
    ############ Change detector #############################################################################################
    for i in range(Img_size_rows):
        operations=Img_size_rows - i
        operations=100-(operations*100/Img_size_rows)
        #print ("Change detector at position [",r,',',u_1,']',int(round(operations)),"% completed") #u+1       
        for j in range(Img_size_columns):
    
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
            # Optimisation for Normalised Difference
            Tc = (T_I2 - T)/(np.trace(T)+np.trace(T_I2))
            Tc = np.nan_to_num(Tc)
            eigenvalues, eigenvectors = np.linalg.eigh(Tc)
            
            ind = np.argsort(eigenvalues)# organize eigenvectors from higher to lower value
                        
            L1 = eigenvalues[ind[2]]
            L2 = eigenvalues[ind[1]]
            L3 = eigenvalues[ind[0]]
            
    # Only Positive eigenvalues from L1
            if L1>0:
                L1a[i][j]=L1
            else: 
                L1a[i][j]=0    
    # Only Negative eigenvalues from L1
            if L1<0:
                L1r[i][j]=L1
            else: 
                L1r[i][j]=0 
    # Only Positive eigenvalues from L2
            if L2>0:
                L2a[i][j]=L2
            else: 
                L2a[i][j]=0    
    # Only Negative eigenvalues from L2
            if L2<0:
                L2r[i][j]=L2
            else: 
                L2r[i][j]=0 
     # Only Positive eigenvalues from L3
            if L3>0:
                L3a[i][j]=L3
            else: 
                L3a[i][j]=0    
    # Only Negative eigenvalues from L3
            if L3<0:
                L3r[i][j]=L3
            else: 
                L3r[i][j]=0                                      
                
            U_11[i][j]=np.abs(eigenvectors[0,2]) 
            U_21[i][j]=np.abs(eigenvectors[1,2])
            U_31[i][j]=np.abs(eigenvectors[2,2])
     
            U_12[i][j]=np.abs(eigenvectors[0,1])
            U_22[i][j]=np.abs(eigenvectors[1,1])
            U_32[i][j]=np.abs(eigenvectors[2,1])
            
            U_13[i][j]=np.abs(eigenvectors[0,0])
            U_23[i][j]=np.abs(eigenvectors[1,0])
            U_33[i][j]=np.abs(eigenvectors[2,0])     
            
    return(U_13, U_23, U_33,U_12, U_22, U_32,U_11, U_21, U_31,L1a,L2a,L3a,L1r,L2r,L3r)
    
def Change_detector_ratio(C11,C12,C13,C22,C23,C33,
                         C11_I2,C12_I2,C13_I2,C22_I2,C23_I2,C33_I2,
                         path1,Img_size_rows,Img_size_columns):    
    ############################## Create empty arrays #############################################################################################  
    C = np.zeros([3,3],dtype=np.complex64)
    C_I2 = np.zeros([3,3],dtype=np.complex64)
    U=np.matrix([[1, 0, 1], [1, 0, -1],[0, math.sqrt(2), 0]])
    U=(1/math.sqrt(2))* np.matrix(U)     #U matrix to transform Covariance to coherency matrix
    #   Uinv=np.array([[0.707107, 0.707107, 0], [0, 0, 1],[0, -0.707107, 0]])
    Uinv=U.getH()                #U inverse matrix to transform Covariance to coherency matrix
    T = np.zeros([3,3],dtype=np.complex64) 
    T_I2 = np.zeros([3,3],dtype=np.complex64)
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
    ############ Change detector #############################################################################################
    for i in range(Img_size_rows):
        operations=Img_size_rows - i
        operations=100-(operations*100/Img_size_rows)
        #print ("Change detector at position [",r,',',u_1,']',int(round(operations)),"% completed") #u+1       
        for j in range(Img_size_columns):
    
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
            # Optimisation for Normalised Difference
            #Tc = (T_I2 - T)/(np.trace(T)+np.trace(T_I2))
            #Tc = np.nan_to_num(Tc)
            invT11 = ln.inv(T)
            A = np.dot(T_I2, invT11)
#            eigenvalues, eigenvectors = np.linalg.eigh(A)
            A = np.nan_to_num(A)
            eigenvalues, eigenvectors = np.linalg.eig(A)
#            [d1, v1] = ln.eigh(T)
#            [d2, v2] = ln.eig(A)
            
            ind = np.argsort(eigenvalues)# organize eigenvectors from higher to lower value
                        
            L1 = eigenvalues[ind[2]]
            L2 = eigenvalues[ind[1]]
            L3 = eigenvalues[ind[0]]
            
    # Only Positive eigenvalues from L1
            if L1>1:
                L1a[i][j]=L1
            else: 
                L1a[i][j]=0    
    # Only Negative eigenvalues from L1
            if L1<1:
                L1r[i][j]=L1
            else: 
                L1r[i][j]=0 
    # Only Positive eigenvalues from L2
            if L2>1:
                L2a[i][j]=L2
            else: 
                L2a[i][j]=0    
    # Only Negative eigenvalues from L2
            if L2<1:
                L2r[i][j]=L2
            else: 
                L2r[i][j]=0 
     # Only Positive eigenvalues from L3
            if L3>1:
                L3a[i][j]=L3
            else: 
                L3a[i][j]=0    
    # Only Negative eigenvalues from L3
            if L3<1:
                L3r[i][j]=L3
            else: 
                L3r[i][j]=0                                      
                
            U_11[i][j]=np.abs(eigenvectors[0,2]) 
            U_21[i][j]=np.abs(eigenvectors[1,2])
            U_31[i][j]=np.abs(eigenvectors[2,2])
     
            U_12[i][j]=np.abs(eigenvectors[0,1])
            U_22[i][j]=np.abs(eigenvectors[1,1])
            U_32[i][j]=np.abs(eigenvectors[2,1])
            
            U_13[i][j]=np.abs(eigenvectors[0,0])
            U_23[i][j]=np.abs(eigenvectors[1,0])
            U_33[i][j]=np.abs(eigenvectors[2,0])     
            
    return(U_13, U_23, U_33,U_12, U_22, U_32,U_11, U_21, U_31,L1a,L2a,L3a,L1r,L2r,L3r)
 
    
def Eigendecomposition(C11,C12,C13,C22,C23,C33,Img_size_rows,Img_size_columns):    
    ############################## Create empty arrays #############################################################################################  
        C = np.zeros([3,3],dtype=np.complex64)
        U=np.matrix([[1, 0, 1], [1, 0, -1],[0, math.sqrt(2), 0]])
        U=(1/math.sqrt(2))* np.matrix(U)     #U matrix to transform Covariance to coherency matrix
        #   Uinv=np.array([[0.707107, 0.707107, 0], [0, 0, 1],[0, -0.707107, 0]])
        Uinv=U.getH()                #U inverse matrix to transform Covariance to coherency matrix
        T = np.zeros([3,3],dtype=np.complex64) 
        L1 = np.zeros([Img_size_rows,Img_size_columns])
        L1r = np.zeros([Img_size_rows,Img_size_columns])
        L2 = np.zeros([Img_size_rows,Img_size_columns])
        L2r = np.zeros([Img_size_rows,Img_size_columns])
        L3 = np.zeros([Img_size_rows,Img_size_columns])
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
        ############ Change detector #############################################################################################
        for i in range(Img_size_rows):
            operations=Img_size_rows - i
            operations=100-(operations*100/Img_size_rows)
            #print ("Change detector at position [",r,',',u_1,']',int(round(operations)),"% completed") #u+1       
            for j in range(Img_size_columns):
        
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
                # Optimisation for Normalised Difference
                Tc = T
                Tc = np.nan_to_num(Tc)
                eigenvalues, eigenvectors = np.linalg.eigh(Tc)
                
                ind = np.argsort(eigenvalues)# organize eigenvectors from higher to lower value
                            
                L1[i][j] = eigenvalues[ind[2]]
                L2[i][j] = eigenvalues[ind[1]]
                L3[i][j] = eigenvalues[ind[0]]
                
    #    # Only Positive eigenvalues from L1
    #            if L1>0:
    #                L1a[i][j]=L1
    #            else: 
    #                L1a[i][j]=0    
    #    # Only Negative eigenvalues from L1
    #            if L1<0:
    #                L1r[i][j]=L1
    #            else: 
    #                L1r[i][j]=0 
    #    # Only Positive eigenvalues from L2
    #            if L2>0:
    #                L2a[i][j]=L2
    #            else: 
    #                L2a[i][j]=0    
    #    # Only Negative eigenvalues from L2
    #            if L2<0:
    #                L2r[i][j]=L2
    #            else: 
    #                L2r[i][j]=0 
    #     # Only Positive eigenvalues from L3
    #            if L3>0:
    #                L3a[i][j]=L3
    #            else: 
    #                L3a[i][j]=0    
    #    # Only Negative eigenvalues from L3
    #            if L3<0:
    #                L3r[i][j]=L3
    #            else: 
    #                L3r[i][j]=0                                      
                    
                U_11[i][j]=np.abs(eigenvectors[0,2]) 
                U_21[i][j]=np.abs(eigenvectors[1,2])
                U_31[i][j]=np.abs(eigenvectors[2,2])
         
                U_12[i][j]=np.abs(eigenvectors[0,1])
                U_22[i][j]=np.abs(eigenvectors[1,1])
                U_32[i][j]=np.abs(eigenvectors[2,1])
                
                U_13[i][j]=np.abs(eigenvectors[0,0])
                U_23[i][j]=np.abs(eigenvectors[1,0])
                U_33[i][j]=np.abs(eigenvectors[2,0])     
                
        return(U_13, U_23, U_33,U_12, U_22, U_32,U_11, U_21, U_31,L1,L2,L3)    
############################################# Alpha angle from eigenvector ############################################    
def obtain_alphas(U_11,U_13):
    alpha1=np.arccos(np.abs(U_11))
    alpha1=np.degrees(alpha1)
    alpha3=np.arccos(np.abs(U_13))
    alpha3=np.degrees(alpha3)    
    return(alpha1,alpha3)
############################################# lineal fitting #############################################
def lineal_fit(x,y,title,x_label,y_label,outall,save,show_figs):

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(title)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    #ax.text(1, 50, 'boxed italics text in data coords', style='italic',
    #        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    #ax.text(10, 10, r'an equation: $E=mc^2$', fontsize=45)
    print("r-squared:", r_value**2)
    plt.scatter((x), y, c='green', s=200, alpha=0.5, label='original data')
    plt.plot((x), intercept + slope*(x), 'r--', label='fitted line \n'+
                                                        "r-squared:"+str(round(r_value**2, 3)))
    ax.text(0.05, 55, r'Equation: $y='+str(round(intercept,2))+ ' + '+ str(round(slope,2))+'*x$', fontsize=12,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    plt.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if save == 1:
        outall=outall+title+".png"
        fig.savefig(outall, bbox_inches='tight') 
    plt.tight_layout()
    plt.show()
    if show_figs==0:
        plt.close()
 ############################################ exponential fitting ###########################################
def expo_fit(x,y,title,x_label, y_label,outall,save,show_figs):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(func, x, y)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(title)
    yhat = func(x,popt[0],popt[1],popt[2])                        # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results = ssreg / sstot
    ax.scatter(x,y, c='red', s=200, alpha=0.5,label='fitted line \n'+
                                                        "r-squared:"+str(round(results, 3)));
    plt.plot(x, func(x, *popt), 'r-',label='fitted line \n'+
                                                        "r-squared:"+str(round(results, 3)))
    ax.text(0, 75, r'Exponencial fitting'+str((popt)), fontsize=12,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    #plt.xlim([x0[0]-1, x0[-1] + 1 ])
    ax.legend(loc="upper left")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if save == 1:
        outall=outall+title+".png"
        fig.savefig(outall, bbox_inches='tight') 
    plt.tight_layout()
    plt.show()
    if show_figs==0:
        plt.close()
################################################ Polynomial fitting ###########################################
def poly_fit(x,y,title,x_label, y_label,outall,save,show_figs):
        
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(title)
    z = np.polyfit(x, y, 3)
    f = np.poly1d(z)
    # calculate new x's and y's
    x_new = np.linspace(0, np.amax(x), 200)
    y_new = f(x_new)
        # r-squared
    # fit values, and mean
    yhat = f(x)                         # or [p(z) for z in x]
    ybar = np.sum(y)/len(y)          # or sum(y)/len(y)
    ssreg = np.sum((yhat-ybar)**2)   # or sum([ (yihat - ybar)**2 for yihat in yhat])
    sstot = np.sum((y - ybar)**2)    # or sum([ (yi - ybar)**2 for yi in y])
    results = ssreg / sstot
    ax.scatter(x,y, c='red', s=200, alpha=0.5);
    plt.plot(x,y, x_new, y_new,label='fitted line \n'+
                                                        "r-squared:"+str(round(results, 3)))
    ax.text(0, 55, r'polynomial fitting, degree 3', fontsize=12,bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    #plt.xlim([x0[0]-1, x0[-1] + 1 ])
    ax.legend(loc="upper left")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    #outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\'+' Added SM on time'+'.png'
    if save == 1:
        outall=outall+title+".png"
        fig.savefig(outall, bbox_inches='tight') 
    plt.tight_layout()
    plt.show()
    if show_figs==0:
        plt.close()
################################################ Plot info from change matrix ##################################
def plot_info_change_matrix(x,y0,y1,y2,title,outall,save,show_figs):
        
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_title(title)
    #ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(x, y0, c='red', s=200, alpha=0.5);
    ax.plot(x, y0, 'r--o', label='Double bounce - x0')
    ax.scatter(x, y1, c='green', s=200, alpha=0.5);
    ax.plot(x, y1, 'g--o', label='Volume - x1')
    ax.scatter(x, y2, c='blue', s=200, alpha=0.5);
    ax.plot(x, y2, 'b--o', label='Surface - x2')
    ax.legend(loc="upper left")
    ax.set_xlabel('BBCH')
    ax.set_ylabel('Intensity')
    #outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\'+' Added SM on time'+'.png'
    if save == 1:
        outall=outall+title+".png"
        fig.savefig(outall, bbox_inches='tight') 
    plt.tight_layout()
    plt.show()
    #plt.close()
    if show_figs==0:
        plt.close()
    ################################################### 3D plot from 3 vectors #################################################
def plot_3D_from_vector(x0, x1, x2,title,x_label,y_label,z_label,outall,save,show_figs):    
   
    fig1, ax3 = plt.subplots(figsize=(7, 6))
    ax3 = Axes3D(fig1, elev=-150, azim=110)
    ax3.set_title(title)
    ax3.scatter(x0, x1, x2, c='cyan', s=200, alpha=0.5);
    ax3.plot(x0, x1, x2,'c--o', label='Crop evolution')
    #ax3.legend(loc="upper right")
    ax3.legend()
    ax3.set_xlabel('Double bounce')
    ax3.set_ylabel('Volume')
    ax3.set_zlabel('Surface')
    ax3.invert_xaxis()
    ax3.invert_yaxis()
    ax3.invert_zaxis()
#    ax3.set_xlim(0, np.amax(x0))
#    ax3.set_ylim(0, np.amax(x1))
#    ax3.set_zlim(0, np.amax(x2))
    #outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\'+title+'.png'
    if save == 1:
        outall=outall+title+".png" 
        fig1.savefig(outall, bbox_inches='tight') 
    plt.show()
    #plt.close()
    if show_figs==0:
        plt.close()
    ######################################################### Plot phenology Vs DOY ##################################################
def plot_phenology_and_DoY(DoY, BBCH,outall,save):     
    fig, ax = plt.subplots(figsize=(11, 9))
    title='Phenology evolution (Ground truth)'
    ax.set_title(title)
    #ax = Axes3D(fig, elev=-150, azim=110)
    ax.scatter(DoY, BBCH, c='cyan', s=200, alpha=0.5);
    ax.plot(DoY, BBCH, 'c--o', label='Phenology')
    ax.legend(loc="upper left")
    ax.set_xlabel('DoY')
    ax.set_ylabel('BBCH')
    if save==1:
        fig.savefig(outall, bbox_inches='tight') 
    plt.tight_layout()
    plt.show()
    #plt.close()

def visRGB_L_Contrast_eigedecomposition_test(img1, img2, img3, title,L,x,y,beam,Acquisition_1,Parcela,SM):
    """
    Visualise the RGB of a single acquisition
    """           
    img1 =mask_element(img1,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img2 =mask_element(img2,x,y)
    img3 =mask_element(img3,x,y)
    size = np.shape(img2)           
    iRGB1 = np.zeros([size[0],size[1],3])    
    G1=2*img2
    R1=0.5*(img3-img1) # for components of C matrix. Delete if the components are those of the T matrix
    B1=0.5*(img3+img1)
    K=1
    iRGB1[:,:,0] = (np.abs(R1)/(np.abs(R1).max()))
    iRGB1[:,:,1] = (np.abs(G1)/(np.abs(G1).max()))
    iRGB1[:,:,2] = (np.abs(B1)/(np.abs(B1).max()))  
    
    iRGB1[:,:,0] = np.multiply(iRGB1[:,:,0],L)
    iRGB1[:,:,1] = np.multiply(iRGB1[:,:,1],L)
    iRGB1[:,:,2] = np.multiply(iRGB1[:,:,2],L)   
    
    iRGB1[:,:,0] = iRGB1[:,:,0]*K
    iRGB1[:,:,1] = iRGB1[:,:,1]*K
    iRGB1[:,:,2] = iRGB1[:,:,2]*K 
    iRGB1[np.abs(iRGB1) > 1] = 1

    fig, (ax3) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    #title='W with L and with K'
    ax3.set_title(title)
    ax3.imshow(np.abs(iRGB1))
    plt.axis("off")
    plt.tight_layout()   
    outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\Eigendecomposition '+SM+' '+str(Acquisition_1)+'.png'
    #fig.savefig(outall, bbox_inches='tight') 
    plt.close()    
    return(iRGB1)    
    
def visRGB_L_Contrast_test(img1, img2, img3, img4,img5,img6,img7,img8,img9, title,L1,L2,L3,x,y,beam,Acquisition_1,Acquisition_2,Parcela,SM):
    """
    Visualise the RGB of a single acquisition
    """           
    img1 =mask_element(img1,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img2 =mask_element(img2,x,y)
    img3 =mask_element(img3,x,y)
    img4 =mask_element(img4,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img5 =mask_element(img5,x,y)
    img6 =mask_element(img6,x,y)
    img7 =mask_element(img7,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img8 =mask_element(img8,x,y)
    img9 =mask_element(img9,x,y)
    size = np.shape(img2)           
    
    iRGB1 = np.zeros([size[0],size[1],3])
    B1=0.5*(img2-img3) 
    G1=0.5*(img2+img3)
    R1=2*img1
    K=1
    iRGB1[:,:,0] = (np.abs(G1)/(np.abs(G1).max()))
    iRGB1[:,:,1] = (np.abs(B1)/(np.abs(B1).max()))
    iRGB1[:,:,2] = (np.abs(R1)/(np.abs(R1).max()))  
    
    iRGB1[:,:,0] = np.multiply(iRGB1[:,:,0],L1)
    iRGB1[:,:,1] = np.multiply(iRGB1[:,:,1],L1)
    iRGB1[:,:,2] = np.multiply(iRGB1[:,:,2],L1)   
    
    iRGB1[:,:,0] = iRGB1[:,:,0]*K
    iRGB1[:,:,1] = iRGB1[:,:,1]*K
    iRGB1[:,:,2] = iRGB1[:,:,2]*K 
    iRGB1[np.abs(iRGB1) > 1] = 1
    
    iRGB2 = np.zeros([size[0],size[1],3])
    
    R2=2*img4
    G2=0.5*(img5+img6)
    B2=0.5*(img5-img6)
    
    iRGB2[:,:,0] = (np.abs(G2)/(np.abs(G2).max()))
    iRGB2[:,:,1] = (np.abs(B2)/(np.abs(B2).max()))
    iRGB2[:,:,2] = (np.abs(R2)/(np.abs(R2).max()))
    
    iRGB2[:,:,0] = np.multiply(iRGB2[:,:,0],L2)
    iRGB2[:,:,1] = np.multiply(iRGB2[:,:,1],L2)
    iRGB2[:,:,2] = np.multiply(iRGB2[:,:,2],L2)    
    
    iRGB2[:,:,0] = iRGB2[:,:,0]*K
    iRGB2[:,:,1] = iRGB2[:,:,1]*K
    iRGB2[:,:,2] = iRGB2[:,:,2]*K 
    iRGB2[np.abs(iRGB2) > 1] = 1

    iRGB3 = np.zeros([size[0],size[1],3])
    R3=2*img7
    G3=0.5*(img8+img9)
    B3=0.5*(img8-img9)                                                  
    iRGB3[:,:,0] = (np.abs(G3)/(np.abs(G3).max()))
    iRGB3[:,:,1] = (np.abs(B3)/(np.abs(B3).max()))
    iRGB3[:,:,2] = (np.abs(R3)/(np.abs(R3).max()))
    
    iRGB3[:,:,0] = np.multiply(iRGB3[:,:,0],L3)
    iRGB3[:,:,1] = np.multiply(iRGB3[:,:,1],L3)
    iRGB3[:,:,2] = np.multiply(iRGB3[:,:,2],L3)    
    
    iRGB3[:,:,0] = iRGB3[:,:,0]*K
    iRGB3[:,:,1] = iRGB3[:,:,1]*K
    iRGB3[:,:,2] = iRGB3[:,:,2]*K 
    iRGB3[np.abs(iRGB3) > 1] = 1
    #iRGB=iRGB3
    iRGB=iRGB1+iRGB2+iRGB3
    
    fig, (ax3) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    #title='W with L and with K'
    ax3.set_title(title)
    ax3.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()   
    outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\'+SM+' scattering from '+str(Acquisition_1)+' to '+str(Acquisition_2)+'.png'
    fig.savefig(outall, bbox_inches='tight') 
    plt.close()    
    return(iRGB)   
    
def first_row_column_analysis(change_matrix,No_dates,BBCH,outall,save,show_figs):
    dates=No_dates
    x0=np.zeros([dates])
    x1=np.zeros([dates])
    x2=np.zeros([dates])
    x0r=np.zeros([dates])
    x1r=np.zeros([dates])
    x2r=np.zeros([dates])
    x0d=np.zeros([dates])
    x1d=np.zeros([dates])
    x2d=np.zeros([dates])
    x0_fr=np.zeros([dates-1])
    x1_fr=np.zeros([dates-1])
    x2_fr=np.zeros([dates-1])
    x0[0]=change_matrix[0,0,0]
    x1[0]=change_matrix[0,0,1]
    x2[0]=change_matrix[0,0,2]
    title='Evolution of added Scattering mechanisms (first row)'
    plot_info_change_matrix(BBCH,change_matrix[0,:,0],change_matrix[0,:,1],change_matrix[0,:,2],title,outall,save,show_figs)
    title='Evolution of removed Scattering mechanisms (first column)'
    plot_info_change_matrix(BBCH,change_matrix[:,0,0],change_matrix[:,0,1],change_matrix[:,0,2],title,outall,save,show_figs)
    title='3D Crop evolution (First row-First column)'
    plot_3D_from_vector((change_matrix[0,:,0]+change_matrix[:,0,0]),(change_matrix[0,:,1]+change_matrix[:,0,1]),(change_matrix[0,:,2]+change_matrix[:,0,2]),title,'Double bounce','Volume','Surface',outall,save,show_figs)

    x0r[0]=change_matrix[0,0,0]
    x1r[0]=change_matrix[0,0,1]
    x2r[0]=change_matrix[0,0,2]
    for s in range(BBCH.size-1):
        x0[s+1]=x0[s]+change_matrix[0,s+1,0]
        x1[s+1]=x1[s]+change_matrix[0,s+1,1]
        x2[s+1]=x2[s]+change_matrix[0,s+1,2]
        
        x0r[s+1]=x0r[s]+change_matrix[s+1,0,0]
        x1r[s+1]=x1r[s]+change_matrix[s+1,0,1]
        x2r[s+1]=x2r[s]+change_matrix[s+1,0,2]    
     
    title='Accumulated addition of Scattering mechanisms (first row)'
    plot_info_change_matrix(BBCH,x0,x1,x2,title,outall,save,show_figs)
    title='Accumulated removal of Scattering mechanisms (first column)'
    plot_info_change_matrix(BBCH,x0r,x1r,x2r,title,outall,save,show_figs)
    title='Accumulated addition - removal of SM (first row-first column)'
    plot_info_change_matrix(BBCH,(x0+x0r),(x1+x1r),(x2+x2r),title,outall,save,show_figs)
    ############################################# lineal fitting #############################################
    title='Lineal correlation of addition of 2nd pauli (DB) and phenology (1st row)'
    x_label='Acum. added SM (first row)'
    y_label='BBCH'
    lineal_fit((x0), BBCH, title,x_label, y_label,outall,save,show_figs)
     ############################################ exponential fitting ###########################################
    title='Exponential correlation of addition - removal of 2nd pauli and phenology (1st row - 1st col)'
    x_label='Acum. added SM (first row)'
    y_label='BBCH'
    #lib.expo_fit((x0), BBCH, title,x_label, y_label,outall,save,show_figs) 
    ############################################# polynomial fitting ############################################
    title='Polynomial correlation of addition of 2nd pauli and phenology (1st row - 1st col)'
    x_label='Acum. added SM (first row)'
    y_label='BBCH'
    poly_fit((x0), BBCH, title,x_label, y_label,outall,save,show_figs) 
    ###############################################################################################################
    x0_fr=x0+x0r
    x1_fr=x1+x1r
    x2_fr=x2+x2r
    title='3D Crop evolution (First row-First column accumulated)'
    plot_3D_from_vector(x0_fr,x1_fr,x2_fr,title,'Double bounce','Volume','Surface',outall,save,show_figs)
        
    return(x0_fr,x1_fr,x2_fr)
    
def interpolate_matrix(Source, days):
#    Source=np.array([ [0, 1, 1],
#                      [0, 2, 0],
#                      [0, 3, 1],
#                      [0, 4, 0],
#                      [0, 5, 1]])
    
    x = np.arange(0, Source.shape[0])
    #x = np.arange(0, 15)
    fit = scipy.interpolate.interp1d(x, Source, axis=0)
    Target = fit(np.linspace(0, Source.shape[0]-1, days))
    #Target = fit(np.linspace(0, 15-1, 15))
    #print(Target)
    x = np.arange(0,Target.shape[1])
    #x = np.arange(0, 15)
    fit = scipy.interpolate.interp1d(x, Target, axis=1)
    Target = fit(np.linspace(0, Target.shape[1]-1, days))
    #print(Target)
    return(Target)
    
def visRGB_L_Contrast_log10(img1, img2, img3, img4,img5,img6,img7,img8,img9, title,L1,L2,L3,x,y,beam,Acquisition_1,Acquisition_2,Parcela,SM):
    from scipy.linalg import logm, expm
    img1 =mask_element(img1,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img2 =mask_element(img2,x,y)
    img3 =mask_element(img3,x,y)
    img4 =mask_element(img4,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img5 =mask_element(img5,x,y)
    img6 =mask_element(img6,x,y)
    img7 =mask_element(img7,x,y) #Mask again to put zeros outside the parcel if is not a square parcel
    img8 =mask_element(img8,x,y)
    img9 =mask_element(img9,x,y)
    size = np.shape(img2)      
        
    iRGB1 = np.zeros([size[0],size[1],3])
    iRGB1 = np.zeros([size[0],size[1],3])
    iRGB1 = np.zeros([size[0],size[1],3])
    B1=img1
    R1=img2
    G1=img3
    iRGB1[:,:,0] = (np.abs(R1))
    iRGB1[:,:,1] = (np.abs(G1))
    iRGB1[:,:,2] = (np.abs(B1))  
    
    iRGB1[:,:,0] = np.multiply(iRGB1[:,:,0],L1)
    iRGB1[:,:,1] = np.multiply(iRGB1[:,:,1],L1)
    iRGB1[:,:,2] = np.multiply(iRGB1[:,:,2],L1)   
    
    iRGB1[:,:,0] = np.log10(iRGB1[:,:,0]) 
    iRGB1[:,:,1] = np.log10(iRGB1[:,:,1]) 
    iRGB1[:,:,2] = np.log10(iRGB1[:,:,2]) 
#    
    iRGB1[:,:,0] = np.square(iRGB1[:,:,0]) 
    iRGB1[:,:,1] = np.square(iRGB1[:,:,1]) 
    iRGB1[:,:,2] = np.square(iRGB1[:,:,2]) 
    
    iRGB2 = np.zeros([size[0],size[1],3])
    B2=img4
    R2=img5
    G2=img6
    iRGB2[:,:,0] = (np.abs(R2))
    iRGB2[:,:,1] = (np.abs(G2))
    iRGB2[:,:,2] = (np.abs(B2))
    
    iRGB2[:,:,0] = np.multiply(iRGB2[:,:,0],L2)
    iRGB2[:,:,1] = np.multiply(iRGB2[:,:,1],L2)
    iRGB2[:,:,2] = np.multiply(iRGB2[:,:,2],L2)  
    
    iRGB2[:,:,0] = np.log10(iRGB2[:,:,0]) 
    iRGB2[:,:,1] = np.log10(iRGB2[:,:,1]) 
    iRGB2[:,:,2] = np.log10(iRGB2[:,:,2]) 
#    
    iRGB2[:,:,0] = np.square(iRGB2[:,:,0]) 
    iRGB2[:,:,1] = np.square(iRGB2[:,:,1]) 
    iRGB2[:,:,2] = np.square(iRGB2[:,:,2]) 
    
    iRGB3 = np.zeros([size[0],size[1],3])
    B3=img7
    R3=img8
    G3=img9                                                  
    iRGB3[:,:,0] = (np.abs(R3))
    iRGB3[:,:,1] = (np.abs(G3))
    iRGB3[:,:,2] = (np.abs(B3))
    
    iRGB3[:,:,0] = np.multiply(iRGB3[:,:,0],L3)
    iRGB3[:,:,1] = np.multiply(iRGB3[:,:,1],L3)
    iRGB3[:,:,2] = np.multiply(iRGB3[:,:,2],L3)
    
    iRGB3[:,:,0] = np.log10(iRGB3[:,:,0]) 
    iRGB3[:,:,1] = np.log10(iRGB3[:,:,1]) 
    iRGB3[:,:,2] = np.log10(iRGB3[:,:,2]) 
#    
    iRGB3[:,:,0] = np.square(iRGB3[:,:,0]) 
    iRGB3[:,:,1] = np.square(iRGB3[:,:,1]) 
    iRGB3[:,:,2] = np.square(iRGB3[:,:,2])
    #iRGB=iRGB3
    iRGB=iRGB1+iRGB2+iRGB3
    iRGB=np.sqrt(iRGB)
    K=0.25
    iRGB=K*iRGB
    
    fig, (ax3) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    #title='W with L and with K'
    ax3.set_title(title)
    ax3.imshow(np.abs(iRGB))
    plt.axis("off")
    plt.tight_layout()   
    outall= 'D:\\Juanma\\'+str(beam)+'\\Change detector\\Parcela '+str(Parcela)+'\\'+SM+' scattering from '+str(Acquisition_1)+' to '+str(Acquisition_2)+'.png'
    #fig.savefig(outall, bbox_inches='tight') 
    #plt.close()  
