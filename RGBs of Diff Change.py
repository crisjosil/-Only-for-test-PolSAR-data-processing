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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame, Panel
from skimage import color
from mpl_toolkits.mplot3d import Axes3D
import scipy.linalg as ln

save=0
outall=""
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

def plot_backscatter_log10(img,title,save,outall):
    epsilon=1e-12
    img=img+epsilon
    img=10*np.log10(img)
    fig, (ax11) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))############# Total amount of power that is transmitted
    ax11.set_title(title)
    im11=ax11.imshow(img, cmap = 'gray', vmin=-30, vmax=0)
    ax11.axis('off')
    color_bar(im11)
    plt.tight_layout()
    if save == 1:
        outall= outall+title+'.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    return
def plot_backscatter(img,title,save,outall):
    fig, (ax1) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
    ax1.set_title(title)
    ax1.imshow(np.abs(img), cmap = 'gray', vmin=0, vmax=np.abs(img).mean()*2)
    plt.axis("off")
    plt.tight_layout()
    if save == 1:
        outall= outall+title+'.png'
        fig.savefig(outall, bbox_inches='tight', dpi = 1200)
    return
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

x1=1500
x2=6000
y1=1500
y2=6000
Img_size_columns=x2-x1
Img_size_rows=y2-y1

paths=["D:\\Juanma\\FQ19W\\2014-06-05.rds2\\",
       "D:\\Juanma\\FQ19W\\2014-06-29.rds2\\",
       "D:\\Juanma\\FQ19W\\2014-07-23.rds2\\",
       "D:\\Juanma\\FQ19W\\2014-08-16.rds2\\",
       "D:\\Juanma\\FQ19W\\2014-09-09.rds2\\"]
Tu_Tc_db_stack = np.zeros([Img_size_rows,Img_size_columns,(len(paths)-1)])
for i in range(len(paths)-1):
    print(paths[i])
    path1=paths[i]
    path2=paths[i+1]
############################# Determine new coordinate system based on the ROI #################################################################
#subset the image to the 4 points given by the clicks
#Img_size_rows = 7080
#Img_size_columns = 8000
#path1    = "D:\\Juanma\\FQ19W\\2014-06-05.rds2\\"       
 ################################ Open image i ########################## 
   # Open the covariance matrix elements
    print ("Opening Image "+str(paths[i])+"...")
    C11 = lib.Open_C_diag_element(path1 + nameC11)
    #C12 = lib.Open_C_element(path1 + nameC12R, path1 + nameC12I)
    #C13 = lib.Open_C_element(path1 + nameC13R, path1 + nameC13I)
    C22 = lib.Open_C_diag_element(path1 + nameC22)
    #C23 = lib.Open_C_element(path1 + nameC23R, path1 + nameC23I)
    C33 = lib.Open_C_diag_element(path1 + nameC33)
    ################################ Open image i+1 ##########################    
    #path2    = "D:\\Juanma\\FQ19W\\2014-06-29.rds2\\"
    print ("Opening Image "+str(paths[i+1])+"...")
    C11_I2 = lib.Open_C_diag_element(path2 + nameC11)
    #C12_I2 = lib.Open_C_element(path1 + nameC12R, path1 + nameC12I)
    #C13_I2 = lib.Open_C_element(path1 + nameC13R, path1 + nameC13I)
    C22_I2 = lib.Open_C_diag_element(path2 + nameC22)
    #C23_I2 = lib.Open_C_element(path1 + nameC23R, path1 + nameC23I)
    C33_I2 = lib.Open_C_diag_element(path2 + nameC33)
#x1=5000
#x2=6000
#y1=2900
#y2=4000
    C11=C11[y1:y2,x1:x2] # For seville dataset
    C22=C22[y1:y2,x1:x2]
    C33=C33[y1:y2,x1:x2]     
#C12=C12[y1:y2,x1:x2] 
#C13=C13[y1:y2,x1:x2]
#C23=C23[y1:y2,x1:x2] 

    #Tu_1=0.25*(C11+(2*C22)+C33)
    #title="Tu 1"
    #plot_backscatter(Tu_1,title,save,outall)
    #plot_backscatter_log10(Tu_1,title,save,outall)
    C11_I2=C11_I2[y1:y2,x1:x2] # For seville dataset
    C22_I2=C22_I2[y1:y2,x1:x2]
    C33_I2=C33_I2[y1:y2,x1:x2]     
#C12_I2=C12_I2[y1:y2,x1:x2] # For seville dataset
#C13_I2=C13_I2[y1:y2,x1:x2]
#C23_I2=C23_I2[y1:y2,x1:x2] 
    #Tu_2=0.25*(C11_I2+(2*C22_I2)+C33_I2)
    #title="Tu 2"
    #plot_backscatter(Tu_2,title,save,outall)
    #plot_backscatter_log10(Tu_2,title,save,outall)
    #visRGB(C22, C33, C11, "RGB from [T1]")
    #visRGB(C22_I2, C33_I2, C11_I2, "RGB from [T2]")
    Tc_2=C22_I2-C22
    Tc_3=C33_I2-C33
    Tc_1=C11_I2-C11

    title="Change (RGB Incoherent measure of change)"
    visRGB(Tc_2, Tc_3, Tc_1, title)
    Tu_Tc=0.25*(Tc_1+(2*Tc_2)+Tc_3)
    title="Tu of Change (Incoherent measure of change)"
    plot_backscatter(np.abs(Tu_Tc),title,save,outall)
    title="Tu of Change (Incoherent measure of change) - dB"
    plot_backscatter_log10(np.abs(Tu_Tc),title,save,outall)
    epsilon=1e-12
    Tu_Tc_db=np.abs(Tu_Tc)+epsilon
####################################### Tu of change #########################
    Tu_Tc_db_stack[:,:,i]=10*np.log10(Tu_Tc_db) # Tu of each couple of consecutive acquisitions
    C11=np.zeros([Img_size_rows,Img_size_columns])
    C22=np.zeros([Img_size_rows,Img_size_columns])
    C33=np.zeros([Img_size_rows,Img_size_columns])
    C11_I2=np.zeros([Img_size_rows,Img_size_columns])
    C22_I2=np.zeros([Img_size_rows,Img_size_columns])
    C33_I2=np.zeros([Img_size_rows,Img_size_columns])
    
Tu_Tc_db_total=Tu_Tc_db_stack[:,:,0]+Tu_Tc_db_stack[:,:,1]+Tu_Tc_db_stack[:,:,2]+Tu_Tc_db_stack[:,:,3]

title="Accumulated change (dB)"
fig, (ax11) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
ax11.set_title(title)
im11=ax11.imshow(Tu_Tc_db_total, cmap = 'gray', vmin=-120, vmax=4)
ax11.axis('off')
color_bar(im11)
plt.tight_layout()




######################################### Threshold ###########################
threshold_of_no_change=-110
Tu_no_change = Tu_Tc_db_total*(Tu_Tc_db_total < threshold_of_no_change)# Mask

title="Mask Less change than "+str(threshold_of_no_change)+" dB"
fig, (ax11) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
ax11.set_title(title)
im11=ax11.imshow(Tu_no_change, cmap = 'gray', vmin=-120, vmax=4)
ax11.axis('off')
color_bar(im11)
plt.tight_layout()

Tu_no_change_mask = 1*(Tu_no_change == 0 ) # Mask
Tu_no_change_mask = 1*(Tu_no_change != 0 ) # Mask

title="Mask of no change (Threshold of "+str(threshold_of_no_change)+" dB)"
fig, (ax11) = plt.subplots(1, sharex=True, sharey=True,figsize=(11, 9))
ax11.set_title(title)
im11=ax11.imshow(Tu_no_change_mask, cmap = 'gray', vmin=0, vmax=1)
ax11.axis('off')
color_bar(im11)
plt.tight_layout()
###############################################################################
#ratio?
#Tu_ratio=Tu_2/Tu_1

