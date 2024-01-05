import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from math import pi
from system_setup import *

from matplotlib import colors as pltcolors

KTHdarkturq = '#1C434C'
KTHturq = '#339C9C'
KTHlightturq = '#B2E0E0'

KTHdarkbrick = '#78001A'
KTHbrick = '#E86A58'
KTHlightbrick = '#FFCCC4'

KTHblue = "#004791"
KTHdarkblue = "#000061"
KTHlightblue = "#DEF0FF"
KTHskyblue = "#6298D2"

KTHyellow = "#FFBE00"
KTHdarkyellow = "#A65900"
KTHlightyellow = "#FFF0B0"

KTHgreen = "#4DA061"
KTHdarkgreen = "#0D4A21"
KTHlightgreen = "#C7EBBA"

KTHcolourlist = [KTHdarkturq,KTHturq,KTHdarkbrick,KTHbrick,KTHdarkblue,KTHblue,KTHskyblue,KTHdarkblue,KTHdarkyellow,KTHyellow,KTHdarkgreen,KTHgreen]

KTHcmap = pltcolors.LinearSegmentedColormap.from_list("", ['#FCFCFC','#B2E0E0','#339C9C','#1C434C'])


def linear_diffusion(t,D):
    return 2*D*t

def horizontal_line(t,L):
    return L**2/6.

def nonlinear_diffusion(t,D,L):    
    pisq = pi**2
    result = np.zeros(len(t))
    n = 1.
    
    while True:
        temp = np.divide(np.multiply(-1.*D*(n**2)*pisq,t),L**2)
        temp = np.exp(temp)
        term = np.multiply((1. - (-1.)**n)/n**4,temp)
        result = np.add(result,term)
        if np.all(abs(term) < 1e-15):  # Adjust the tolerance level as needed
            break
        n = n + 1.
    
    result = np.multiply(L**2/6.,np.subtract(1.,result))
    
    return result


def plot_fit_msd(input_filename, title, linear = True):
    
    [time,msd] = np.loadtxt(filename,dtype='float',comments=['#','@'],unpack=True)
    time = np.divide(time,1000.)
    if linear:
        popt,pcov = curve_fit(linear_diffusion,time,msd)
    else:
        popt,pcov = curve_fit(nonlinear_diffusion,time,msd)
        perr = np.sqrt(np.diag(pcov))
        D = popt[0]
        D_error = perr[0]
        
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(time,msd,color=KTHturq)
    if linear:
        ax1.plot(time,linear_diffusion(time,*popt),linestyle="--",color=KTHbrick,label='Fit: $D$ = %5.3f $\pm$ %5.3f' % (popt[0],perr[0]),zorder=2)
    else:
        ax1.plot(time,nonlinear_diffusion(time,*popt),linestyle="--",color=KTHbrick,label='Fit: $D$ = %5.3f $\pm$ %5.3f' % (D,D_error),zorder=2)
    ax1.set_xlabel("time [ns]") 
    ax1.set_ylabel("MSD $[nm^2]$")
    plt.legend(loc="best")
    ax1.set_ylim(bottom=0.)
    ax1.set_xlim(left=0.)
    plt.title(title)
    output_filename = filename[:-3]+"png"
    plt.savefig(output_filename,bbox_inches='tight',dpi=600,transparent=True)
    plt.show()
    
    return D,D_error


def plot_densities(input_list,system_IDs):
    #takes a list of files and a list of system IDs (usually in the format of "L = ..."), plots all of the densities in the files
    # designed to correct effects from having water on both sides of a cellulose slab
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for filename in input_list:
        [yc,rhoyc] = np.loadtxt(filename,dtype='float',comments=['#','@'],unpack=True)
        halfpoint = int(len(yc)/2)
        yc_1 = yc[:halfpoint]
        yc_2 = yc[halfpoint:]
        rhoyc_1 = rhoyc[:halfpoint]
        rhoyc_2 = rhoyc[halfpoint:]
        rhoyc_1 = rhoyc_1[::-1]
        rhoyc_avg = (rhoyc_1 + rhoyc_2)/2.
        yc_corrected = yc_2- cellulose_thickness/2.
        ax.plot(yc_corrected,rhoyc_avg,label=system_IDs[input_list.index(filename)],color=colourlist[input_list.index(filename)])
    
    ax.set_xlabel("distance from the cellulose surface [nm]") 
    ax.set_ylabel(r"$\rho_y$ [$kg/m^3$]")
    ax.set_xlim(-0.25,1.5)
    ax.set_ylim(-20,1250)
    plt.legend(loc="lower right")

    plt.savefig("Ydensity_averaged.png",bbox_inches='tight',transparent=True)
    plt.show()
    
    return ax,fig


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)
def calculate_2dift(input):
    ift = np.fft.ifftshift(input)
    ift = np.fft.ifft2(ift)
    ift = np.fft.fftshift(ift)
    return ift.real

def calculate_distance_from_center(coords, centre): 
    #helping function for radial averaging, assumes square image
    return np.sqrt((coords[0] - centre) ** 2. + (coords[1] - centre) ** 2.)


def radial_avg(abs_ft):   
    #radial averages an image of different intensities that you get from a 2D Fourier transform
    ft_distances = np.zeros(np.shape(abs_ft))
    center = int(len(abs_ft)/2)
    for i in range(len(abs_ft)): #go through the whole ft results 2D array
        for j in range(len(abs_ft)): #square!!!
            index = (i,j) #where we are in the array rn
            ft_distances[i][j] = calculate_distance_from_center(index,center) #this distance is in pixels, distance of each array point from the center gets saved

    flat_ft = abs_ft.flatten() #flatten both arrays for easier handling
    flat_dist = ft_distances.flatten()
    sorted_indices = sorted(range(len(flat_dist)), key=lambda k: flat_dist[k]) #this and next two: sort the two arrays based on distance
    flat_ft = [flat_ft[i] for i in sorted_indices]
    flat_dist = [flat_dist[i] for i in sorted_indices]

    #find the maximum and minimum (0) distance, create bins + an array to store the averaged results
    max_dist = np.max(flat_dist)
    
    dist_range = np.linspace(0,max_dist,501) #total datapoints is 200x200 = 40k-ish, this is 1000 bins
    
    counts = np.zeros(len(dist_range)-1) #create the bins, one for counting how many are in each bin and one for addign the intensities up
    intensities = np.zeros(len(dist_range)-1)
    
    #go through the bins, for each bind find the values that belong in it, average them
    for i in range(len(counts)):
        for datapoint in flat_dist:
            if (datapoint>=dist_range[i])&(datapoint<dist_range[i+1]):   #identifies each datapoint (dist) that is within the bin
                intensities[i] = intensities[i] + flat_ft[flat_dist.index(datapoint)] #adds up the ft data correspoinding to the distances
                counts[i] = counts[i] + 1
    
    for i in range(len(intensities)):
        if counts[i]!=0: #to supress the 0 division warnings
            intensities[i] = np.divide(intensities[i],counts[i]) #do the averaging
    
    return dist_range[:-1],intensities

def plot_densmap(input_filename,title):
    global KTHcmap
    #unlike plot_densities, this only takes in one file
    
    data = np.loadtxt(input_filename, delimiter="\t") 
    z_axis = data[0][:]
    z_axis = z_axis[1:]
    #z_axis = np.append(z_axis,z_axis[-1]+(z_axis[1]-z_axis[0])) #adding last bin
    x_axis = []


    data = data[:][1:] #take off the first row that is a header

    cleaned_data = [] #all this cleaning needed because of how gmx densmap works

    for line in data: #take off the first element of each row which is a coloumn header 
        cleaned_line = line[1:]
        x_axis.append(line[0])
        cleaned_data.append(cleaned_line)
        
    densities = np.asarray(cleaned_data)
    densities = np.transpose(densities)
    #x_axis.append(x_axis[-1]+(x_axis[1]-x_axis[0]))
    x_axis = np.asarray(x_axis)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    c = ax.pcolormesh(x_axis, z_axis, densities, shading="nearest", cmap=KTHcmap, vmin=np.min(densities), vmax=np.max(densities))
    ax.set_xlabel("x [nm]") 
    ax.set_ylabel("z [nm]")

    plt.title(title)
    plt.colorbar(c, ax=ax,label=r"number density [$1/nm^3$]")
    output_filename = input_filename[:-3]+"png"
    plt.savefig(output_filename,bbox_inches='tight',transparent=True)
    plt.show()
    
    return x_axis,z_axis,densities,fig,ax

def densmap_2DFourier(densities,output_filename,limits=None):
    #limits is: xmin,xmax,zmin,zmax
    global KTHcmap
    ft = calculate_2dft(densities)
    
    abs_ft = abs(ft)
    center = int(len(abs_ft)/2)
    abs_ft[center][center] = 0. #beamstop :)
    
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    image = plt.imshow(abs_ft,cmap=KTHcmap)
    ax1.set_xlabel(r"$x^{-1}$ $[{nm}^{-1}]$")
    ax1.set_ylabel(r"$z^{-1}$ $[{nm}^{-1}]$")
    plt.colorbar(image, ax=ax1,label=r"intensity")
    plt.title(r"2DFFT density map")
    if limits != None:
        ax1.set_xlim(limits[0],limits[1])
        ax1.set_ylim(limits[2],limits[3])
    #output_filename = filename[:-4]+"_2DFFT.png"
    plt.savefig(output_filename,bbox_inches='tight',transparent=True)
    plt.show()
    
    dist_range, intensities = radial_avg(abs_ft)
    return dist_range,intensities

