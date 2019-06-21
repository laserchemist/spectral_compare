# Spectral comparison and peak finder
#
# JMS 18 June 2019
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
from scipy import interpolate
from scipy.signal import correlate
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks_cwt, find_peaks, peak_widths
from scipy.misc import electrocardiogram # Early testing of signals
import math
import pandas as pd
import os
from os import listdir
from os.path import join as osjoin
# Define functions
def printMatrixE(a):
   print("Matrix["+("%d" %a.shape[0])+"]["+("%d" %a.shape[1])+"]")
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print("%6.3f" %a[i,j], end = '  '),
      print()
   print()      

# get the list of files in mypath and store in a list
mypath = os.getcwd()
print(mypath) # Works
# Set up correlation array
specx=[]
specy=[]
corr=np.zeros(shape = (5,5))
# Peak data storage
peakindex=[]
peak=[]
ph=[]
pw=[]
# Find data files in current directory
onlycsv = [f for f in listdir(mypath) if f.endswith('.csv') ]
# print out all the files with it's corresponding index
print("+++++++++++++++++++++++++++++++++++++++++++++++++")
print("Spec_compare Spectral Processing and comparison program") 
print("J. Smith v.1.0 21 June")
print("+++++++++++++++++++++++++++++++++++++++++++++++++")

# prompt the user to select the file
option=[] # List to store file number choices
xs = np.linspace(100, 4500, 1000) # Define basis for spline grid
num=0
while num<5:
    print("+++++++++++++++++\n++ Available Spectral Files ++\n+++++++++++++++++")
    for i in range(len(onlycsv)):
        print( i, onlycsv[i] )
    option.append(int(input("-> Select a file ({}) by number or none (-1): ".format(num))))
    if option[-1]==-1:
        break
    filename = osjoin(mypath, onlycsv[option[-1]])
    print("File selected: ",onlycsv[option[-1]])
    headers = ["wavenumber","AU","W1","W2","W3","WG"]
    data = pd.read_csv(filename, sep=',\s+', skiprows=1, names = headers, engine='python')
    x = data["wavenumber"]
    y = data["AU"]
    if num==-1:
        specx=x
        specy=y
    else:
        print("num: ",num)
        specx.append(x)
        specy.append(y)
    spl = UnivariateSpline(x, y,k=4,s=0)        
    indexes = find_peaks_cwt(y, np.arange(1, 550),noise_perc=95)
    avg=np.mean(y) #Upper bound estimate of flat baseline
    peaks, properties = find_peaks(spl(x),height=avg*3, width=2) 
    results_half = peak_widths(spl(x), peaks, rel_height=0.5)
    peakheight=np.array(properties["peak_heights"])
    #print(peaks,peakheight,results_half[0])
    peak_data=np.stack((x[peaks],peakheight,results_half[0]),axis=-1)
    print("Peak data shape",peak_data.shape)
    peakindex.append(peaks)
    peak.append(peak_data[:,0])
    ph.append(peak_data[:,1])
    pw.append(peak_data[:,2])
    for i,value  in enumerate(peak_data, 1):
        print(' '.join([' %8.4f' % (value[n]) for n in range(len(value))])) # Try to display wavenumbers
    # Try peak finding with first derivative evaluated at all the points
    d1s = spl.derivative()
    d1 = d1s(y)
    # we can get the roots directly here, which correspond to minima and
    # maxima.
    #print('Roots = {}'.format(spl.derivative().roots())) #Alternative peak find
    minmax = spl.derivative().roots()
    #print("Roots wavenumber: ",[x2[i] for i in int(minmax)])
    # START PLOTTING
    plt.xlabel('Wavenumber')
    plt.ylabel('arbitrary units')
    plt.axis([min(x), 4500, 0, max(y)*1.1])
    plt.plot(x,spl(x), 'g-', linewidth=2.0, label=onlycsv[option[-1]])
    plt.plot(x[peaks], spl(x[peaks]), 'ro ', label='peaks')
    plt.legend(loc='best')
    plt.savefig(filename+"peaks.pdf",dpi=300)
    plt.close()
    num+=1
# Do overall analysis
for i in range(num):
            for j in range(num):
                spli=UnivariateSpline(specx[i], specy[i],k=4,s=0)
                splj=UnivariateSpline(specx[j], specy[j],k=4,s=0)
                correl = np.corrcoef(spli(xs), splj(xs)) # Compared using same x = "xs"
                corr[i][j]=correl[1][0]
# Now plot all resulting plot panels
if num>1:
    f, axarr = plt.subplots(num, sharex=True)
    for i in range(num):
        axarr[i].plot(specx[i], specy[i], 'C'+str(i),lw=1, label=onlycsv[option[i]])
        #  Try to fix to only plot top few peaks
        axarr[i].plot(peak[i], ph[i], 'ro ', markersize=1, label='peaks')
        axarr[i].legend( shadow=True,   fontsize=6,loc='best')
        axarr[i].axis([100, 4500, -max(specy[i])*0.05, max(specy[i])*1.1])
        axarr[i].annotate( "Correlation:{:.{}f}".format( corr[0,i], 3 ), (.5, .9),size=5, weight='bold',  
            xycoords='axes fraction', va='center')
        axarr[i].grid(True)
        axarr[i].xaxis.set_major_locator(MultipleLocator(500))
        axarr[i].xaxis.set_minor_locator(MultipleLocator(100))
        axarr[i].set_yticks([])
        axarr[i].tick_params(which='minor', length=4, color='r')
        axarr[i].set_xlabel('Wavenumber')
    # 
    # Bring subplots close to each other.
    f.subplots_adjust(hspace=0)
    for ax in axarr:
        ax.label_outer()
    plt.savefig(filename+".pdf",dpi=300)
    plt.close()
else:
    fig, ax = plt.subplots()
    ax.plot(specx[0], specy[0], 'C'+str(7),lw=1, label=onlycsv[option[0]])
    #  Try to fix to only plot top few peaks
    ax.plot(peak[0], ph[0], 'ro ', markersize=1, label='peaks')
    ax.legend( shadow=True,   fontsize=6,loc='best')
    ax.axis([100, 4500, -max(specy[0])*0.05, max(specy[0])*1.1])
    ax.grid(True)
    ax.xaxis.set_major_locator(MultipleLocator(500))
    ax.xaxis.set_minor_locator(MultipleLocator(100))
    ax.set_yticks([])
    ax.tick_params(which='minor', length=4, color='r')
    plt.xlabel('Wavenumber')
    plt.savefig(filename+".pdf",dpi=300)
    plt.close()
print("Peak data\n++++++++")
fn=filename+"_peakdata.txt"
peakfile=open(fn,"w+")
for i in range(num):
    print("++++++++\n","Spectrum",i,onlycsv[option[i]],"\n=================")
    peakfile.write("Spectrum "+str(i)+": "+onlycsv[option[i]]+"\n")
    peakfile.write("#, wavenumber, height, width\n")
    for j, pk in enumerate(peak[i]):
        print(j,"{:.{}f}".format( pk, 1 ),"{:.{}f}".format( ph[i][j], 3 ),"{:.{}f}".format( pw[i][j], 3 ))
        peakfile.write(str(j)+", {:.{}f}".format( pk, 1 )+", {:.{}f}".format( ph[i][j], 3 )+", {:.{}f}".format( pw[i][j], 3 )+"\n")
## Now print spectral correlation data computed above
print("\nSpectral Correlation matrix\n______________________\n")
printMatrixE(corr[:num,:num])
# Now add to file
peakfile.write("Spectral Correlation Matrix\n")
a=corr[:num,:num]
rows = a.shape[0]
cols = a.shape[1]
for i in range(0,rows):
    for j in range(0,cols):
        peakfile.write("%6.3f, "%a[i,j])
    peakfile.write("\n")      
peakfile.close()