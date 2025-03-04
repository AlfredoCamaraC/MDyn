# Created by Alfredo Camara Casado
# -------------------------------
# Description: program to determine the total weighted rms acceleration of the driver seat to study the ride comfort
# -------------------------------

import  numpy as np
from matplotlib import rc
from matplotlib.pyplot import *
import pylab
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import os
import matplotlib.ticker as mticker
from matplotlib.font_manager import FontProperties
import matplotlib.image as image
import scipy.signal as sig
from scipy.integrate import simps


# ............
fig=pylab.figure()
ax=fig.add_subplot(111)



###############################
a = np.loadtxt('./ABAQUS/U1_NODE2_ABAQUS.txt')
t_abaqus = a[:,0]
u1_abaqus = a[:,1]*1000

###############################
a = np.loadtxt('./time_histories/displacement_node_2_MDyn.txt')
t_MDyn = a[:,0]
u1_MDyn = a[:,1]

###############################
ax.plot(t_abaqus,u1_abaqus,'o',linewidth=1.0,markersize=6,markevery=5,color='red',label='MRHA ABAQUS')
ax.plot(t_MDyn,u1_MDyn,'-',linewidth=1.0,markersize=6,markevery=1,color='blue',label='MRHA MDyn')


# --------- Custom plot -----------

for tl in ax.get_yticklabels():
    tl.set_size(12)
for tl in ax.get_xticklabels():
    tl.set_size(12)


ax.grid()
axhline(y=0.0,color='black',linestyle='-',linewidth=1.0)


xlim(0,max(t_abaqus))

    # ------ legend ------
    # add the legend in the upper right corner of the plot
leg = legend(fancybox=True, loc='best', shadow=False, prop={'size': 10})
    # set the alpha value of the legend: it will be translucent
leg.get_frame().set_alpha(0.6)

xlabel(r'Time; $t$ [s]', fontsize=13)
ylabel(r'Horizontal displacement; $r_x$ [mm]', fontsize=13)

savefig('comparison_abaqus_Mdyn_U1_node2.pdf')
close() # Close a figure window

#show()
