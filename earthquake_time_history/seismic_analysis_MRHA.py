"""
Script that uses the Python Library MDyn for Modal Superposition Analysis
to solve the response of a frame under seismic loading (accelerogram)

Modal dynamic solver, cite as:
Camara A (2021). A fast mode superposition algorithm and its application to the analysis of bridges under moving loads,
Advances in Engineering Software, 151: 102934.

"""

import numpy as np
from tqdm import tqdm
import sys
import os

sys.path.append('..\code')
from MDyn_forcing import *
from MDyn_utils import *
from MDyn_solver import *
from MDyn_vibsystems import *



#from matplotlib import pyplot as plt
import time
t = time.time()

if __name__ == '__main__':

    ######################## ADD YOUR DATA HERE #########################
    caseName = 'results'	# Name of the folder that will contain the results (please don't add spaces and don't start it with a number)
    modesToInclude = range(1,9+1)  # [1,2,7,9] # List of modes to be included in the calculations
    DOFactive = [1,2,3,4,5,6]  # Interesting DOF of the problem, 1 is Ux, 2 is Uy, 3 is Uz, 4 is URx, 5 is URy, 6 is URz
    # Newmark analysis
    beta=1./4
    gamma=1./2

    ################# READ BUILDING INFORMATION FILES

    # Load modal information from mmodal analysis
    NodeInformationFile = './structureData/NodeInformation.txt'
    ModalInformationFile = './structureData/ModalInformation.txt'
    FrequencyInformationFile = './structureData/FrequencyInformation.txt'
    BeamInformationFile = './structureData/BeamInformation.txt'


    ################# READ Earthquake INFORMATION FILES - SYNCHRONOUS
    GammaInformation = np.loadtxt('./structureData/GammaInformation.txt')   # Participation factors of each mode (x,y,z,urx,ury,urz)
    # For animation only if SYNC motion:
    earthquakeInfoGeometry = np.loadtxt('./earthquakeData/earthquakeInfoGeometry.txt')
    NodesEarthquake = earthquakeInfoGeometry[:,0]
    DirectionsEarthquake = earthquakeInfoGeometry[:,1:4]    # col 1 x, col 2 y, col 3 z, 0 for not applied, 1 for eq. applied without scale factor. e.g. if eq is read in g, instead of 1 is 9.81.
    ######
    accel = np.loadtxt('./earthquakeData/accel_1.txt')
    accel_t = accel[:,0]
    accel_X = accel[:,1]*9.81       # Accelerogram in m/s2
    #accel = np.loadtxt('./earthquakeData/accel_1.txt')
    #accel_Y = accel[:,1]*9.81       # Accelerogram in m/s2
    accel = np.loadtxt('./earthquakeData/accel_3.txt')
    accel_Z = accel[:,1]*9.81       # Accelerogram in m/s2
    accel_Y = np.zeros(len(accel_X))


    tmax = accel_t[-1]    # Total calculation time.
    dt = accel_t[1] - accel_t[0] # 1./500 # Step time in s.


    ##### Read structure files
    NodeNumber,NodeX,NodeY,NodeZ,NumberOfNodes,NumberOfModes,Phi,wnv,xinv,BeamNumber,BeamLength,BeamNode1,BeamNode2,BeamSectionLabel = readBridgeInfoFiles(NodeInformationFile,ModalInformationFile,FrequencyInformationFile,BeamInformationFile,DOFactive)


    ########## UNITS
    kPhiWithUnits = 1.  # Some software, e.g. sofistik, give mode shapes with "units", like tonf m, in that case kPhiWithUnits = X applies a correction factor (to be check). Otherwise set = 1.

    # OUTPUT
    #Create output folder and path to results
    NodeNumberToWrite = [2] #24 for midspan at centreline
    #VehicleOrderToWrite = [0]   # Order of vehicle to get the response from
    writeOutputRate = 1
    # Animation
    animationRate = 0   #10 # Rate of frame recording for the animations, = 0 if no animation is to be recorded
    scaleFactorAnimation = 20
    durationAnimation = 100
    #maxLoad = abs(PHarmonicLoad[0])  # To normalise the load, for visualisation
    #xyzLoad = [[3,0,6]] # xyzLoad is a array with 3x1 arrays that contain the xyz coord of the loads
    removeAnimationFigures = 1 # = 1 to remove all files in the folder from which the animation is created. = 0 otherwise

    print('Pre-processing: COMPLETE')

    ######################################################

    tCPU = time.time()
    kwargs = {
        'DOFactive': DOFactive,
        'modesToInclude': modesToInclude,
        'NumberOfNodes': NumberOfNodes,
        'Phi': Phi,
        'wnv': wnv,
        'xinv': xinv,
        'beta': beta,
        'gamma': gamma,
        'NodeX': NodeX,
        'NodeY': NodeY,
        'NodeZ': NodeZ,
        'NodeNumber': NodeNumber,
        #'VehicleBeams': VehicleBeams,
        'BeamNode1': BeamNode1,
        'BeamNode2': BeamNode2,
        'BeamLength': BeamLength,
        'BeamNumber': BeamNumber,
        'BeamSectionLabel': BeamSectionLabel,
        'GammaInformation': GammaInformation,
        'accel_t': accel_t,
        'accel_X': accel_X,
        'accel_Y': accel_Y,
        'accel_Z': accel_Z,
        'NodesEarthquake':NodesEarthquake,
        'DirectionsEarthquake':DirectionsEarthquake,
        'caseName':caseName,
        'scaleFactorAnimation':scaleFactorAnimation,
        #'plotStructurePart':'spineOnly',
        'NodeNumberToWrite':NodeNumberToWrite,
        'writeOutputRate':writeOutputRate
    }

    # Plot structural model and vibration modes in folder "./preprocessing"
    plots = PlotModel(**kwargs)
    #plots.plotUndeformedModel()
    plots.plotMode(1)
    #plots.plotMode(2)

    t = np.arange(0.,tmax,dt)

    # Simulation
    sl = SeismicLoads(**kwargs)
    initfilter = ModalMatrices(**kwargs)

    # Set control of written output
    writers = WriteToFile(**kwargs)
    r_Output = writers.initVectorOutput(t)
    r2dot_Output = writers.initVectorOutput(t)

    # Filter mode shape matrix
    M,C,K,PhiReduced = initfilter.filterModalMatrices()
    # Initialise vectors
    Pn0,u,udot = initfilter.initialiseModalVectors()
    filenamesAnimation = []

    pbar = tqdm(total=len(t),leave=False)

    #### DYNAMIC response
    r = np.zeros(NumberOfNodes*len(DOFactive))
    rdot = np.zeros(NumberOfNodes*len(DOFactive))
    for i in range(1,len(t)): # Main loop - time
        # Initialise Newmark solver
        u2dot,kbar,a,b = InitialiseMDOFNewmark(Pn0,u,udot,beta,gamma,t[i]-t[i-1],M,C,K)

        # Solution
        ag = sl.ground_acceleration(t[i])   # Ground acceleration at time ti
        Pn = sl.seismic_mrha_sync(ag) # np.dot(np.transpose(PhiReduced),P)
        Pn.shape = u.shape
        u,udot,u2dot = MDOFNewmark(Pn0,Pn,u,udot,u2dot,beta,gamma,t[i]-t[i-1],kbar,a,b)		# v1_1 don't consider impulsive load but ramp load from start of step to end.
        r,rdot,r2dot = structuralResponse(u,udot,u2dot,PhiReduced)
        if kPhiWithUnits != 1:
            r=r*kPhiWithUnits # Sofistik correction
            rdot=rdot*kPhiWithUnits # Sofistik correction
            r2dot=r2dot*kPhiWithUnits # Sofistik correction

        Pn0 = Pn	# Update Pn0 to value in previous step

        # Write output
        output_flag = writeOutputRate > 0 and (i % writeOutputRate == 0 or i == 1 or i == len(t))
        if output_flag:
            writers.writeOutput(r_Output,r,i)
            writers.writeOutput(r2dot_Output,r2dot,i)

        # Animation
        output_flag = animationRate > 0 and (i % animationRate == 0 or i == 1 or i == len(t))
        if output_flag:
#            plots.plotDeformedStructureSection(i,t[i],r,showPlotSection,plotSectionNodes)
#            plots.plotDeformedStructureSectionLoads(i,t[i],r,LoadAmplitude,maxLoad,xyCGVehiclev,showPlotSection,plotSectionNodes)
            #plots.plotDeformedStructure(i,t[i],r)
            plots.plotDeformedStructureEarthquake(i,t[i],r,ag)

            filenamesAnimation.append('deformation_iteration_'+str(i)+'.png')

        # Progress bar
        pbar.update(1)

    elapsed = time.time() - tCPU
    print('Calculation time: ', elapsed, ' s')


    print(' Calculation: COMPLETE')

    writers.writeOutputFile(r_Output,'displacement','plotsYes') # plotsYes
    writers.writeOutputFile(r2dot_Output,'acceleration','plotsYes') # plotsYes

    print(' Results output: COMPLETE')

    # Create animation from files
    if animationRate!=0:
        plots.createAnimation(filenamesAnimation,durationAnimation,removeAnimationFigures)
