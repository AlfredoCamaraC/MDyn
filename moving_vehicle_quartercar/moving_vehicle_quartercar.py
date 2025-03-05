"""
Script that uses the Python Library MDyn for Modal Superposition Analysis
to solve the response of a bridge under stepping pedestrian loads without
human-structure interaction

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
    modesToInclude = range(1,7+1)  # [1,2,7,9] # List of modes to be included in the calculations
    DOFactive = [1,2,3,4,5,6]  # Interesting DOF of the problem, 1 is Ux, 2 is Uy, 3 is Uz, 4 is URx, 5 is URy, 6 is URz
    # Newmark analysis
    beta=1./4
    gamma=1./2

    ################# READ BRIDGE INFORMATION FILES
    # For a bridge with vehicles only:
    VehicleBeams = np.loadtxt('./bridgeData/VehicleBeams.txt') # Col 1: Beam numbers on which the pedestrian crosses the bridge, Col 2: distance between shear center and asphalt (not relevant here)

    # Load modal information from mmodal analysis
    NodeInformationFile = './bridgeData/NodeInformation.txt'
    ModalInformationFile = './bridgeData/ModalInformation.txt'
    FrequencyInformationFile = './bridgeData/FrequencyInformation.txt'
    BeamInformationFile = './bridgeData/BeamInformation.txt'

    ################# READ PAVEMENT INFORMATION FILE
    IrregularityFlag = 'Off' # 'On' if irregularities are defined or Off otherwise (perfect surface)
    IrregularityData = np.loadtxt(r'./bridgeData/bump.txt')   # np.loadtxt(r'bump.txt') # col 1 x, col 2 r in wheel line 1, col 2 r in line 2, all units in m
    yCoordIrregularityData = [-1,1] # y coordinates of the profile lines given in IrregularityData, in m
    ################# DEFINE VEHICLE
    # DEFINE vehicle (properties of quarter car, or whole car for heave motion)
    Mv = 18.5e3    # Mass of quarter vehicle in kg
    Cv = 10e3  # Vertical damping of suspension/tyre of one wheel, in Nm/s
    Kv = 730e3  # Vertical stiffness of suspension/tyre of one wheel, in N/m
    fcar = np.sqrt(Kv/Mv)/(2*np.pi)
    print('vertical frequency of the car [Hz]:',fcar)
    MassVehicles = [[Mv]] # [Vehicle 1, vehicle 2, ...]
    DampingVehicles = [[Cv]] # [Vehicle 1, vehicle 2, ...]
    StiffnessVehicles = [[Kv]] # [Vehicle 1, vehicle 2, ...]
    VehicleType = ['quarterCar']    # [Vehicle 1, Vehicle 2]

    YVehicle = [[0]] # Local distance between centre of car and wheel contact point, in lateral direction, in m [[wheel 1 of car 1, wheel 2 of car 1, etc],[wheel 1 of car 2, wheel 2 of car 2, etc],...]
    XVehicle = [[0]] # Local distance between centre of car and wheel contact point, in longitudinal direction, in m  [[wheel 1 of car 1, wheel 2 of car 1, etc],[wheel 1 of car 2, wheel 2 of car 2, etc],...]
    XCGVehicle0 = [0] # Global distance between origin of coordinates of bridge and center of vehicle at start of analysis, in longitudinal direction, in m, [vehicle 1, vehicle 2, etc]
    YCGVehicle0 = [0] # Global distance between origin of coordinates of bridge and center of vehicle at start of analysis, in transverse direction, in m, [vehicle 1, vehicle 2, etc]

    VLoad = 100./3.6 # Velocity of the vehicle in m/s
    VVehicle = [VLoad] # [vehicle 1,vehicle 2,...]

    DirectionLoad = 3 # ONLY WORKS FOR =3 IN THE MOVING LOAD ANALYSIS, DO NOT CHANGE
    DirectionMoment = 4 # Because the vertical load may induce torsion

    L = 40.                 # Total length of the deck
    tmax = 2.0*L/VLoad     # Total calculation time. Keep this to have it 25% larger than the time it takes for a single load to cross the bridge if it starts at the left abutment
    dt = 0.01 # Step time in s.


    ##### Read structure files
    NodeNumber,NodeX,NodeY,NodeZ,NumberOfNodes,NumberOfModes,Phi,wnv,xinv,BeamNumber,BeamLength,BeamNode1,BeamNode2,BeamSectionLabel = readBridgeInfoFiles(NodeInformationFile,ModalInformationFile,FrequencyInformationFile,BeamInformationFile,DOFactive)

    ########## UNITS
    kPhiWithUnits = 1.  # Some software, e.g. sofistik, give mode shapes with "units", like tonf m, in that case kPhiWithUnits = X applies a correction factor (to be check). Otherwise set = 1.

    # OUTPUT
    #Create output folder and path to results
    NodeNumberToWrite = [10051] #24 for midspan at centreline
    VehicleOrderToWrite = [0]   # Order of vehicle to get the response from
    writeOutputRate = 1
    # Animation
    animationRate = 0   #10 # Rate of frame recording for the animations, = 0 if no animation is to be recorded
    scaleFactorAnimation = 1000
    VehicleVisual = [[1,1,1],[1,1,1]]    # size of vehicle in video [[vehicle 1 length,vehicle 1 width, vehicle 1 height],[vehicle 2 length,vehicle 2 width, vehicle 2 height],...]
    #yc = 2. # Shift nodes to refer to center of section
    #zc = 0.3 # Shift nodes to refer to center of section
    #showPlotSection = [[0-yc,0-zc],[4-yc,0-zc],[4-yc,-0.1-zc],[3-yc,-0.1-zc],[3-yc,-1.1-zc],[1-yc,-1.1-zc],[1-yc,-0.1-zc],[0-yc,-0.1-zc]] # Coordinates of nodes of section to be plotted, connected with lines and closed.
    #plotSectionNodes = [10001,100011,10021,100031,10041,100051,10061,100071,10081,100091,10101] # Nodes in which sections are plot
    durationAnimation = 100
    #maxLoad = abs(Wpedestrian)  # To normalise the load, for visualisation
    removeAnimationFigures = 1 # = 1 to remove all files in the folder from which the animation is created. = 0 otherwise

    # Vehicle output

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
        'DirectionMoment': DirectionMoment,
        'DirectionLoad': DirectionLoad,
        'caseName':caseName,
        'scaleFactorAnimation':scaleFactorAnimation,
        #'plotStructurePart':'spineOnly',
        'NodeNumberToWrite':NodeNumberToWrite,
        'writeOutputRate':writeOutputRate,
        ##### Vehicles VBI
        'VehicleOrderToWrite':VehicleOrderToWrite,
        'VehicleBeams':VehicleBeams,
        'MassVehicles':MassVehicles,
        'DampingVehicles':DampingVehicles,
        'StiffnessVehicles':StiffnessVehicles,
        'VehicleType':VehicleType,
        'YVehicle':YVehicle,
        'XVehicle':XVehicle,
        'XCGVehicle0':XCGVehicle0,
        'YCGVehicle0':YCGVehicle0,
        'VVehicle':VVehicle,
        'IrregularityFlag':IrregularityFlag,
        'IrregularityData':IrregularityData,
        'yCoordIrregularityData':yCoordIrregularityData,
        'VehicleVisual':VehicleVisual
    }

    # Plot structural model and vibration modes in folder "./preprocessing"
    plots = PlotModel(**kwargs)
    #plots.plotUndeformedModel()
    #plots.plotMode(1)
    #plots.plotMode(2)

    t = np.arange(0.,tmax,dt)

    # Simulation
    ml = MovingLoads(**kwargs)
    initfilter = ModalMatrices(**kwargs)

    # Set control of written output
    writers = WriteToFile(**kwargs)
    r_Output = writers.initVectorOutput(t)
    r2dot_Output = writers.initVectorOutput(t)
    q_Output = writers.initVectorOutputVehicle(t)
    q2dot_Output = writers.initVectorOutputVehicle(t)
    F_Output = writers.initVectorOutputReactionsVehicle(t)  # Vehicle reactions

    # Filter mode shape matrix
    M,C,K,PhiReduced = initfilter.filterModalMatrices()
    # Initialise vectors
    Pn0,u,udot = initfilter.initialiseModalVectors()
    filenamesAnimation = []

    pbar = tqdm(total=len(t),leave=False)

    #### DYNAMIC response
    r = np.zeros(NumberOfNodes*len(DOFactive))
    rdot = np.zeros(NumberOfNodes*len(DOFactive))
    Ncars = len(VVehicle)
    q = ml.initialiseVehicleResponse()
    qdot = ml.initialiseVehicleResponse()
    q2dot = ml.initialiseVehicleResponse()

    F = ml.InitialiseVehicleReactions()
    zpavement = ml.InitialiseContactIrregularities()


    for i in range(1,len(t)): # Main loop - time
        ts = [t[i-1],t[i]]
        # Initialise Newmark solver
        u2dot,kbar,a,b = InitialiseMDOFNewmark(Pn0,u,udot,beta,gamma,t[i]-t[i-1],M,C,K)
        # Initialise Modal forcing
        P = np.zeros(NumberOfNodes*len(DOFactive))
        ####### Sum of all the dynamic actions
        VehicleReaction = np.zeros(NumberOfNodes*len(DOFactive))
        #LoadAmplitude = np.zeros(len(fmv))
        xyCGVehiclev = np.zeros((Ncars,2))
        zgv = []

        for dl in range(Ncars): # Loop in vehicle
            # Find position of vehicle centroid
            xyCGVehicle = ml.FindCentroidLoadConstantSpeed(dl,t[i])
            xyCGVehiclev[dl,:] = xyCGVehicle  # For the animation
            # Obtain load
            [VehicleReactionT,q[dl],qdot[dl],F[dl],zpavement[dl],zg] = ml.VBIvertical(dl,xyCGVehicle,r,rdot,q[dl],qdot[dl],F[dl],zpavement[dl],ts)
            zgv.append(zg) # For the animation


            #LoadAmplitude[dl] = ml.SteppingMovingLoads(dl,xyCGVehicle,t[i])[1] # For the animation
            VehicleReaction += VehicleReactionT

        P = VehicleReaction #PmovingLoad #+ PharmonicFixedLoad

        # Solution
        Pn = np.dot(np.transpose(PhiReduced),P)
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
            writers.writeOutputVehicleDynamics(q_Output,q,i)
            writers.writeOutputVehicleForces(F_Output,F,i)

        # Animation
        output_flag = animationRate > 0 and (i % animationRate == 0 or i == 1 or i == len(t))
        if output_flag:
#            plots.plotDeformedStructureSection(i,t[i],r,showPlotSection,plotSectionNodes)
            plots.plotDeformedStructureVBI(i,t[i],r,q,xyCGVehiclev,zgv)
            filenamesAnimation.append('deformation_iteration_'+str(i)+'.png')

        # Progress bar
        pbar.update(1)

    elapsed = time.time() - tCPU
    print('Calculation time: ', elapsed, ' s')


    print(' Calculation: COMPLETE')

    writers.writeOutputFile(r_Output,'displacement','plotsYes') # plotsYes
    writers.writeOutputFile(r2dot_Output,'acceleration','plotsYes') # plotsYes
    writers.writeOutputFileResponseVehicles(q_Output,'displacement','plotsYes')
    writers.writeOutputFileReactionVehicles(F_Output,'plotsYes')

    print(' Results output: COMPLETE')

    # Create animation from files
    if animationRate!=0:
        plots.createAnimation(filenamesAnimation,durationAnimation,removeAnimationFigures)
