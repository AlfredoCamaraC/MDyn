# -*- coding: utf-8 -*-
"""
Created by Alfredo Camara Casado
For extra capabilities of the code, code development, and bug reporting please contact the author at: acamara@ciccp.es
"""

from MDyn_solver import *
from MDyn_utils import *
from MDyn_vibsystems import *

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve

#from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import odeint
#from scipy.integrate import simps
from scipy import integrate

#import time
import os
from subprocess import call
from bisect import bisect_left
import multiprocessing as mp

#################
print('')

print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
print('')

print('Please cite as:')
print('')

print('Camara A (2021). A fast mode superposition algorithm and its application to the analysis of bridges under moving loads,')
print('Advances in Engineering Software, Volume 151, 102934.')
print('')

print('::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
print('')


class MovingLoads():

    def __init__(self, **kwargs):
        self.DOFactive = kwargs.get("DOFactive")
        self.modesToInclude = kwargs.get("modesToInclude")
        self.NumberOfNodes = kwargs.get("NumberOfNodes")
        self.Phi = kwargs.get("Phi")
        self.wnv = kwargs.get("wnv")
        self.xinv = kwargs.get("xinv")
        self.beta = kwargs.get("beta")
        self.gamma = kwargs.get("gamma")
        self.NodeX = kwargs.get("NodeX")
        self.NodeNumber = kwargs.get("NodeNumber")
        self.BeamNode1 = kwargs.get("BeamNode1")
        self.BeamNode2 = kwargs.get("BeamNode2")
        self.BeamLength = kwargs.get("BeamLength")
        self.BeamNumber = kwargs.get("BeamNumber")
        self.TipeOfLoad = kwargs.get("TipeOfLoad")

        self.VehicleBeams = kwargs.get("VehicleBeams")
        self.DirectionMoment = kwargs.get("DirectionMoment")
        self.DirectionLoad = kwargs.get("DirectionLoad")
        self.indexDOFLoad=self.DOFactive.index(self.DirectionLoad)
        self.indexDOFMoment=self.DOFactive.index(self.DirectionMoment)

        self.PVehicle = kwargs.get("PVehicle")
        self.YVehicle = kwargs.get("YVehicle")
        self.XVehicle = kwargs.get("XVehicle")
        self.XCGVehicle0 = kwargs.get("XCGVehicle0")
        self.YCGVehicle0 = kwargs.get("YCGVehicle0")
        self.VVehicle = kwargs.get("VVehicle")
        self.IrregularityFlag = kwargs.get("IrregularityFlag")
        if self.IrregularityFlag == 'On':
            self.IrregularityData = kwargs.get("IrregularityData")
            self.yCoordIrregularityData = kwargs.get("yCoordIrregularityData")
            self.xPavement = self.IrregularityData[:,0]
            self.zPavementLine1 = self.IrregularityData[:,1]
            self.zPavementLine2 = self.IrregularityData[:,1]

        self.VLoad = kwargs.get("VLoad")
        self.index0=np.where(self.BeamNumber==self.VehicleBeams[0,0])[0][0]
        self.index1=np.where(self.BeamNumber==self.VehicleBeams[-1,0])[0][0]

        self.xFirstNodeVehicleBeams=self.NodeX[self.index0]  # First node of the beam elements over which the moving load passes
        self.xLastNodeVehicleBeams=self.NodeX[self.index1+1]  # Last node of the beam elements over which the moving load passes

        self.NodeXVehicleBeams = self.NodeX[self.index0:self.index1+2]# ASSUMES CONSECUTIVE NUMBERING IN VEHICLEBEAMS AND nodes

        # Pedestrians
        self.Gv = kwargs.get("Gv")
        self.phiv = kwargs.get("phiv")
        self.fmv = kwargs.get("fmv")

        # Vehicle VBI
        self.MassVehicles = kwargs.get("MassVehicles")
        self.DampingVehicles = kwargs.get("DampingVehicles")
        self.StiffnessVehicles = kwargs.get("StiffnessVehicles")
        self.VehicleType = kwargs.get("VehicleType")
        self.FzStatic = []  # Static load of each wheel in vertical direction [[wheel 1 vehicle 1, wheel 2 vehicle 1...][wheel 1 vehicle 2, ...]]
        if self.VehicleType is not None:
            for dl in range(len(self.VVehicle)):
                if self.VehicleType[dl] == 'quarterCar':
                    self.FzStatic.append(-1*self.MassVehicles[dl][0]*9.81)

    def FindCentroidLoadConstantSpeed(self,dl,time):
        # Assumed constant speed and straight path parallel to bridge
        XCGVehicle = self.XCGVehicle0[dl]+(self.VVehicle[dl]*time)
        YCGVehicle = self.YCGVehicle0[dl]
        return [XCGVehicle,YCGVehicle]

    def InterpolateIrregularity(self,xyCGVehicle):
        # Obtain pavement irregularities at single wheel contact point

        x_input = xyCGVehicle[0]
        y_input = xyCGVehicle[1]

        # Ensure x_input is within range
        if x_input < self.xPavement[0] or x_input > self.xPavement[-1]:
            raise ValueError("x_input is out of bounds")

        # Find the two closest x-values for interpolation
        idx = np.searchsorted(self.xPavement, x_input) - 1
        idx = max(0, min(idx, len(self.xPavement) - 2))

        x1, x2 = self.xPavement[idx], self.xPavement[idx + 1]
        z1_y_col1, z2_y_col1 = self.zPavementLine1[idx], self.zPavementLine1[idx + 1]
        z1_y_col2, z2_y_col2 = self.zPavementLine2[idx], self.zPavementLine2[idx + 1]

        # Interpolate in x-direction at y_col1 and y_col2
        z_x_y_col1 = z1_y_col1 + (z2_y_col1 - z1_y_col1) * (x_input - x1) / (x2 - x1)
        z_x_y_col2 = z1_y_col2 + (z2_y_col2 - z1_y_col2) * (x_input - x1) / (x2 - x1)

        # Interpolate in y-direction to find final z value
        zpavement = z_x_y_col1 + (z_x_y_col2 - z_x_y_col1) * (y_input - self.yCoordIrregularityData[0]) / (self.yCoordIrregularityData[1] - self.yCoordIrregularityData[0])

        return zpavement

    def ConstantMovingLoads(self,dl,xyCGVehicle):
        XCGVehicle0 = xyCGVehicle[0]
        YCGVehicle0 = xyCGVehicle[1]
        P = np.zeros(self.NumberOfNodes*len(self.DOFactive))
        for wh in range(len(self.XVehicle[dl])):	# Loop in wheels
            Ptemp = np.zeros(self.NumberOfNodes*len(self.DOFactive))
            xLoad = XCGVehicle0 + self.XVehicle[dl][wh] # Position of the load
            yLoad = YCGVehicle0 + self.YVehicle[dl][wh]
            if self.xFirstNodeVehicleBeams <= xLoad and xLoad <= self.xLastNodeVehicleBeams: # The load is on the Bridge
                ks = bisect_left(self.NodeXVehicleBeams,xLoad)-1
                xNode1 = self.NodeX[ks]	# Coordinate of the start node in the loaded beam
                BeamElementLength = self.NodeX[ks+1]-xNode1
                PNode1 = (self.PVehicle[dl][wh]*(1-((xLoad-xNode1)/BeamElementLength)))
                PNode2 = self.PVehicle[dl][wh]-PNode1
                MNode1 = PNode1*yLoad
                MNode2 = PNode2*yLoad
                # Vertical loads
                Ptemp[(self.NumberOfNodes*self.indexDOFLoad)+ks] = PNode1
                Ptemp[(self.NumberOfNodes*self.indexDOFLoad)+ks+1] = PNode2
                # Torsional moments
                Ptemp[(self.NumberOfNodes*self.indexDOFMoment)+ks] = MNode1
                Ptemp[(self.NumberOfNodes*self.indexDOFMoment)+ks+1] = MNode2
                P += Ptemp

        return P

    def InitialiseVehicleReactions(self):
        F = []
        for dl in range(len(self.VVehicle)):
            if self.VehicleType[dl] == 'quarterCar':
                NumWheelsVehicle = 1 # There is one wheel in this vehicle
                NumDirectionsRections = 1 # Only vertical reaction forces
                F.append(np.zeros((NumDirectionsRections,NumWheelsVehicle))) # Vibrating system reaction forces: [vehicle order in vector self.VehicleOrderToWrite] [Rows Direction of reaction force; Cols. wheel]
        return F

    def InitialiseContactIrregularities(self):
        zpavement = []
        for dl in range(len(self.VVehicle)):
            if self.VehicleType[dl] == 'quarterCar':
                NumWheelsVehicle = 1 # There is one wheel in this vehicle
                zpavement.append(np.zeros(NumWheelsVehicle)) # Vibrating system contact irregularities: [vehicle order in vector self.VehicleOrderToWrite] [Rows wheel]
        return zpavement

    def calculateResponseAtContactPoint(self,r,dl,wh,ks,xNode1,xLoad,BeamElementLength):

        rZbridge1 = r[ks+(2*self.NumberOfNodes)] 	# Vertical displacement of the bridge in the start node of the element in which the load is located
        rZbridge2 = r[ks+1+(2*self.NumberOfNodes)] 	# Vertical displacement of the bridge in the end node of the element in which the load is located
        urXbridge1 = r[ks+(3*self.NumberOfNodes)] 	# Rotation URX of the bridge in the start node of the element in which the load is located
        urXbridge2 = r[ks+1+(3*self.NumberOfNodes)] 	# Rotation URX of the bridge in the end node of the element in which the load is located

        rZContactBridgeCentreline = rZbridge1+((xLoad-xNode1)*(rZbridge2-rZbridge1)/BeamElementLength)
        urXContactBridgeCentreline = urXbridge1+((xLoad-xNode1)*(urXbridge2-urXbridge1)/BeamElementLength)
        rZContactBridge = rZContactBridgeCentreline + urXContactBridgeCentreline*(self.YVehicle[dl][wh]+self.YCGVehicle0[dl])	# Vertical displacement of the deck at the contact point of the wheel.

        return rZContactBridge

    def SteppingMovingLoads(self,dl,xyCGVehicle,time):
        numberharmonics = len(self.Gv[dl])
        XCGVehicle0 = xyCGVehicle[0]
        YCGVehicle0 = xyCGVehicle[1]
        Ptemp = np.zeros(self.NumberOfNodes*len(self.DOFactive))
        xLoad = XCGVehicle0 + self.XVehicle[dl] # Position of the load
        yLoad = YCGVehicle0 + self.YVehicle[dl]
        if self.xFirstNodeVehicleBeams <= xLoad and xLoad <= self.xLastNodeVehicleBeams: # The load is on the Bridge
            ks = bisect_left(self.NodeXVehicleBeams,xLoad)-1
            xNode1 = self.NodeX[ks]	# Coordinate of the start node in the loaded beam
            BeamElementLength = self.NodeX[ks+1]-xNode1
            Fp = self.Gv[dl][0] + self.Gv[dl][1]*np.sin(2*np.pi*self.fmv[dl]*time)
            if numberharmonics > 2:
                for nharmonic in range(2,numberharmonics):
                    Fp += self.Gv[dl][nharmonic]*np.sin((2*np.pi*self.fmv[dl]*time)-self.phiv[dl][nharmonic])

            PNode1 = (Fp*(1-((xLoad-xNode1)/BeamElementLength)))
            PNode2 = Fp-PNode1
            MNode1 = PNode1*yLoad
            MNode2 = PNode2*yLoad
            # Vertical loads
            Ptemp[(self.NumberOfNodes*self.indexDOFLoad)+ks] = PNode1
            Ptemp[(self.NumberOfNodes*self.indexDOFLoad)+ks+1] = PNode2
            # Torsional moments
            Ptemp[(self.NumberOfNodes*self.indexDOFMoment)+ks] = MNode1
            Ptemp[(self.NumberOfNodes*self.indexDOFMoment)+ks+1] = MNode2
            ret = [Ptemp,Fp]

        else:
            ret = [0,0]

        return ret

    def initialiseVehicleResponse(self):

        DofCont = 0

        for dl in range(len(self.VVehicle)):
            if self.VehicleType[dl] == 'quarterCar':
                DofCont = DofCont + 1

        return np.zeros(DofCont)

    def VBIvertical(self,dl,xyCGVehicle,r,rdot,q,qdot,F,zpavement,ts):
        XCGVehicle0 = xyCGVehicle[0]
        YCGVehicle0 = xyCGVehicle[1]
        Ptemp = np.zeros(self.NumberOfNodes*len(self.DOFactive))
		#ContDOFVehicle = 0
		#ContWheelVehicle = 0

        if self.VehicleType[dl] == 'quarterCar':
            NumberOfWheels = 1
            NumDOFsVehicle = 1
            wh = 0  # Order of wheel in vehicle
            xLoad = XCGVehicle0 # + self.XVehicle[dl] # Position of the load
            yLoad = YCGVehicle0 # + self.YVehicle[dl]
            if self.xFirstNodeVehicleBeams <= xLoad and xLoad <= self.xLastNodeVehicleBeams: # The load is on the Bridge
                ks = bisect_left(self.NodeXVehicleBeams,xLoad)-1
                xNode1 = self.NodeX[ks]	# Coordinate of the start node in the loaded beam
                BeamElementLength = self.NodeX[ks+1]-xNode1 # Assumes deck straight in x direction

                rZContactBridge = self.calculateResponseAtContactPoint(r,dl,wh,ks,xNode1,xLoad,BeamElementLength)
                rZdotContactBridge = self.calculateResponseAtContactPoint(rdot,dl,wh,ks,xNode1,xLoad,BeamElementLength)

                # To do: add pavement irregularity

            else:
                rZContactBridge = 0.0
                rZdotContactBridge = 0.0

            # Irregularity
            dti = ts[1] - ts[0]
            zpavementn = np.zeros(NumberOfWheels)
            zdotpavement = np.zeros(NumberOfWheels)
            if self.IrregularityFlag == 'On':
                zpavementn[0] = self.InterpolateIrregularity(xyCGVehicle)
                zdotpavement[0] = (zpavementn[0] - zpavement[0])/dti  # zpavement[0] because there is only one wheel
            else:
                zpavementn[0] = 0.0
                zdotpavement[0] = 0.0

            zg = rZContactBridge + zpavementn[0]
            zgdot = rZdotContactBridge + zdotpavement[0]

            # VEHICLE RESPONSE AT THE VEHICLE LEVEL (CONSIDERING ALL THE WHEELS)
            i0ls = np.zeros(NumDOFsVehicle*2)
            contdofi = 0
            for dofv in range(NumDOFsVehicle):
                i0ls[contdofi] = q
                i0ls[contdofi+1] = qdot
                contdofi = contdofi + 2

            # roughnessAtWheels = [[zg,zg],[zgdot,zgdot]], zg and zgdot are doubled to make the general loop work for any vehicle, but the second components of each element in this variable are not used
            #roughnessAtWheels = [[zg,zg],[zgdot,zgdot]]
            qlint = odeint(SDOFUnsprungMass,i0ls,ts,args=(self.MassVehicles[dl][0],self.DampingVehicles[dl][0],self.StiffnessVehicles[dl][0],zg,zgdot,),mxstep=5000000)
            q = qlint[-1,0]
            qdot = qlint[-1,1]

            # Reaction forces at the bridge

            FzDynamic = self.StiffnessVehicles[dl][0]*(q-zg) + self.DampingVehicles[dl][0]*(qdot-zgdot) # Positive force related to a movement upwards of the vehicle wheel mass and negative movement of the road #### Vertical EFFECT OF THE VEHICLE ON THE BRIDGE, global coordinates of the bridge
            # To do!!! Longitudinal effects (traction, rolling resistance, bracking effects ...) Introduce FxDynamic and MyyDynamic
            FzTotal = self.FzStatic[dl] + FzDynamic  # GLOBAL coordinates of the bridge

            F[0,0] = FzTotal # Output of vehicle reactions [direction,wheel]

            if self.xFirstNodeVehicleBeams <= xLoad and xLoad <= self.xLastNodeVehicleBeams: # The load is on the Bridge

                PNode1 = (FzTotal*(1-((xLoad-xNode1)/BeamElementLength)))
                PNode2 = FzTotal-PNode1
                MNode1 = PNode1*yLoad
                MNode2 = PNode2*yLoad
                # Vertical loads
                Ptemp[(self.NumberOfNodes*self.indexDOFLoad)+ks] = PNode1
                Ptemp[(self.NumberOfNodes*self.indexDOFLoad)+ks+1] = PNode2
                # Torsional moments
                Ptemp[(self.NumberOfNodes*self.indexDOFMoment)+ks] = MNode1
                Ptemp[(self.NumberOfNodes*self.indexDOFMoment)+ks+1] = MNode2

        return [Ptemp,q,qdot,F,zpavementn,zg]


class HarmonicLoads():

    def __init__(self, **kwargs):
        #self.t = kwargs.get("t")
        self.DOFactive = kwargs.get("DOFactive")
        self.modesToInclude = kwargs.get("modesToInclude")
        self.NumberOfNodes = kwargs.get("NumberOfNodes")
        self.Phi = kwargs.get("Phi")
        self.wnv = kwargs.get("wnv")
        self.xinv = kwargs.get("xinv")
        self.beta = kwargs.get("beta")
        self.gamma = kwargs.get("gamma")
        self.NodeX = kwargs.get("NodeX")
        self.NodeNumber = kwargs.get("NodeNumber")
        self.BeamNode1 = kwargs.get("BeamNode1")
        self.BeamNode2 = kwargs.get("BeamNode2")
        self.BeamLength = kwargs.get("BeamLength")
        self.BeamNumber = kwargs.get("BeamNumber")

        self.PHarmonicLoad = kwargs.get("PHarmonicLoad")
        self.wHarmonicLoad = kwargs.get("wHarmonicLoad")
        self.NodeHarmonicLoad = kwargs.get("NodeHarmonicLoad")
        self.DofHarmonicLoad = kwargs.get("DofHarmonicLoad")

    def harmonicFixedLoad(self,time,P,dl):

        #for dl in range(len(self.PHarmonicLoad)):		 # Loop in loads
        index0=np.where(self.NodeNumber==self.NodeHarmonicLoad[dl])[0][0]
        loadAmplitude = self.PHarmonicLoad[dl]*np.sin(self.wHarmonicLoad[dl]*time)
        P[(self.NumberOfNodes*(self.DofHarmonicLoad[dl]-1))+index0] = loadAmplitude
        return [P,loadAmplitude]



class SeismicLoads():

    def __init__(self, **kwargs):
        #self.t = kwargs.get("t")
        self.DOFactive = kwargs.get("DOFactive")
        self.modesToInclude = kwargs.get("modesToInclude")
        self.NumberOfNodes = kwargs.get("NumberOfNodes")
        self.Phi = kwargs.get("Phi")
        self.wnv = kwargs.get("wnv")
        self.xinv = kwargs.get("xinv")
        self.beta = kwargs.get("beta")
        self.gamma = kwargs.get("gamma")
        self.NodeX = kwargs.get("NodeX")
        self.NodeNumber = kwargs.get("NodeNumber")
        self.BeamNode1 = kwargs.get("BeamNode1")
        self.BeamNode2 = kwargs.get("BeamNode2")
        self.BeamLength = kwargs.get("BeamLength")
        self.BeamNumber = kwargs.get("BeamNumber")

        self.GammaInformation = kwargs.get("GammaInformation")
        self.accel_t = kwargs.get("accel_t")
        self.accel_X = kwargs.get("accel_X")
        self.accel_Y = kwargs.get("accel_Y")
        self.accel_Z = kwargs.get("accel_Z")

        # Interpolate accelerogram at step times of analysis
        self.axinterp1d = interp1d(self.accel_t,self.accel_X)
        self.ayinterp1d = interp1d(self.accel_t,self.accel_Y)
        self.azinterp1d = interp1d(self.accel_t,self.accel_Z)

    def ground_acceleration(self,time):
        ax = self.axinterp1d(time)
        ay = self.ayinterp1d(time)
        az = self.azinterp1d(time)
        ag = [ax,ay,az]

        return ag

    def seismic_mrha_sync(self,ag):
        ax = ag[0]
        ay = ag[1]
        az = ag[2]
        Pn = (-1*self.GammaInformation[:,0]*ax) + (-1*self.GammaInformation[:,1]*ay) + (-1*self.GammaInformation[:,2]*az)

        return Pn

class WindLoads():

    def __init__(self, **kwargs):
        #self.t = kwargs.get("t")
        self.DOFactive = kwargs.get("DOFactive")
        self.modesToInclude = kwargs.get("modesToInclude")
        self.NumberOfNodes = kwargs.get("NumberOfNodes")
        self.Phi = kwargs.get("Phi")
        self.wnv = kwargs.get("wnv")
        self.xinv = kwargs.get("xinv")
        self.beta = kwargs.get("beta")
        self.gamma = kwargs.get("gamma")
        self.NodeX = kwargs.get("NodeX")
        self.NodeZ = kwargs.get("NodeZ")

        self.NodeNumber = kwargs.get("NodeNumber")
        self.BeamNode1 = kwargs.get("BeamNode1")
        self.BeamNode2 = kwargs.get("BeamNode2")
        self.BeamLength = kwargs.get("BeamLength")
        self.BeamNumber = kwargs.get("BeamNumber")

        self.WindBeams = kwargs.get("WindBeams")
        self.node_windGeneration = kwargs.get("node_windGeneration")
        self.X_node_windGeneration = self.node_windGeneration[:,1]
        self.crossSections = kwargs.get("crossSections")
        self.dimensionsCrossSections = kwargs.get("dimensionsCrossSections")


        #self.U = kwargs.get("U")
        #self.Usq = self.U*self.U
        self.ut = kwargs.get("ut")
        self.vt = kwargs.get("vt")
        self.wt = kwargs.get("wt")
        self.tramp = kwargs.get("tramp")
        self.rho = kwargs.get("rho")
        self.m1 = kwargs.get("m1")  # Relative position of aerodynamic centre in section. Total distance from centroid: m1*B*0.5, positive leeward
                                    # m1 = 0.5, the downward velocity at the leeward three-quarter-chord point is selected
                                    #           for the calculation of the effective angle of incidence as that for airfoil section
                                    #           It creates problems when Cmder > 0, because then A2* > 0 (flutter).
                                    # m1 = -0.5, the downward velocity at the forward three-quarter-chord point (Miyata et al. 1995)
                                    #           It may lead to inconsistency in the sign of H2* and P2*
                                    # m1 = 0 in Strommen
        self.Umean = kwargs.get("Umean")
        self.Nr = kwargs.get("Nr")
        self.Nrb = kwargs.get("Nrb")

        self.dratapprox = kwargs.get("dratapprox")

        self.MovingWindow = kwargs.get("MovingWindow")
        self.turbulentFlag = kwargs.get("turbulentFlag")
        self.IntegrationLUConvolution = kwargs.get("IntegrationLUConvolution")

        # Index of the active DOF in which the load and moments are applied.
        self.indexDOFZLoad=self.DOFactive.index(3)		# Vertical load
        self.indexDOFMomentXX=self.DOFactive.index(4)	# Torsional moment
        self.indexDOFYLoad=self.DOFactive.index(2)		# Lateral load

        self.alphaCi = self.crossSections[0][:,0]*np.pi/180 # Angles of attack for which Aerodynamic coefficients are given, in rad
        self.tmax = kwargs.get("tmax")
        self.dt = kwargs.get("dt")
        self.time = np.arange(0.,self.tmax,self.dt)

        self.WindSectionLabel = self.WindBeams[:,1]
        self.WindSectionLabel = np.array(self.WindSectionLabel).astype(int)

        self.B = self.WindBeams[:,2]
        self.D = self.WindBeams[:,3]

        # Find X coordinates of wind nodes
        WindBeamOrder = [index for index, element in enumerate(self.BeamNumber) if element in self.WindBeams[:,0]]
        WindBeamNodes = list(self.BeamNode1[WindBeamOrder])
        WindBeamNodes.append(self.BeamNode2[int(WindBeamOrder[-1])])
        self.WindBeamNodes = [int(x) for x in WindBeamNodes]
        self.WindBeamLength = self.BeamLength[WindBeamOrder]

        self.WindBeamLengthforNodes = np.zeros(len(self.WindBeamLength)+1)
        for i in range(len(self.WindBeamLengthforNodes)):
            if i == 0:
                self.WindBeamLengthforNodes[i] = 0.5*self.WindBeamLength[i]
            elif i== len(self.WindBeamLengthforNodes) - 1:
                self.WindBeamLengthforNodes[i] = 0.5*self.WindBeamLength[-1]
            else:
                self.WindBeamLengthforNodes[i] = 0.5*(self.WindBeamLength[i-1] + self.WindBeamLength[i])

        self.indexWindBeamNodes = [index for index, element in enumerate(self.NodeNumber) if element in self.WindBeamNodes]

        self.indexYDOFatWindBeamNodes = list((self.NumberOfNodes*self.indexDOFYLoad)+np.array(self.indexWindBeamNodes))
        self.indexZDOFatWindBeamNodes = list((self.NumberOfNodes*self.indexDOFZLoad)+np.array(self.indexWindBeamNodes))
        self.indexXXDOFatWindBeamNodes = list((self.NumberOfNodes*self.indexDOFMomentXX)+np.array(self.indexWindBeamNodes))

        self.NodeXWindBeams = self.NodeX[self.indexWindBeamNodes]   # X coordinates of nodes in which wind is applied

        self.numberWindBeams = len(self.B)


        if self.turbulentFlag == 1:
        # Time in which the wind is defined in each direction (it may differ slightly)
            self.tWindU = self.ut[:,0]
            self.tWindV = self.vt[:,0]
            self.tWindW = self.wt[:,0]

            self.ut = self.ut[:,1:]
            self.vt = self.vt[:,1:]
            self.wt = self.wt[:,1:]

            # Interpolate wind histories at nodes of the structure
            uinterp1d = interp1d(self.X_node_windGeneration,np.transpose(self.ut),axis=0)
            vinterp1d = interp1d(self.X_node_windGeneration,np.transpose(self.vt),axis=0)
            winterp1d = interp1d(self.X_node_windGeneration,np.transpose(self.wt),axis=0)

            self.uinterpol = uinterp1d(self.NodeXWindBeams) # [wind nodes x time where wind is generated]
            self.vinterpol = vinterp1d(self.NodeXWindBeams)
            self.winterpol = winterp1d(self.NodeXWindBeams)

            # Obtain average wind at centre of beam elements
            sizetime = np.shape(self.uinterpol)[1]
            self.uinterpolBeam = np.zeros((self.numberWindBeams,sizetime))
            self.vinterpolBeam = np.zeros((self.numberWindBeams,sizetime))
            self.winterpolBeam = np.zeros((self.numberWindBeams,sizetime))

        self.numberWindNodes = len(self.indexWindBeamNodes)

        self.UmeanBeam = np.zeros(len(self.B))

        for i in range(len(self.indexWindBeamNodes)-1):
            #if i == 0 or i == len(self.indexWindBeamNodes)-2:
            if self.turbulentFlag == 1:
                self.uinterpolBeam[i,:] = (self.uinterpol[i,:]+self.uinterpol[i+1,:])*0.5
                self.vinterpolBeam[i,:] = (self.vinterpol[i,:]+self.vinterpol[i+1,:])*0.5
                self.winterpolBeam[i,:] = (self.winterpol[i,:]+self.winterpol[i+1,:])*0.5
            self.UmeanBeam[i] = (self.Umean[i]+self.Umean[i+1])*0.5

        # Interpolate wind turbulence at all time instants of the analysis, for LU buffeting
        if self.turbulentFlag == 1:
            uinterp1dtimeanalysis = interp1d(self.tWindU,np.transpose(self.uinterpolBeam),axis=0)
            winterp1dtimeanalysis = interp1d(self.tWindW,np.transpose(self.winterpolBeam),axis=0)
            Utottimeanalysis = np.transpose(uinterp1dtimeanalysis(self.time))
            self.utimeanalysis = np.zeros(np.shape(Utottimeanalysis))
            for i in range(len(self.indexWindBeamNodes)-1):
                self.utimeanalysis[i,:]=Utottimeanalysis[i,:]-self.UmeanBeam[i]
            self.wtimeanalysis = np.transpose(winterp1dtimeanalysis(self.time))

            # Derivative in time of wind velocity for buffeting with rational approximation
            # compute vector of forward differences
            udotwind = np.diff(self.utimeanalysis)/self.dt
            wdotwind = np.diff(self.wtimeanalysis)/self.dt
            # copy last value to make length same as original signal
            self.udotthistory = np.hstack((udotwind, np.tile(udotwind[:, [-1]], 1)))
            self.wdotthistory = np.hstack((wdotwind, np.tile(wdotwind[:, [-1]], 1)))


        self.HalfRhoUmeanB = 0.5*self.rho*self.UmeanBeam*self.B
        self.HalfRhoUmean = 0.5*self.rho*self.UmeanBeam
        self.HalfRhoUmeanSqB = 0.5*self.rho*(self.UmeanBeam**2)*self.B
        self.HalfRhoUmeanSq = 0.5*self.rho*(self.UmeanBeam**2)
        self.HalfRhoUmeanSqBSq = 0.5*self.rho*(self.UmeanBeam**2)*(self.B**2)
        self.HalfRho = 0.5*self.rho
        self.DoverB = self.D/self.B
        self.BoverU = self.B/self.UmeanBeam
        self.BoverUSq = (self.B/self.UmeanBeam)**2
        self.BSq = (self.B)**2

        self.CdMatrix = np.zeros((self.numberWindBeams,len(self.alphaCi))) # [wind beam x alphaCi]
        self.ClMatrix = np.zeros((self.numberWindBeams,len(self.alphaCi))) # [wind beam x alphaCi]
        self.CmMatrix = np.zeros((self.numberWindBeams,len(self.alphaCi))) # [wind beam x alphaCi]
        self.CdDerMatrix = np.zeros((self.numberWindBeams,len(self.alphaCi))) # [wind beam x alphaCi]
        self.ClDerMatrix = np.zeros((self.numberWindBeams,len(self.alphaCi))) # [wind beam x alphaCi]
        self.CmDerMatrix = np.zeros((self.numberWindBeams,len(self.alphaCi))) # [wind beam x alphaCi]

        for i in range(self.numberWindBeams):
            self.CdMatrix[i,:] = self.crossSections[self.WindSectionLabel[i]-1][:,1]
            self.ClMatrix[i,:] = self.crossSections[self.WindSectionLabel[i]-1][:,2]
            self.CmMatrix[i,:] = self.crossSections[self.WindSectionLabel[i]-1][:,3]
            self.CdDerMatrix[i,:] = self.crossSections[self.WindSectionLabel[i]-1][:,4]
            self.ClDerMatrix[i,:] = self.crossSections[self.WindSectionLabel[i]-1][:,5]
            self.CmDerMatrix[i,:] = self.crossSections[self.WindSectionLabel[i]-1][:,6]

        #self.Cd_interp = RegularGridInterpolator((self.alphaCi,range(self.numberWindBeams)),self.CdMatrix)
        self.Cd_interp = interp1d(self.alphaCi, self.CdMatrix,axis=1)
        self.Cl_interp = interp1d(self.alphaCi, self.ClMatrix,axis=1)
        self.Cm_interp = interp1d(self.alphaCi, self.CmMatrix,axis=1)

        self.CdDer_interp = interp1d(self.alphaCi, self.CdDerMatrix,axis=1)
        self.ClDer_interp = interp1d(self.alphaCi, self.ClDerMatrix,axis=1)
        self.CmDer_interp = interp1d(self.alphaCi, self.CmDerMatrix,axis=1)





    def AverageAtBeamCenter(self,r):  # average of wind nodal values at wind beam center
        rBeam = np.zeros(self.numberWindBeams)
        for i in range(len(r)-1):
            rBeam[i] = (r[i]+r[i+1])*0.5
        return rBeam

    def LumpedAtBeamNodes(self,r): # Lump the response or action at nodes of wind beams
        rhalf = r*0.5
        rNodes1 = np.zeros(self.numberWindNodes)
        rNodes2 = np.zeros(self.numberWindNodes)
        rNodes1[:-1] = rhalf
        rNodes2[1:] = rhalf
        rNodes = rNodes1 + rNodes2
        return rNodes

    def InterpolateWind(self,t):    # Interpolate wind at time of analysis and wind beams of the structure

        # Interpolate wind histories at each time step of the analysis
        uinterp1d = interp1d(self.tWindU,self.uinterpolBeam,axis=1)
        #vinterp1d = interp1d(self.tWindV,self.vinterpolBeam,axis=1)
        winterp1d = interp1d(self.tWindW,self.winterpolBeam,axis=1)
        uturbulent = np.subtract(uinterp1d(t).T,self.UmeanBeam).T # Wind element x time
        #vturbulent = vinterp1d(t)
        wturbulent = winterp1d(t)

        return uturbulent,wturbulent

    def Cint(self,r):   # Aerodynamic coefficients interpolated along Wind beams for given displacement of whole structure

        alphaWindNodes = -1*r[self.indexXXDOFatWindBeamNodes]   #-1* because the coefficients are given with positive angles nose down, and in MDyn positive is nose up for the deck
        #print(alphaWindNodes)
        alphaWindBeams = self.AverageAtBeamCenter(alphaWindNodes)
        # CHANGE INTERPOLATION, IT DOES UNNECESSARY OPS
        Cd = self.Cd_interp(alphaWindBeams)[0]
        Cl = self.Cl_interp(alphaWindBeams)[0]
        Cm = self.Cm_interp(alphaWindBeams)[0]

        return Cd, Cl, Cm

    def CDerint(self,r):   # Angle derivatives of aerodynamic coefficients interpolated along Wind beams for given displacement of whole structure

        alphaWindNodes = -1*r[self.indexXXDOFatWindBeamNodes]
        #print(alphaWindNodes)
        alphaWindBeams = self.AverageAtBeamCenter(alphaWindNodes)
        # CHANGE INTERPOLATION, IT DOES UNNECESSARY OPS
        self.CdDer = self.CdDer_interp(alphaWindBeams)[0]
        self.ClDer = self.ClDer_interp(alphaWindBeams)[0]
        self.CmDer = self.CmDer_interp(alphaWindBeams)[0]

        return self.CdDer, self.ClDer, self.CmDer

    def CintAlpha(self,alphaWindBeams):   # Aerodynamic coefficients interpolated along Wind beams for given alpha at wind beams

        # CHANGE INTERPOLATION, IT DOES UNNECESSARY OPS
        Cd = self.Cd_interp(alphaWindBeams)[0]
        Cl = self.Cl_interp(alphaWindBeams)[0]
        Cm = self.Cm_interp(alphaWindBeams)[0]

        return Cd, Cl, Cm

    def CDerintAlpha(self,alphaWindBeams):   # Angle-derivatives of aerodynamic coefficients interpolated along Wind beams for given alpha at wind beams

        # CHANGE INTERPOLATION, IT DOES UNNECESSARY OPS
        self.CdDer = self.CdDer_interp(alphaWindBeams)[0]
        self.ClDer = self.ClDer_interp(alphaWindBeams)[0]
        self.CmDer = self.CmDer_interp(alphaWindBeams)[0]

    def rotmax(self,r): # Obtain the maximum rotation of wind beam Nodes

        alphaWindNodes = r[self.indexXXDOFatWindBeamNodes]
        return max(abs(alphaWindNodes))

    def rotWindBeamNodes(self,r): # Obtain the maximum rotation of wind beam Nodes

        alphaWindNodes = r[self.indexXXDOFatWindBeamNodes]
        return alphaWindNodes


    def WindQuasiStaticLoad(self,r):

        P = np.zeros(self.NumberOfNodes*len(self.DOFactive))

        # Convert r to wind positive and interpolate Ci at static angle of rotation
        Cd, Cl, Cm = self.Cint(r)

        # Wind forces at the beam centres, in N or Nm, according to convention in Chen and Kareem
        Dst = self.HalfRhoUmeanSqB*Cd*self.WindBeamLength       # Positive windward
        Lst = -1*self.HalfRhoUmeanSqB*Cl*self.WindBeamLength    # Positive downwards
        Mst = self.HalfRhoUmeanSqBSq*Cm*self.WindBeamLength     # Positive nose up

        # Wind forces in MDyn convention
        qYst = Dst # Positive +Y (windward)
        qZst = -1*Lst # Positive +Z (upwards)
        qXXst = -1*Mst # Positive +XX (nose down)

        # Lump force at beam Nodes
        qYstNodes = self.LumpedAtBeamNodes(qYst)
        qZstNodes = self.LumpedAtBeamNodes(qZst)
        qXXstNodes = self.LumpedAtBeamNodes(qXXst)

        # Expand to the global DOF of the structure
        P[self.indexYDOFatWindBeamNodes] = qYstNodes
        P[self.indexZDOFatWindBeamNodes] = qZstNodes
        P[self.indexXXDOFatWindBeamNodes] = qXXstNodes

        return P


    def WindBuffetingLoadLQS(self,uturbulent_t,wturbulent_t,Cdbar,Clbar,Cmbar):
        P = np.zeros(self.NumberOfNodes*len(self.DOFactive))

        utoverU = uturbulent_t / self.UmeanBeam
        wtoverU = wturbulent_t / self.UmeanBeam

        # Wind forces at the beam centres, in N or Nm, according to convention in Chen and Kareem
        Db = self.HalfRhoUmeanSqB*((2*Cdbar*utoverU)+((self.CdDer-Clbar)*wtoverU))*self.WindBeamLength       # Positive windward
        Lb = -1*self.HalfRhoUmeanSqB*((2*Clbar*utoverU)+((self.ClDer+Cdbar)*wtoverU))*self.WindBeamLength    # Positive downwards
        Mb = self.HalfRhoUmeanSqBSq*((2*Cmbar*utoverU)+(self.CmDer*wtoverU))*self.WindBeamLength     # Positive nose up

        # Wind forces in MDyn convention
        qY = Db # Positive +Y (windward)
        qZ = -1*Lb # Positive +Z (upwards)
        qXX = -1*Mb # Positive +XX (nose down)

        # Lump force at beam Nodes
        qYNodes = self.LumpedAtBeamNodes(qY)
        qZNodes = self.LumpedAtBeamNodes(qZ)
        qXXNodes = self.LumpedAtBeamNodes(qXX)

        # Expand to the global DOF of the structure
        P[self.indexYDOFatWindBeamNodes] = qYNodes
        P[self.indexZDOFatWindBeamNodes] = qZNodes
        P[self.indexXXDOFatWindBeamNodes] = qXXNodes

        return P

    def WindAeroelasticLoadLQS(self,r,rdot,Cdbar,Clbar,Cmbar):
        P = np.zeros(self.NumberOfNodes*len(self.DOFactive))

        # Movements and velocities of wind beam nodes in previous analysis step, in wind positive convention
        pNodes = r[self.indexYDOFatWindBeamNodes]
        pdotNodes = rdot[self.indexYDOFatWindBeamNodes]
        hNodes = -1*r[self.indexZDOFatWindBeamNodes]           # Positive downwards
        hdotNodes = -1*rdot[self.indexZDOFatWindBeamNodes]     # Positive downwards
        alphaNodes = -1*r[self.indexXXDOFatWindBeamNodes]         # Positive nose up
        alphadotNodes = -1*rdot[self.indexXXDOFatWindBeamNodes]   # Positive nose up

        # Movements and velocities of wind beam centers in previous analysis step
        p = self.AverageAtBeamCenter(pNodes)
        pdot = self.AverageAtBeamCenter(pdotNodes)
        h = self.AverageAtBeamCenter(hNodes)
        hdot = self.AverageAtBeamCenter(hdotNodes)
        alpha = self.AverageAtBeamCenter(alphaNodes)
        alphadot = self.AverageAtBeamCenter(alphadotNodes)

        pdotoveru = pdot/self.UmeanBeam
        rotvelocityoveru = (hdot+(self.m1*self.B*0.5*alphadot))/self.UmeanBeam

        # Wind forces at the beam centres, in N or Nm, according to convention in Chen and Kareem
        Dse = self.HalfRhoUmeanSqB*((-2*Cdbar*pdotoveru)+(self.CdDer*alpha)+((self.CdDer-Clbar)*rotvelocityoveru))*self.WindBeamLength       # Positive windward
        Lse = self.HalfRhoUmeanSqB*((-1*(self.ClDer+Cdbar)*rotvelocityoveru)-(self.ClDer*alpha)+(2*Clbar*pdotoveru))*self.WindBeamLength       # Positive downwards
        Mse = self.HalfRhoUmeanSqBSq*((self.CmDer*rotvelocityoveru)+(self.CmDer*alpha)-(2*Cmbar*pdotoveru))*self.WindBeamLength     # Positive nose up

        # Wind forces in MDyn convention
        qY = Dse # Positive +Y (windward)
        qZ = -1*Lse # Positive +Z (upwards)
        qXX = -1*Mse # Positive +XX (nose down)

        # Lump force at beam Nodes
        qYNodes = self.LumpedAtBeamNodes(qY)
        qZNodes = self.LumpedAtBeamNodes(qZ)
        qXXNodes = self.LumpedAtBeamNodes(qXX)

        # Expand to the global DOF of the structure
        P[self.indexYDOFatWindBeamNodes] = qYNodes
        P[self.indexZDOFatWindBeamNodes] = qZNodes
        P[self.indexXXDOFatWindBeamNodes] = qXXNodes

        return P
