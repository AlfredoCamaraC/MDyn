# -*- coding: utf-8 -*-
"""
Created by Alfredo Camara Casado
For extra capabilities of the code, code development, and bug reporting please contact the author at: acamara@ciccp.es
"""

import numpy as np
#from scipy.interpolate import interp1d
from scipy.integrate import odeint
#import time
import os
from subprocess import call
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v3 as iio
from scipy.interpolate import interp1d
from scipy.optimize import leastsq, least_squares

class ModalMatrices():

	def __init__(self, **kwargs):
        #self.t = kwargs.get("t")
		self.modesToInclude = kwargs.get("modesToInclude")
		self.Phi = kwargs.get("Phi")
		self.wnv = kwargs.get("wnv")
		self.xinv = kwargs.get("xinv")

	def filterModalMatrices(self):
		self.PhiReduced = np.array(self.Phi)[:,np.array(self.modesToInclude)-1]
		self.wnvReduced = np.array(self.wnv)[np.array(self.modesToInclude)-1]
		self.xinvReduced = np.array(self.xinv)[np.array(self.modesToInclude)-1]

		# Modal matrix
		M = np.ones(len(self.modesToInclude))		# Modal mass matrix, with ones if modes are normalised wrt to mass
		C = 2*np.multiply(self.xinvReduced,self.wnvReduced)	# Modal damping matrix, 2*xi*w for each mode
		K = np.multiply(self.wnvReduced,self.wnvReduced)		# Modal stiffness matrix, w**2 for each mode

		# Reshape
		M = np.array(M)
		C = np.array(C)
		K = np.array(K)

		M.shape = (len(self.modesToInclude),1)
		C.shape = (len(self.modesToInclude),1)
		K.shape = (len(self.modesToInclude),1)

		return M,C,K,self.PhiReduced

	def initialiseModalVectors(self):

		u = np.zeros((len(self.modesToInclude),1))					# Modal displacement (Number of modes x 1)
		udot = np.zeros((len(self.modesToInclude),1))				# Modal velocity (Number of modes x 1)

		Pn0 = np.zeros(len(self.modesToInclude)) # Initialise modal force
		Pn0.shape = u.shape
		#u2dot = np.zeros((len(self.modesToInclude),1))				# Modal acceleration (Number of modes x 1)

		return Pn0,u,udot

class PlotModel():
	def __init__(self, **kwargs):
        #self.t = kwargs.get("t")
		self.modesToPlot = kwargs.get("modesToPlot")
		self.scaleFactorPlot = kwargs.get("scaleFactorPlot")
		self.pathToPlots = kwargs.get("pathToPlots")
		self.NumberOfNodes = kwargs.get("NumberOfNodes")
		self.NodeNumber = kwargs.get("NodeNumber")
		self.NodeX = kwargs.get("NodeX")
		self.NodeY = kwargs.get("NodeY")
		self.NodeZ = kwargs.get("NodeZ")
		self.BeamNode1 = kwargs.get("BeamNode1")
		self.BeamNode2 = kwargs.get("BeamNode2")
		self.Phi = kwargs.get("Phi")
		self.wnv = kwargs.get("wnv")
		self.xinv = kwargs.get("xinv")
		self.script_dir = os.getcwd() # os.path.join( os.path.dirname( __file__ ), '..' )
		self.caseName = kwargs.get("caseName")
		self.scaleFactorAnimation = kwargs.get("scaleFactorAnimation")

		self.TMDNodes = kwargs.get("TMDNodes")
		self.TMDDirection = kwargs.get("TMDDirection")
		self.TMDVisual = kwargs.get("TMDVisual")

		# VBI
		self.YVehicle = kwargs.get("YVehicle")
		self.XVehicle = kwargs.get("XVehicle")
		self.XCGVehicle0 = kwargs.get("XCGVehicle0")
		self.YCGVehicle0 = kwargs.get("YCGVehicle0")
		self.VehicleType = kwargs.get("VehicleType")
		self.VehicleVisual = kwargs.get("VehicleVisual")

		# Index of the active DOF in which the TMD is applied
		#self.indexDOFTMD = np.zeros(len(self.TMDNodes))

		if self.TMDNodes is not None:
			self.indexNodeTMD = np.zeros(len(self.TMDNodes))
			for i in range(len(self.TMDNodes)):
				#self.indexDOFTMD[i] = self.DOFactive.index(self.TMDDirection[i])
				self.indexNodeTMD[i] = np.where(self.NodeNumber==self.TMDNodes[i])[0][0]
			#self.indexDOFTMD = np.array(self.indexDOFTMD).astype(int)
			self.indexNodeTMD = np.array(self.indexNodeTMD).astype(int)
			self.xCoordTMD = self.NodeX[self.indexNodeTMD]
			self.yCoordTMD = self.NodeY[self.indexNodeTMD]
			self.zCoordTMD = self.NodeZ[self.indexNodeTMD]

		xdimEstimate = np.ptp(self.NodeX)
		ydimEstimate = np.ptp(self.NodeY)
		zdimEstimate = np.ptp(self.NodeZ)
		maxdimEstimate = max([xdimEstimate,ydimEstimate,zdimEstimate])
		if xdimEstimate == 0: xdimEstimate = maxdimEstimate/10
		if ydimEstimate == 0: ydimEstimate = maxdimEstimate/10
		if zdimEstimate == 0: zdimEstimate = maxdimEstimate/10

		self.xdimEstimate = xdimEstimate
		self.ydimEstimate = ydimEstimate
		self.zdimEstimate = zdimEstimate
		self.dimensionPlot = maxdimEstimate

		self.NodesEarthquake = kwargs.get("NodesEarthquake")
		self.DirectionsEarthquake = kwargs.get("DirectionsEarthquake")

	def plotUndeformedModel(self):
		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		ax.title.set_text('Undeformed model')
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [self.NodeX[orderNode1],self.NodeX[orderNode2]]
			y = [self.NodeY[orderNode1],self.NodeY[orderNode2]]
			z = [self.NodeZ[orderNode1],self.NodeZ[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)
		ax.set_xlabel('$X$ [m]')
		ax.set_ylabel('$Y$ [m]')
		ax.set_zlabel('$Z$ [m]')
		ax.set_xticks(ax.get_xticks()[::10])
		ax.set_yticks([])
		ax.set_zticks(ax.get_zticks()[::2])
		#plt.show()
		results_dir = os.path.join(self.script_dir, 'preprocessing/')
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = "undeformed_model.pdf"#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()

	def plotMode(self,ModeToPlot):
		PhiMode = np.array(self.Phi)[:,np.array(ModeToPlot)-1]
		ux = PhiMode[:self.NumberOfNodes]
		uy = PhiMode[self.NumberOfNodes:self.NumberOfNodes*2]
		uz = PhiMode[self.NumberOfNodes*2:self.NumberOfNodes*3]
		urx = PhiMode[self.NumberOfNodes*3:self.NumberOfNodes*4]
		ury = PhiMode[self.NumberOfNodes*4:self.NumberOfNodes*5]
		urz = PhiMode[self.NumberOfNodes*5:]
		maxux = max(abs(ux))
		maxuy = max(abs(uy))
		maxuz = max(abs(uz))
		maxdisp = max([maxux,maxuy,maxuz])
		#dimensionPlot=max([np.ptp(self.NodeX), np.ptp(self.NodeY), np.ptp(self.NodeZ)])
		scaleFactorPlot = 0.05*self.dimensionPlot/maxdisp
		NodeXdef = self.NodeX+ux*scaleFactorPlot
		NodeYdef = self.NodeY+uy*scaleFactorPlot
		NodeZdef = self.NodeZ+uz*scaleFactorPlot
		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		ax.title.set_text('Mode '+str(ModeToPlot)+'. Frequency:'+str((round(self.wnv[ModeToPlot-1]/(2*np.pi),2)))+' Hz. Damping: '+ str(self.xinv[ModeToPlot-1]*100) +'%')
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [NodeXdef[orderNode1],NodeXdef[orderNode2]]
			y = [NodeYdef[orderNode1],NodeYdef[orderNode2]]
			z = [NodeZdef[orderNode1],NodeZdef[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)
		# Add undeformed model
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [self.NodeX[orderNode1],self.NodeX[orderNode2]]
			y = [self.NodeY[orderNode1],self.NodeY[orderNode2]]
			z = [self.NodeZ[orderNode1],self.NodeZ[orderNode2]]
			ax.plot(x, y, z, c='gray', linewidth=0.5)
		ax.set_axis_off()
		#plt.show()
		results_dir = os.path.join(self.script_dir, 'preprocessing/')
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = 'mode_'+str(ModeToPlot)+'.pdf'#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()

	def plotDeformedStructure(self,it,t,r):
		ux = r[:self.NumberOfNodes]
		uy = r[self.NumberOfNodes:self.NumberOfNodes*2]
		uz = r[self.NumberOfNodes*2:self.NumberOfNodes*3]

		NodeXdef = self.NodeX+ux*self.scaleFactorAnimation
		NodeYdef = self.NodeY+uy*self.scaleFactorAnimation
		NodeZdef = self.NodeZ+uz*self.scaleFactorAnimation
		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [NodeXdef[orderNode1],NodeXdef[orderNode2]]
			y = [NodeYdef[orderNode1],NodeYdef[orderNode2]]
			z = [NodeZdef[orderNode1],NodeZdef[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)

		# Setting the axes properties
		if min(self.NodeX) != max(self.NodeX):
			ax.set(xlim3d=(min(self.NodeX)*1.2, max(self.NodeX)*1.2))
		else:
			ax.set(xlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeY) != max(self.NodeY):
			ax.set(ylim3d=(min(self.NodeY)*1.2, max(self.NodeY)*1.2))
		else:
			ax.set(ylim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeZ) != max(self.NodeZ):
			ax.set(zlim3d=(min(self.NodeZ)*1.2, max(self.NodeZ)*1.2))
		else:
			ax.set(zlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))

		ax.title.set_text('Deformation scale factor: '+str(round(self.scaleFactorAnimation,1)))
		ax.text(max(self.NodeX)*1.1, min(self.NodeY)*1.1, min(self.NodeZ)*1.1, 't = '+str(round(t,2))+'s')
		ax.set_axis_off()
		#plt.show()
		results_dir = os.path.join(self.script_dir, self.caseName+"/deformation_animation/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = 'deformation_iteration_'+str(it)+'.png'#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()


	def plotDeformedStructureEarthquake(self,it,t,r,ag):
		ux = r[:self.NumberOfNodes]
		uy = r[self.NumberOfNodes:self.NumberOfNodes*2]
		uz = r[self.NumberOfNodes*2:self.NumberOfNodes*3]

		NodeXdef = self.NodeX+ux*self.scaleFactorAnimation
		NodeYdef = self.NodeY+uy*self.scaleFactorAnimation
		NodeZdef = self.NodeZ+uz*self.scaleFactorAnimation
		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [NodeXdef[orderNode1],NodeXdef[orderNode2]]
			y = [NodeYdef[orderNode1],NodeYdef[orderNode2]]
			z = [NodeZdef[orderNode1],NodeZdef[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)

		# Plot accelerograma arrows
		arrowDimension = self.dimensionPlot/50
		# x Direction
		for i in range(len(self.NodesEarthquake)): # loop in nodes with ground motion applied
			NodeOrder = NodeNumberCList.index(int(self.NodesEarthquake[i]))
			xNode1 = NodeXdef[NodeOrder]
			yNode1 = NodeYdef[NodeOrder]
			zNode1 = NodeZdef[NodeOrder]

			if self.DirectionsEarthquake[i,0] != 0:	# Earthquake applied in x direction
				xNode2 = xNode1 + ag[0]
				yNode2 = yNode1
				zNode2 = zNode1
				x = [xNode1,xNode2]
				y = [yNode1,yNode2]
				z = [zNode1,zNode2]
				# line
				ax.plot(x, y, z, c='tab:red', linewidth=1)
				# arrow head
				ax.plot([xNode2,xNode2+arrowDimension], y, [zNode2+arrowDimension*1.2,zNode2], c='tab:red', linewidth=1)
				ax.plot([xNode2,xNode2+arrowDimension], y, [zNode2-arrowDimension*1.2,zNode2], c='tab:red', linewidth=1)
				ax.plot([xNode2,xNode2], y, [zNode2+arrowDimension*1.2,zNode2-arrowDimension*1.2], c='tab:red', linewidth=1)
			if self.DirectionsEarthquake[i,1] != 0:	# Earthquake applied in y direction
				xNode2 = xNode1
				yNode2 = yNode1 + ag[1]
				zNode2 = zNode1
				x = [xNode1,xNode2]
				y = [yNode1,yNode2]
				z = [zNode1,zNode2]
				# line
				ax.plot(x, y, z, c='tab:red', linewidth=1)
				# arrow head
				ax.plot(x, [yNode2,yNode2+arrowDimension], [zNode2+arrowDimension*1.2,zNode2], c='tab:red', linewidth=1)
				ax.plot(x, [yNode2,yNode2+arrowDimension], [zNode2-arrowDimension*1.2,zNode2], c='tab:red', linewidth=1)
				ax.plot(x, [yNode2,yNode2], [zNode2+arrowDimension*1.2,zNode2-arrowDimension*1.2], c='tab:red', linewidth=1)
			if self.DirectionsEarthquake[i,2] != 0:	# Earthquake applied in y direction
				xNode2 = xNode1
				yNode2 = yNode1
				zNode2 = zNode1 + ag[2]
				x = [xNode1,xNode2]
				y = [yNode1,yNode2]
				z = [zNode1,zNode2]
				# line
				ax.plot(x, y, z, c='tab:red', linewidth=1)
				# arrow head
				ax.plot([xNode2+arrowDimension*1.2,xNode2], y, [zNode2,zNode2+arrowDimension*1.2], c='tab:red', linewidth=1)
				ax.plot([xNode2-arrowDimension*1.2,xNode2], y, [zNode2,zNode2+arrowDimension*1.2], c='tab:red', linewidth=1)
				ax.plot([xNode2-arrowDimension*1.2,xNode2+arrowDimension*1.2], y, [zNode2,zNode2], c='tab:red', linewidth=1)

		# Setting the axes properties
		if min(self.NodeX) != max(self.NodeX):
			ax.set(xlim3d=(min(self.NodeX)*1.2, max(self.NodeX)*1.2))
		else:
			ax.set(xlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeY) != max(self.NodeY):
			ax.set(ylim3d=(min(self.NodeY)*1.2, max(self.NodeY)*1.2))
		else:
			ax.set(ylim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeZ) != max(self.NodeZ):
			ax.set(zlim3d=(min(self.NodeZ)*1.2, max(self.NodeZ)*1.2))
		else:
			ax.set(zlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))

		ax.title.set_text('Deformation scale factor: '+str(round(self.scaleFactorAnimation,1)))
		ax.text(max(self.NodeX)*1.1, min(self.NodeY)*1.1, min(self.NodeZ)*1.1, 't = '+str(round(t,2))+'s')
		ax.set_axis_off()
		#plt.show()
		results_dir = os.path.join(self.script_dir, self.caseName+"/deformation_animation/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = 'deformation_iteration_'+str(it)+'.png'#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()


	def plotDeformedStructureTMD(self,it,t,r,s):
		ux = r[:self.NumberOfNodes]
		uy = r[self.NumberOfNodes:self.NumberOfNodes*2]
		uz = r[self.NumberOfNodes*2:self.NumberOfNodes*3]

		NodeXdef = self.NodeX+ux*self.scaleFactorAnimation
		NodeYdef = self.NodeY+uy*self.scaleFactorAnimation
		NodeZdef = self.NodeZ+uz*self.scaleFactorAnimation

		NodeXdefTMD = NodeXdef[self.indexNodeTMD]
		NodeYdefTMD = NodeYdef[self.indexNodeTMD]
		NodeZdefTMD = NodeZdef[self.indexNodeTMD]

		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [NodeXdef[orderNode1],NodeXdef[orderNode2]]
			y = [NodeYdef[orderNode1],NodeYdef[orderNode2]]
			z = [NodeZdef[orderNode1],NodeZdef[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)

		for i in range(len(self.xCoordTMD)):	# Loop in number of TMDs
			# Plot connecting line of TMD
			if self.TMDDirection[i] == 1:	# X motion of TMD
				xNode2 = NodeXdefTMD[i] + s[i]*self.scaleFactorAnimation
				yNode2 = NodeYdefTMD[i]
				zNode2 = NodeZdefTMD[i]
			elif self.TMDDirection[i] == 2:	# Y motion of TMD
				xNode2 = NodeXdefTMD[i]
				yNode2 = NodeYdefTMD[i] + s[i]*self.scaleFactorAnimation
				zNode2 = NodeZdefTMD[i]
			elif self.TMDDirection[i] == 3:	# Y motion of TMD
				xNode2 = NodeXdefTMD[i]
				yNode2 = NodeYdefTMD[i]
				zNode2 = NodeZdefTMD[i] + s[i]*self.scaleFactorAnimation
			else: errorVisualisationTMD 	# Not ready for rotational TMD
			x = [NodeXdefTMD[i],xNode2]
			y = [NodeYdefTMD[i],yNode2]
			z = [NodeZdefTMD[i],zNode2]
			ax.plot(x, y, z, color='black', linewidth=1, linestyle = '--')
			# Plot hexaedra representing the TMD Mass
			# Hexaedra nodes
			l = self.TMDVisual[i][0]
			b = self.TMDVisual[i][1]
			h = self.TMDVisual[i][2]
			xNodeHex = [xNode2-l,xNode2+l,xNode2+l,xNode2-l,xNode2-l,xNode2+l,xNode2+l,xNode2-l]	# x coord of 8 nodes
			yNodeHex = [yNode2-b,yNode2-b,yNode2+b,yNode2+b,yNode2-b,yNode2-b,yNode2+b,yNode2+b]	# y coord of 8 nodes
			zNodeHex = [zNode2-h,zNode2-h,zNode2-h,zNode2-h,zNode2+h,zNode2+h,zNode2+h,zNode2+h]	# z coord of 8 nodes
			# Hexaedra faces
			vertices = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[3,2,6,7],[0,3,7,4]]
			tupleList = list(zip(xNodeHex, yNodeHex, zNodeHex))

			poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
			ax.add_collection3d(Poly3DCollection(poly3d, facecolors='r', linewidths=1, alpha=0.5))

		# Setting the axes properties
		if min(self.NodeX) != max(self.NodeX):
			ax.set(xlim3d=(min(self.NodeX)*1.2, max(self.NodeX)*1.2))
		else:
			ax.set(xlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeY) != max(self.NodeY):
			ax.set(ylim3d=(min(self.NodeY)*1.2, max(self.NodeY)*1.2))
		else:
			ax.set(ylim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeZ) != max(self.NodeZ):
			ax.set(zlim3d=(min(self.NodeZ)*1.2, max(self.NodeZ)*1.2))
		else:
			ax.set(zlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))

		ax.title.set_text('Deformation scale factor: '+str(round(self.scaleFactorAnimation,1)))
		ax.text(max(self.NodeX)*1.1, min(self.NodeY)*1.1, min(self.NodeZ)*1.1, 't = '+str(round(t,2))+'s')
		ax.set_axis_off()
		#plt.show()
		results_dir = os.path.join(self.script_dir, self.caseName+"/deformation_animation/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = 'deformation_iteration_'+str(it)+'.png'#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()


	def plotDeformedStructureVBI(self,it,t,r,q,xyCGVehiclev,zgv):
		ux = r[:self.NumberOfNodes]
		uy = r[self.NumberOfNodes:self.NumberOfNodes*2]
		uz = r[self.NumberOfNodes*2:self.NumberOfNodes*3]

		NodeXdef = self.NodeX+ux*self.scaleFactorAnimation
		NodeYdef = self.NodeY+uy*self.scaleFactorAnimation
		NodeZdef = self.NodeZ+uz*self.scaleFactorAnimation


		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [NodeXdef[orderNode1],NodeXdef[orderNode2]]
			y = [NodeYdef[orderNode1],NodeYdef[orderNode2]]
			z = [NodeZdef[orderNode1],NodeZdef[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)
		for dl in range(len(self.VehicleType)):	# Loop in number of vehicles
			if self.VehicleType[dl] == 'quarterCar':
				dContactVehicleBox = self.VehicleVisual[dl][2]*1.5
				XCGVehicle0 = xyCGVehiclev[dl][0]
				YCGVehicle0 = xyCGVehiclev[dl][1]
				xNode2 = XCGVehicle0
				yNode2 = YCGVehicle0
				zNode2 = zgv[dl]+dContactVehicleBox+q[dl]*self.scaleFactorAnimation
				x = [XCGVehicle0,xNode2]
				y = [YCGVehicle0,yNode2]
				z = [zgv[dl]*self.scaleFactorAnimation,zNode2]

				ax.plot(x, y, z, color='black', linewidth=1, linestyle = '--')

				# Plot hexaedra representing the TMD Mass
				# Hexaedra nodes
				l = self.VehicleVisual[dl][0]
				b = self.VehicleVisual[dl][1]
				h = self.VehicleVisual[dl][2]
				xNodeHex = [xNode2-l,xNode2+l,xNode2+l,xNode2-l,xNode2-l,xNode2+l,xNode2+l,xNode2-l]	# x coord of 8 nodes
				yNodeHex = [yNode2-b,yNode2-b,yNode2+b,yNode2+b,yNode2-b,yNode2-b,yNode2+b,yNode2+b]	# y coord of 8 nodes
				zNodeHex = [zNode2-h,zNode2-h,zNode2-h,zNode2-h,zNode2+h,zNode2+h,zNode2+h,zNode2+h]	# z coord of 8 nodes
				# Hexaedra faces
				vertices = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[3,2,6,7],[0,3,7,4]]
				tupleList = list(zip(xNodeHex, yNodeHex, zNodeHex))

				poly3d = [[tupleList[vertices[ix][iy]] for iy in range(len(vertices[0]))] for ix in range(len(vertices))]
				ax.add_collection3d(Poly3DCollection(poly3d, facecolors='r', linewidths=1, alpha=0.5))

		# Setting the axes properties
		if min(self.NodeX) != max(self.NodeX):
			ax.set(xlim3d=(min(self.NodeX)*1.2, max(self.NodeX)*1.2))
		else:
			ax.set(xlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeY) != max(self.NodeY):
			ax.set(ylim3d=(min(self.NodeY)*1.2, max(self.NodeY)*1.2))
		else:
			ax.set(ylim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeZ) != max(self.NodeZ):
			ax.set(zlim3d=(min(self.NodeZ)*1.2, max(self.NodeZ)*1.2))
		else:
			ax.set(zlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))

		ax.title.set_text('Deformation scale factor: '+str(round(self.scaleFactorAnimation,1)))
		ax.text(max(self.NodeX)*1.1, min(self.NodeY)*1.1, min(self.NodeZ)*1.1, 't = '+str(round(t,2))+'s')
		ax.set_axis_off()
		#plt.show()
		results_dir = os.path.join(self.script_dir, self.caseName+"/deformation_animation/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = 'deformation_iteration_'+str(it)+'.png'#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()


	def plotDeformedStructureSection(self,it,t,r,showPlotSection,plotSectionNodes):
		ux = r[:self.NumberOfNodes]
		uy = r[self.NumberOfNodes:self.NumberOfNodes*2]
		uz = r[self.NumberOfNodes*2:self.NumberOfNodes*3]
		urx = r[self.NumberOfNodes*3:self.NumberOfNodes*4]
		urx = urx*self.scaleFactorAnimation

		NodeXdef = self.NodeX+ux*self.scaleFactorAnimation
		NodeYdef = self.NodeY+uy*self.scaleFactorAnimation
		NodeZdef = self.NodeZ+uz*self.scaleFactorAnimation

		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [NodeXdef[orderNode1],NodeXdef[orderNode2]]
			y = [NodeYdef[orderNode1],NodeYdef[orderNode2]]
			z = [NodeZdef[orderNode1],NodeZdef[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)
		for i in range(len(self.NodeX)):
			if self.NodeNumber[i] in plotSectionNodes:
				x = [NodeXdef[i],NodeXdef[i]]
				for j in range(len(showPlotSection)-1):
					ypoint1BeforeRotation = showPlotSection[j][0]+NodeYdef[i]
					ypoint2BeforeRotation = showPlotSection[j+1][0]+NodeYdef[i]
					zpoint1BeforeRotation = showPlotSection[j][1]+NodeZdef[i]
					zpoint2BeforeRotation = showPlotSection[j+1][1]+NodeZdef[i]
					ypoint1 = ypoint1BeforeRotation*np.cos(urx[i]) - zpoint1BeforeRotation*np.sin(urx[i])
					zpoint1 = zpoint1BeforeRotation*np.cos(urx[i]) + ypoint1BeforeRotation*np.sin(urx[i])
					ypoint2 = ypoint2BeforeRotation*np.cos(urx[i]) - zpoint2BeforeRotation*np.sin(urx[i])
					zpoint2 = zpoint2BeforeRotation*np.cos(urx[i]) + ypoint2BeforeRotation*np.sin(urx[i])
					y = [ypoint1,ypoint2]
					z = [zpoint1,zpoint2]
					ax.plot(x, y, z, c='tab:blue', linewidth=0.75)

				ypoint1BeforeRotation = showPlotSection[-1][0]+NodeYdef[i]
				ypoint2BeforeRotation = showPlotSection[0][0]+NodeYdef[i]
				zpoint1BeforeRotation = showPlotSection[-1][1]+NodeZdef[i]
				zpoint2BeforeRotation = showPlotSection[0][1]+NodeZdef[i]
				ypoint1 = ypoint1BeforeRotation*np.cos(urx[i]) - zpoint1BeforeRotation*np.sin(urx[i])
				zpoint1 = zpoint1BeforeRotation*np.cos(urx[i]) + ypoint1BeforeRotation*np.sin(urx[i])
				ypoint2 = ypoint2BeforeRotation*np.cos(urx[i]) - zpoint2BeforeRotation*np.sin(urx[i])
				zpoint2 = zpoint2BeforeRotation*np.cos(urx[i]) + ypoint2BeforeRotation*np.sin(urx[i])
				y = [ypoint1,ypoint2]
				z = [zpoint1,zpoint2]

				ax.plot(x, y, z, c='tab:blue', linewidth=0.75)

		# Setting the axes properties
		if min(self.NodeX) != max(self.NodeX):
			ax.set(xlim3d=(min(self.NodeX)*1.2, max(self.NodeX)*1.2))
		else:
			ax.set(xlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeY) != max(self.NodeY):
			ax.set(ylim3d=(min(self.NodeY)*1.2, max(self.NodeY)*1.2))
		else:
			ax.set(ylim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeZ) != max(self.NodeZ):
			ax.set(zlim3d=(min(self.NodeZ)*1.2, max(self.NodeZ)*1.2))
		else:
			ax.set(zlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))

		ax.title.set_text('Deformation scale factor: '+str(round(self.scaleFactorAnimation,1)))
		ax.text(max(self.NodeX)*1.1, min(self.NodeY)*1.1, min(self.NodeZ)*1.1, 't = '+str(round(t,2))+'s')
		ax.set_axis_off()
		#plt.show()
		results_dir = os.path.join(self.script_dir, self.caseName+"/deformation_animation/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = 'deformation_iteration_'+str(it)+'.png'#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()

	def plotDeformedStructureSectionLoads(self,it,t,r,LoadAmplitude,maxLoad,xyCGVehiclev,showPlotSection,plotSectionNodes):
		ux = r[:self.NumberOfNodes]
		uy = r[self.NumberOfNodes:self.NumberOfNodes*2]
		uz = r[self.NumberOfNodes*2:self.NumberOfNodes*3]
		urx = r[self.NumberOfNodes*3:self.NumberOfNodes*4]
		urx = urx*self.scaleFactorAnimation

		# force vector
		LoadAmplitudeNorm = abs(np.array(LoadAmplitude))*(self.dimensionPlot*0.3)/maxLoad
		arrowDimension = self.dimensionPlot/50

		NodeXdef = self.NodeX+ux*self.scaleFactorAnimation
		NodeYdef = self.NodeY+uy*self.scaleFactorAnimation
		NodeZdef = self.NodeZ+uz*self.scaleFactorAnimation

		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [NodeXdef[orderNode1],NodeXdef[orderNode2]]
			y = [NodeYdef[orderNode1],NodeYdef[orderNode2]]
			z = [NodeZdef[orderNode1],NodeZdef[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)
		for i in range(len(self.NodeX)):
			if self.NodeNumber[i] in plotSectionNodes:
				x = [NodeXdef[i],NodeXdef[i]]
				for j in range(len(showPlotSection)-1):
					ypoint1BeforeRotation = showPlotSection[j][0]+NodeYdef[i]
					ypoint2BeforeRotation = showPlotSection[j+1][0]+NodeYdef[i]
					zpoint1BeforeRotation = showPlotSection[j][1]+NodeZdef[i]
					zpoint2BeforeRotation = showPlotSection[j+1][1]+NodeZdef[i]
					ypoint1 = ypoint1BeforeRotation*np.cos(urx[i]) - zpoint1BeforeRotation*np.sin(urx[i])
					zpoint1 = zpoint1BeforeRotation*np.cos(urx[i]) + ypoint1BeforeRotation*np.sin(urx[i])
					ypoint2 = ypoint2BeforeRotation*np.cos(urx[i]) - zpoint2BeforeRotation*np.sin(urx[i])
					zpoint2 = zpoint2BeforeRotation*np.cos(urx[i]) + ypoint2BeforeRotation*np.sin(urx[i])
					y = [ypoint1,ypoint2]
					z = [zpoint1,zpoint2]
					ax.plot(x, y, z, c='tab:blue', linewidth=0.75)

				ypoint1BeforeRotation = showPlotSection[-1][0]+NodeYdef[i]
				ypoint2BeforeRotation = showPlotSection[0][0]+NodeYdef[i]
				zpoint1BeforeRotation = showPlotSection[-1][1]+NodeZdef[i]
				zpoint2BeforeRotation = showPlotSection[0][1]+NodeZdef[i]
				ypoint1 = ypoint1BeforeRotation*np.cos(urx[i]) - zpoint1BeforeRotation*np.sin(urx[i])
				zpoint1 = zpoint1BeforeRotation*np.cos(urx[i]) + ypoint1BeforeRotation*np.sin(urx[i])
				ypoint2 = ypoint2BeforeRotation*np.cos(urx[i]) - zpoint2BeforeRotation*np.sin(urx[i])
				zpoint2 = zpoint2BeforeRotation*np.cos(urx[i]) + ypoint2BeforeRotation*np.sin(urx[i])
				y = [ypoint1,ypoint2]
				z = [zpoint1,zpoint2]

				ax.plot(x, y, z, c='tab:blue', linewidth=0.75)

		for i in range(len(self.YVehicle)): # loop in vehicles
			for j in range(len(self.YVehicle[i])): # loop in wheels
				XWheel = xyCGVehiclev[i][0] + self.XVehicle[i][j]
				YWheel = xyCGVehiclev[i][1] + self.YVehicle[i][j]
				x = [XWheel,XWheel]
				y = [YWheel,YWheel]
				z = [0,LoadAmplitudeNorm[i][j]]
				# line
				ax.plot(x, y, z, c='tab:red', linewidth=1)
				# arrow head
				ax.plot([XWheel,XWheel+arrowDimension], y, [0,arrowDimension*1.2], c='tab:red', linewidth=1)
				ax.plot([XWheel,XWheel-arrowDimension], y, [0,arrowDimension*1.2], c='tab:red', linewidth=1)
				ax.plot([XWheel-arrowDimension,XWheel+arrowDimension], y, [arrowDimension*1.2,arrowDimension*1.2], c='tab:red', linewidth=1)


		# Setting the axes properties
		if min(self.NodeX) != max(self.NodeX):
			ax.set(xlim3d=(min(self.NodeX)*1.2, max(self.NodeX)*1.2))
		else:
			ax.set(xlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeY) != max(self.NodeY):
			ax.set(ylim3d=(min(self.NodeY)*1.2, max(self.NodeY)*1.2))
		else:
			ax.set(ylim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeZ) != max(self.NodeZ):
			ax.set(zlim3d=(min(self.NodeZ)*1.2, max(self.NodeZ)*1.2))
		else:
			ax.set(zlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))

		#ax.title.set_text('Deformation scale factor: '+str(round(self.scaleFactorAnimation,1)))
		font1 = {'family':'fantasy','color':'red','size':22}
		font2 = {'family':'fantasy','color':'black','size':15}

		#ax.set_title('La UD de Puentes de la UPM te desea Feliz Navidad!', fontdict = font1)
		#ax.set_title('Merry Christmas and vibrant 2025!', fontdict = font1)
		#ax.text(12, 0, 30, 'Alfredo & MDyn', fontdict = font2)
		ax.title.set_text('Deformation scale factor: '+str(round(self.scaleFactorAnimation,1)))
		ax.text(max(self.NodeX)*1.1, min(self.NodeY)*1.1, min(self.NodeZ)*1.1, 't = '+str(round(t,2))+'s')
		ax.set_axis_off()
		#plt.show()
		results_dir = os.path.join(self.script_dir, self.caseName+"/deformation_animation/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = 'deformation_iteration_'+str(it)+'.png'#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()



	def plotDeformedStructureLoads(self,it,t,r,LoadAmplitude,maxLoad,xyzLoad,DofLoad):
		"""
		xyzLoad is a array with 3x1 arrays that contain the xyz coord of the loads
		DofLoad is an array with the DOF of each load
		"""
		ux = r[:self.NumberOfNodes]
		uy = r[self.NumberOfNodes:self.NumberOfNodes*2]
		uz = r[self.NumberOfNodes*2:self.NumberOfNodes*3]
		urx = r[self.NumberOfNodes*3:self.NumberOfNodes*4]
		urx = urx*self.scaleFactorAnimation

		# force vector
		LoadAmplitudeNorm = LoadAmplitude*(self.dimensionPlot*0.2)/maxLoad
		arrowDimension = self.dimensionPlot/50

		NodeXdef = self.NodeX+ux*self.scaleFactorAnimation
		NodeYdef = self.NodeY+uy*self.scaleFactorAnimation
		NodeZdef = self.NodeZ+uz*self.scaleFactorAnimation

		#ax = self.prepare_fig()
		fig = plt.figure()
		ax = fig.add_subplot(projection='3d')
		ax.set_box_aspect((self.xdimEstimate, self.ydimEstimate, self.zdimEstimate))  # aspect ratio is 1:1:1 in data space
		#ax.set_zlim(-1, 1)
		NodeNumberCList = list(self.NodeNumber)
		for i in range(len(self.BeamNode1)):
			Node1 = self.BeamNode1[i]
			Node2 = self.BeamNode2[i]
			orderNode1 = NodeNumberCList.index(Node1)
			orderNode2 = NodeNumberCList.index(Node2)
			x = [NodeXdef[orderNode1],NodeXdef[orderNode2]]
			y = [NodeYdef[orderNode1],NodeYdef[orderNode2]]
			z = [NodeZdef[orderNode1],NodeZdef[orderNode2]]
			ax.plot(x, y, z, c='tab:blue', linewidth=1)

		for i in range(len(LoadAmplitudeNorm)): # loop in loads
			xload = xyzLoad[i][0]
			yload = xyzLoad[i][1]
			zload = xyzLoad[i][2]

			if DofLoad[i] == 1:		# X load
				x = [xload,xload-LoadAmplitudeNorm[i]]
				y = [yload,yload]
				z = [zload,zload]
				xa1 = [xload,xload-arrowDimension]
				ya1 = [yload,yload]
				za1 = [zload,zload-arrowDimension]
				xa2 = [xload,xload-arrowDimension]
				ya2 = [yload,yload]
				za2 = [zload,zload+arrowDimension]
				xa3 = [xload-arrowDimension,xload-arrowDimension]
				ya3 = [yload,yload]
				za3 = [zload-arrowDimension,zload+arrowDimension]
			elif DofLoad[i] == 2:	# Y load
				x = [xload,xload]
				y = [yload,yload-LoadAmplitudeNorm[i]]
				z = [zload,zload]
				xa1 = [xload,xload-arrowDimension]
				ya1 = [yload,yload-arrowDimension]
				za1 = [zload,zload]
				xa2 = [xload,xload+arrowDimension]
				ya2 = [yload,yload-arrowDimension]
				za2 = [zload,zload]
				xa3 = [xload-arrowDimension,xload+arrowDimension]
				ya3 = [yload-arrowDimension,yload-arrowDimension]
				za3 = [zload,zload]
			elif DofLoad[i] == 3:		# Z load
				x = [xload,xload]
				y = [yload,yload]
				z = [zload,zload-LoadAmplitudeNorm[i]]
				xa1 = [xload,xload-arrowDimension]
				ya1 = [yload,yload]
				za1 = [zload,zload+arrowDimension]
				xa2 = [xload,xload+arrowDimension]
				ya2 = [yload,yload]
				za2 = [zload,zload+arrowDimension]
				xa3 = [xload-arrowDimension,xload+arrowDimension]
				ya3 = [yload,yload]
				za3 = [zload+arrowDimension,zload+arrowDimension]

			# line
			ax.plot(x, y, z, c='tab:red', linewidth=1)
			# arrow head
			ax.plot(xa1,ya1,za1, c='tab:red', linewidth=1)
			ax.plot(xa2,ya2,za2, c='tab:red', linewidth=1)
			ax.plot(xa3,ya3,za3, c='tab:red', linewidth=1)


		# Setting the axes properties
		if min(self.NodeX) != max(self.NodeX):
			ax.set(xlim3d=(min(self.NodeX)*1.2, max(self.NodeX)*1.2))
		else:
			ax.set(xlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeY) != max(self.NodeY):
			ax.set(ylim3d=(min(self.NodeY)*1.2, max(self.NodeY)*1.2))
		else:
			ax.set(ylim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))
		if min(self.NodeZ) != max(self.NodeZ):
			ax.set(zlim3d=(min(self.NodeZ)*1.2, max(self.NodeZ)*1.2))
		else:
			ax.set(zlim3d=(-1*self.dimensionPlot/10, self.dimensionPlot/10))

		ax.title.set_text('Deformation scale factor: '+str(round(self.scaleFactorAnimation,1)))
		ax.text(max(self.NodeX)*1.1, min(self.NodeY)*1.1, min(self.NodeZ)*1.1, 't = '+str(round(t,2))+'s')
		ax.set_axis_off()
		#plt.show()
		results_dir = os.path.join(self.script_dir, self.caseName+"/deformation_animation/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		filename = 'deformation_iteration_'+str(it)+'.png'#, bbox_inches = None)
		plt.savefig(results_dir + filename)
		plt.close()





	def createAnimation(self,filenamesAnimation,durationAnimation,removeAnimationFigures):
		animation_dir = os.path.join(self.script_dir, self.caseName+"/deformation_animation/")
		images = [ ]
		for filename in filenamesAnimation:
			images.append(iio.imread(animation_dir+filename))
		iio.imwrite(animation_dir+self.caseName+'.gif', images, duration = durationAnimation, loop = 0)
		if removeAnimationFigures == 1:
			for filename in filenamesAnimation:
				os.remove(animation_dir+filename)



class WriteToFile():
	def __init__(self, **kwargs):
		self.NumberOfNodes = kwargs.get("NumberOfNodes")
		self.NodeNumber = kwargs.get("NodeNumber")
		self.script_dir = os.getcwd() # os.path.join( os.path.dirname( __file__ ), '..' )
		self.caseName = kwargs.get("caseName")
		self.NodeNumberToWrite = kwargs.get("NodeNumberToWrite")
		self.writeOutputRate = kwargs.get("writeOutputRate")
		self.DOFactive = kwargs.get("DOFactive")
		self.indexElements = [index for index, element in enumerate(self.NodeNumber) if element in self.NodeNumberToWrite]
		self.indexElementsExpandedDOF = [] # np.zeros(len(self.DOFactive)*len(self.NodeNumberToWrite))
		self.TMDNodes = kwargs.get("TMDNodes")
		self.TMDDirection = kwargs.get("TMDDirection")
		self.NodeX = kwargs.get("NodeX")
		self.NodeY = kwargs.get("NodeY")
		self.NodeZ = kwargs.get("NodeZ")
		self.VehicleType = kwargs.get("VehicleType")
		self.VehicleOrderToWrite = kwargs.get("VehicleOrderToWrite")

		# Check convergency in time loop iterations
		self.NodeCheckIter = kwargs.get("NodeCheckIter")
		if self.NodeCheckIter is not None:
			self.indexNodeCheckIter = [index for index, element in enumerate(self.NodeNumber) if element in self.NodeCheckIter]
			self.indexUYCheck=self.DOFactive.index(2)
			self.indexUZCheck=self.DOFactive.index(3)
			self.indexUYCheckNodes = list((self.NumberOfNodes*self.indexUYCheck)+np.array(self.indexNodeCheckIter))
			self.indexUZCheckNodes = list((self.NumberOfNodes*self.indexUZCheck)+np.array(self.indexNodeCheckIter))

		for i in range(len(self.indexElements)):
			tempvec = np.zeros(len(self.DOFactive)).astype(int)
			for j in range(len(self.DOFactive)):
				tempvec[j] = self.indexElements[i]+(self.NumberOfNodes*j)
			self.indexElementsExpandedDOF.append(list(tempvec))


	def dispmaxiter(self,r): # Obtain the maximum displacement in direction "orderdirection" of nodeset ""
		riterUY = r[self.indexUYCheckNodes]
		riterUZ = r[self.indexUZCheckNodes]
		return max(abs(riterUY)),max(abs(riterUZ))

	def initVectorOutput(self,t): # Create a zeros vector to be filled with the output to write
		counter = 1
		toutput = [t[0]]
		for i in range(1,len(t)): # Main loop - time
			output_flag = self.writeOutputRate > 0 and (i % self.writeOutputRate == 0 or i == 1 or i == len(t))
			if output_flag:
				toutput.append(t[i])
				counter += 1
		self.timeOutput = toutput
		return np.zeros((len(self.DOFactive),counter,len(self.NodeNumberToWrite)))# Structural movements: Rows DOF; Cols. time, layer node order


	def initVectorOutputVehicle(self,t): # Create a zeros vector to be filled with the output to write for the auxiliary vibrating system of order dl
		counter = 1
		toutput = [t[0]]
		for i in range(1,len(t)): # Main loop - time
			output_flag = self.writeOutputRate > 0 and (i % self.writeOutputRate == 0 or i == 1 or i == len(t))
			if output_flag:
				toutput.append(t[i])
				counter += 1
		self.timeOutputVibSys = toutput
		q_Output = []
		for dl in range(len(self.VehicleType)):
			if dl in self.VehicleOrderToWrite:
				if self.VehicleType[dl] == 'quarterCar': #'SDOFVehicle':
					NumDOFsVehicle = 1
				q_Output.append(np.zeros((NumDOFsVehicle,counter)))# Vibrating system movements: [vehicle order referred to order in self.VehicleOrderToWrite] [Rows DOF of vib system; Cols. time]
		return q_Output

	def initVectorOutputReactionsVehicle(self,t): # Create a zeros vector to be filled with the output to write for the auxiliary vibrating system of order dl
		counter = 1
		toutput = [t[0]]
		for i in range(1,len(t)): # Main loop - time
			output_flag = self.writeOutputRate > 0 and (i % self.writeOutputRate == 0 or i == 1 or i == len(t))
			if output_flag:
				toutput.append(t[i])
				counter += 1
		self.timeOutputVibSys = toutput
		F_Output = []
		for dl in range(len(self.VehicleType)):
			if dl in self.VehicleOrderToWrite:
				if self.VehicleType[dl] == 'quarterCar': #'SDOFVehicle':
					NumWheelsVehicle = 1 # There is one wheel in this vehicle
					NumDirectionsRections = 1 # Only vertical reaction forces
				F_Output.append(np.zeros((NumDirectionsRections,counter,NumWheelsVehicle))) # Vibrating system reaction forces: [vehicle order in vector self.VehicleOrderToWrite] [Rows Direction of reaction force; Cols. time; layer wheel]
		return F_Output

	def initVectorOutputStatic(self): # Create a zeros vector to be filled with the static output to write
		return np.zeros((len(self.DOFactive),len(self.NodeNumberToWrite)))# Structural movements: Rows DOF; Cols. node order


	def initVectorOutputTMD(self,t): # Create a zeros vector to be filled with the output to write
		counter = 1
		toutput = [t[0]]
		for i in range(1,len(t)): # Main loop - time
			output_flag = self.writeOutputRate > 0 and (i % self.writeOutputRate == 0 or i == 1 or i == len(t))
			if output_flag:
				toutput.append(t[i])
				counter += 1
		self.timeOutput = toutput
		return np.zeros((len(self.TMDNodes),counter))# TMD movements: Rows TMD; Cols. time

	def initVectorOutputConvCheck(self,t): # Create a zeros vector to be filled with the output to write
		counter = 1
		toutput = [t[0]]
		for i in range(1,len(t)): # Main loop - time
			output_flag = self.writeOutputRate > 0 and (i % self.writeOutputRate == 0 or i == 1 or i == len(t))
			if output_flag:
				toutput.append(t[i])
				counter += 1
		self.timeOutput = toutput
		return np.zeros((3,counter))# TMD movements: Rows number iterations, DrCheck, quality flag (0 if not reached, 1 if reached); Cols. time


	def writeOutput(self,rout,r,i):
		for j in range(len(self.indexElements)):
			rout[:,i,j] = r[self.indexElementsExpandedDOF[j]]

	def writeOutputStatic(self,rout,r):
		for j in range(len(self.indexElements)):
			rout[:,j] = r[self.indexElementsExpandedDOF[j]]

	def writeOutputTMD(self,sout,s,i):
		sout[:,i] = s

	def writeOutputConvCheck(self,convcheck_Output,countIterCheck,DrCheck,ToleranceIter,i):
		convcheck_Output[0,i] = countIterCheck
		convcheck_Output[1,i] = DrCheck
		if DrCheck > ToleranceIter:
			convcheck_Output[2,i] = 0
		else:
			convcheck_Output[2,i] = 1


	def writeOutputVehicleDynamics(self,qout,q,i):
		# Vibrating system movements: [vehicle order referred to order in self.VehicleOrderToWrite] [Rows DOF of vib system; Cols. time]
		for dl in range(len(self.VehicleType)):
			if dl in self.VehicleOrderToWrite:
				qout[dl][:,i] = q[self.VehicleOrderToWrite[dl]]

	def writeOutputVehicleForces(self,Fout,F,i):
		# Fout - Vibrating system forces at contact points: [vehicle order in vector self.VehicleOrderToWrite] [Rows Direction of reaction force; Cols. time; layer wheel]
		# F - All Vibrating system forces at contact points: # [order vehicle] [order of Direction,order of wheel]
		for dl in range(len(self.VehicleType)):
			if dl in self.VehicleOrderToWrite:
				if self.VehicleType[dl] == 'quarterCar': #'SDOFVehicle':
					NumWheelsVehicle = 1
					for wh in range(NumWheelsVehicle):	# Loop in wheel
						Fout[dl][0,i] = F[self.VehicleOrderToWrite[dl]][0,wh]



	def writeOutputFile(self,rout,flagoutput,flagplots):
		results_dir = os.path.join(self.script_dir, self.caseName+"/time_histories/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)

		for i in range(len(self.NodeNumberToWrite)):
			NodetoSave = self.NodeNumberToWrite[i]

			filename = flagoutput+'_node_'+str(NodetoSave)+'_MDyn'
			fichero = open(os.path.join(results_dir, filename)+'.txt' , 'w')

			if flagoutput == 'displacement':
				flagoutputplot = 'Displacements [mm]'
				fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Movement UX [mm]
# 	Col 3 --> Movement UY [mm]
# 	Col 4 --> Movement UZ [mm]
# 	Col 5 --> Movement URX [mrad]
# 	Col 6 --> Movement URY [mrad]
# 	Col 7 --> Movement URZ [mrad]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")
			elif flagoutput == 'velocity':
				flagoutputplot = 'Velocities [mm/s]'
				fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Velocity VX [mm/s]
# 	Col 3 --> Velocity VY [mm/s]
# 	Col 4 --> Velocity VZ [mm/s]
# 	Col 5 --> Velocity VRX [mrad/s]
# 	Col 6 --> Velocity VRY [mrad/s]
# 	Col 7 --> Velocity VRZ [mrad/s]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")

			elif flagoutput == 'acceleration':
				flagoutputplot = r'Accelerations [mm/s$^2$]'
				fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Acceleration AX [mm/s2]
# 	Col 3 --> Acceleration AY [mm/s2]
# 	Col 4 --> Acceleration AZ [mm/s2]
# 	Col 5 --> Acceleration ARX [mrad/s2]
# 	Col 6 --> Acceleration ARY [mrad/s2]
# 	Col 7 --> Acceleration ARZ [mrad/s2]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")


			for j in range(len(self.timeOutput)):
				fichero.write('%.4f %2.4f %2.4f %2.4f %2.6f %2.6f %2.6f \n' % (self.timeOutput[j],rout[0,j,i]*1e3,rout[1,j,i]*1e3,rout[2,j,i]*1e3,rout[3,j,i]*1e3,rout[4,j,i]*1e3,rout[5,j,i]*1e3))

			fichero.close()

			if flagplots == 'plotsYes':

				# Plot figure rZ
				# ............
				fig=plt.figure()
				ax=fig.add_subplot(111)
				# ............
				ax.plot(self.timeOutput,rout[0,:,i]*1e3,'-.',linewidth=1,markersize=6,markevery=1,color='green',label = r'Longitudinal displacement: $r_X$')
				ax.plot(self.timeOutput,rout[1,:,i]*1e3,'--',linewidth=1,markersize=6,markevery=1,color='blue',label = r'Transverse displacement: $r_Y$')
				ax.plot(self.timeOutput,rout[2,:,i]*1e3,'-',linewidth=1,markersize=6,markevery=1,color='red',label = r'Vertical displacement: $r_Z$')

				# --------- Custom plot -----------
				ax.set_xlabel(r'Time; $t$ [s]')
				ax.set_xlim(0,)
				ax.set_ylabel(flagoutputplot)
				# ------ legend ------
				# add the legend in the upper right corner of the plot
				leg = ax.legend(fancybox=True, loc='best', shadow=False)
				# set the alpha value of the legend: it will be translucent
				leg.get_frame().set_alpha(0.6)
				filename = flagoutput+'_node_'+str(NodetoSave)+'_MDyn_plot_time.pdf'

				plt.savefig(os.path.join(results_dir, filename))
				plt.close()

	def writeOutputFileResponseVehicles(self,qout,flagoutput,flagplots):
		results_dir = os.path.join(self.script_dir, self.caseName+"/time_histories/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		# Vibrating system movements: [vehicle order referred to order in self.VehicleOrderToWrite] [Rows DOF of vib system; Cols. time]
		for i in range(len(self.VehicleOrderToWrite)):
			VehicletoSave = self.VehicleOrderToWrite[i]

			filename = flagoutput+'_vehicle_'+str(VehicletoSave)+'_MDyn'
			fichero = open(os.path.join(results_dir, filename)+'.txt' , 'w')

			NumberDOFVehicle = np.shape(qout[i])[0]

			if flagoutput == 'displacement':
				flagoutputplot = 'Displacements [mm]'
				if NumberDOFVehicle == 1:
					fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Vertical movement of unsprung mass: qZ [mm]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")
			elif flagoutput == 'velocity':
				flagoutputplot = 'Velocities [mm/s]'
				if NumberDOFVehicle == 1:
					fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Vertical velocity of unsprung mass: qdotZ [mm/s]
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")

			elif flagoutput == 'acceleration':
				flagoutputplot = r'Accelerations [mm/s$^2$]'
				if NumberDOFVehicle == 1:
					fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Vertical acceleration of unsprung mass: q2dotZ [mm2/s]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")

			if NumberDOFVehicle == 1:

				for j in range(len(self.timeOutputVibSys)):
					fichero.write('%.4f %2.4f \n' % (self.timeOutputVibSys[j],qout[i][0,j]*1e3))

			fichero.close()

			if flagplots == 'plotsYes':

				# Plot figure rZ
				# ............
				fig=plt.figure()
				ax=fig.add_subplot(111)
				# ............
				if NumberDOFVehicle == 1:
					ax.plot(self.timeOutputVibSys,qout[i][0,:]*1e3,'-',linewidth=1,markersize=6,markevery=1,color='red',label = r'Vertical response of unsprung mass')

				ax.axhline(y=0.0,color='black',linestyle='--',linewidth=0.5)
				# --------- Custom plot -----------
				ax.set_xlabel(r'Time; $t$ [s]')
				ax.set_xlim(0,)
				ax.set_ylabel(flagoutputplot)
				# ------ legend ------
				# add the legend in the upper right corner of the plot
				leg = ax.legend(fancybox=True, loc='best', shadow=False)
				# set the alpha value of the legend: it will be translucent
				leg.get_frame().set_alpha(0.6)
				filename = flagoutput+'_vehicle_'+str(VehicletoSave)+'_MDyn_plot_time.pdf'

				plt.savefig(os.path.join(results_dir, filename))
				plt.close()


	def writeOutputFileReactionVehicles(self,Fout,flagplots):
		results_dir = os.path.join(self.script_dir, self.caseName+"/time_histories/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)
		# Vibrating system reactions:  Vibrating system reaction forces: [vehicle order in vector self.VehicleOrderToWrite] [Rows Direction of reaction force; Cols. time; layer wheel]
		for i in range(len(self.VehicleOrderToWrite)):
			VehicletoSave = self.VehicleOrderToWrite[i]

			filename = 'reactions_vehicle_'+str(VehicletoSave)+'_MDyn'
			fichero = open(os.path.join(results_dir, filename)+'.txt' , 'w')

			NumberDirectionsReactionForce = np.shape(Fout[i])[0]
			NumberWheels = np.shape(Fout[i])[2]

			flagoutputplot = 'Reactions [kN]'
			if NumberDirectionsReactionForce == 1 and NumberDirectionsReactionForce == 1:
				fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Vertical reaction at contact point: [kN]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")
				for j in range(len(self.timeOutputVibSys)):
					fichero.write('%.4f %2.4f \n' % (self.timeOutputVibSys[j],Fout[i][0,j,0]*1e-3))

			fichero.close()

			if flagplots == 'plotsYes':

				# Plot figure rZ
				# ............
				fig=plt.figure()
				ax=fig.add_subplot(111)
				# ............
				if NumberDirectionsReactionForce == 1 and NumberDirectionsReactionForce == 1:
					ax.plot(self.timeOutputVibSys,Fout[i][0,:,0]*1e-3,'-',linewidth=1,markersize=6,markevery=1,color='red',label = r'Vertical reaction at contact point')
				ax.axhline(y=0.0,color='black',linestyle='--',linewidth=0.5)
				# --------- Custom plot -----------
				ax.set_xlabel(r'Time; $t$ [s]')
				ax.set_xlim(0,)
				ax.set_ylabel(flagoutputplot)
				# ------ legend ------
				# add the legend in the upper right corner of the plot
				leg = ax.legend(fancybox=True, loc='best', shadow=False)
				# set the alpha value of the legend: it will be translucent
				leg.get_frame().set_alpha(0.6)
				filename = 'reactions_vehicle_'+str(VehicletoSave)+'_MDyn_plot_time.pdf'

				plt.savefig(os.path.join(results_dir, filename))
				plt.close()



	def writeOutputFileStatic(self,rout):
		results_dir = os.path.join(self.script_dir, self.caseName+"/static_response/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)

		filename = 'movements_allrequestednodes_MDyn'
		fichero = open(os.path.join(results_dir, filename)+'.txt' , 'w')
		fichero.write("""# File with:
# 	Col 1 --> Node number
# 	Col 2 --> X coordinate
# 	Col 3 --> Y coordinate
# 	Col 4 --> Z coordinate
# 	Col 5 --> Movement UX [mm]
# 	Col 6 --> Movement UY [mm]
# 	Col 7 --> Movement UZ [mm]
# 	Col 8 --> Movement URX [mrad]
# 	Col 9 --> Movement URY [mrad]
# 	Col 10 --> Movement URZ [mrad]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")
		for i in range(len(self.NodeNumberToWrite)):
			NodetoSave = self.NodeNumberToWrite[i]
			XcoordNode = self.NodeX[self.indexElements[i]]
			YcoordNode = self.NodeY[self.indexElements[i]]
			ZcoordNode = self.NodeZ[self.indexElements[i]]

			fichero.write('%i %2.4f %2.4f %2.4f %2.4f %2.4f %2.4f %2.6f %2.6f %2.6f \n' % (NodetoSave,XcoordNode,YcoordNode,ZcoordNode,rout[0,i]*1e3,rout[1,i]*1e3,rout[2,i]*1e3,rout[3,i]*1e3,rout[4,i]*1e3,rout[5,i]*1e3))

		fichero.close()

		# Plot figure rZ
		# ............
		fig=plt.figure()
		ax=fig.add_subplot(111)
		# ............
		ax.plot(self.NodeX[self.indexElements],rout[0,:]*1e3,'-.',linewidth=1,markersize=6,markevery=1,color='green',label = r'Longitudinal displacement: $r_X$')
		ax.plot(self.NodeX[self.indexElements],rout[1,:]*1e3,'--',linewidth=1,markersize=6,markevery=1,color='blue',label = r'Transverse displacement: $r_Y$')
		ax.plot(self.NodeX[self.indexElements],rout[2,:]*1e3,'-',linewidth=1,markersize=6,markevery=1,color='red',label = r'Vertical displacement: $r_Z$')

			# --------- Custom plot -----------
		ax.set_xlabel(r'Longitudinal distance; $X$ [m]')
		#ax.set_xlim(0,)
		ax.set_ylabel('Displacements [mm]')

		# ------ legend ------
		# add the legend in the upper right corner of the plot
		leg = ax.legend(fancybox=True, loc='best', shadow=False)
		# set the alpha value of the legend: it will be translucent
		leg.get_frame().set_alpha(0.6)
		filename = 'displacements_allrequestednodes_MDyn.pdf'

		plt.savefig(os.path.join(results_dir, filename))
		plt.close()

	def writeOutputFileTMD(self,rout,flagoutput):
		results_dir = os.path.join(self.script_dir, self.caseName+"/time_histories/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)

		for i in range(len(self.TMDNodes)):
			TMDtoSave = self.TMDNodes[i]
			#DOFTMDtoSave = self.indexDOFTMD[i]

			filename = flagoutput+'_TMDatNode_'+str(TMDtoSave)+'_MDyn'
			fichero = open(os.path.join(results_dir, filename)+'.txt' , 'w')

			if flagoutput == 'displacement':
				flagoutputplot = 'Displacements [mm]'
				fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Total movement along DOF of TMD [mm]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")
			elif flagoutput == 'velocity':
				flagoutputplot = 'Velocities [mm/s]'
				fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Total velocity along DOF of TMD [mm/s]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")

			elif flagoutput == 'acceleration':
				flagoutputplot = r'Accelerations [mm/s$^2$]'
				fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Total acceleration along DOF of TMD [mm/s]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")


			for j in range(len(self.timeOutput)):
				fichero.write('%.4f %2.4f \n' % (self.timeOutput[j],rout[i][j]*1e3))

			fichero.close()

			# Plot figure rZ
			# ............
			fig=plt.figure()
			ax=fig.add_subplot(111)
			# ............
			ax.plot(self.timeOutput,rout[i]*1e3,'-',linewidth=1,markersize=6,markevery=1,color='red',label = r'Total TMD movement')

			# --------- Custom plot -----------
			ax.set_xlabel(r'Time; $t$ [s]')
			ax.set_xlim(0,)
			ax.set_ylabel(flagoutputplot)
			# ------ legend ------
			# add the legend in the upper right corner of the plot
			leg = ax.legend(fancybox=True, loc='best', shadow=False)
			# set the alpha value of the legend: it will be translucent
			leg.get_frame().set_alpha(0.6)
			filename = flagoutput+'_TMDatNode_'+str(TMDtoSave)+'_MDyn_plot_time.pdf'

			plt.savefig(os.path.join(results_dir, filename))
			plt.close()




	def writeOutputFileConvCheck(self,rout):
		results_dir = os.path.join(self.script_dir, self.caseName+"/time_histories/")
		if not os.path.isdir(results_dir):
			os.makedirs(results_dir)

		filename = 'interaction_convergency_check_MDyn'
		fichero = open(os.path.join(results_dir, filename)+'.txt' , 'w')
		fichero.write("""# File with:
# 	Col 1 --> time [s]
# 	Col 2 --> Number of iterations performed
# 	Col 3 --> Maximum diference between displacement (UY or UZ) along wind beams in the current iteration and the previous one
#	Col 4 --> Threshold quality reached in current iteration? (0 no, 1 yes)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
""")


		for j in range(len(self.timeOutput)):
			fichero.write('%.4f %i %2.4f %i \n' % (self.timeOutput[j],int(rout[0,j]),rout[1,j],int(rout[2,j])))

		fichero.close()



############### READ FILES


def readBridgeInfoFiles(NodeInformationFile,ModalInformationFile,FrequencyInformationFile,BeamInformationFile,DOFactive):



	NodeInformation = np.loadtxt(NodeInformationFile)
	NodeNumber = NodeInformation[:,0]
	NodeX = NodeInformation[:,1]
	NodeY = NodeInformation[:,2]
	NodeZ = NodeInformation[:,3]

	NumberOfNodes = len(NodeNumber) # Number of nodes including the platforms and joints

	ModalInformation = np.loadtxt(ModalInformationFile)

	NumberOfModes = int(len(ModalInformation[:,0])/NumberOfNodes)

	Phi = np.zeros((len(DOFactive)*NumberOfNodes,NumberOfModes))

	for i in range(NumberOfModes):
		k=0
		for j in DOFactive: # range(6):	# j = DOF - 1
			Phi[(k*NumberOfNodes):((k+1)*NumberOfNodes),i] = ModalInformation[i*NumberOfNodes:(i*NumberOfNodes)+NumberOfNodes,j-1]
			k = k+1

	FrequencyInformation = np.loadtxt(FrequencyInformationFile)
	wnv = FrequencyInformation[:,0]*2*np.pi		# Frequency in rad/s
	xinv = FrequencyInformation[:,1]/100   		# Damping ratio wrt 1 (not %)


	BeamInformation = np.loadtxt(BeamInformationFile)
	BeamNumber = BeamInformation[:,0]		# Element number
	BeamLength = BeamInformation[:,1] 	  	# Element length
	BeamNode1 = BeamInformation[:,2]		# Element node 1
	BeamNode2 = BeamInformation[:,3]		# Element node 2
	BeamSectionLabel = BeamInformation[:,4]		# Element section label (0 for platform)

	return NodeNumber,NodeX,NodeY,NodeZ,NumberOfNodes,NumberOfModes,Phi,wnv,xinv,BeamNumber,BeamLength,BeamNode1,BeamNode2,BeamSectionLabel
