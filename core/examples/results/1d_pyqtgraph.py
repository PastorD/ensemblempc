import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget

import pyqtgraph as pg
import numpy as np

import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt

import random


# Load Simulation Data
#B_ep, N_ep, mpc_cost_ep, t_eval, x_ep, x_th, u_th, ground_altitude = 
result = loadmat('./core/examples/1d_drone.mat')
B_ep = result['B_ep']
N_ep = result['N_ep']
mpc_cost_ep = result['mpc_cost_ep']
t_eval = result['t_eval']
x_ep = result['x_ep']
x_th = result['x_th']
u_th = result['u_th']
u_ep = result['u_ep']
ground_altitude = result['ground_altitude']
T_hover = result['T_hover']

ii = 0

Nt = len(x_th[0])
Ne = len(x_th[0][0])


class Slider(QWidget):
    def __init__(self, minimum, maximum, variable, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = QHBoxLayout(self)
        self.label = QLabel(self)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QVBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.variable = variable
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.label.setText(f"{self.variable}: \n {int(self.x)}")


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent=parent)

        # Load Data

        Nep = len(x_ep)
        self.horizontalLayout = QVBoxLayout(self)
        self.w1 = Slider(0, Nep-1, 'Episode')
        self.horizontalLayout.addWidget(self.w1)

        self.w2 = Slider(0, Nt-1, 'Time to plot MPC prediction')
        self.horizontalLayout.addWidget(self.w2)

        self.win = pg.GraphicsWindow(title="EnMPC Analyzer")
        self.horizontalLayout.addWidget(self.win)
        self.win.setBackground('w')

        self.position_plot = self.win.addPlot(title="Position(m)")
        self.position_plot.setRange(xRange=[0,t_eval.max()*2])
        self.position_plot.setRange(yRange=[x_th.min()-0.1,x_th.max()+0.1])
        self.simpos = self.position_plot.plot(pen=pg.mkPen('b', width=5))
        self.simpos.setData(t_eval[0,:],x_ep[ii][0,:])
        self.pos_curve = [self.position_plot.plot(pen=pg.mkPen(color=(100, int(random.uniform(0,255)), 0))) for i in range(Ne)]
        self.pos_point = self.position_plot.plot(pen='r', symbol='o',symbolSize=10) #, symbolBrush=(100, 100, 255, 50))
        self.position_plot.showGrid(x=True, y=True)

        self.velocity_plot = self.win.addPlot(title="Velocity(m/s)")
        self.simvel = self.velocity_plot.plot(pen=pg.mkPen('b', width=5))
        self.simvel.setData(t_eval[0,:],x_ep[ii][1,:])
        self.vel_curve = [self.velocity_plot.plot(pen=pg.mkPen(color=(255, int(random.uniform(0,255)), 0))) for i in range(Ne)]
        self.vel_point = self.velocity_plot.plot(pen='r', symbol='o',symbolSize=10) #, symbolBrush=(100, 100, 255, 50))
        self.velocity_plot.showGrid(x=True, y=True)
        #self.velocity_plot.setBackground('w')



        self.u_plot = self.win.addPlot(title="Input")
        #self.u_plot.setBackground('w')
        self.simu = self.u_plot.plot(pen=pg.mkPen('b', width=5))
        self.simu.setData(t_eval[0,:-1],u_ep[ii][0,:])
        self.u_curve = self.u_plot.plot(pen='r')
        self.u_point = self.u_plot.plot(pen='r', symbol='o',symbolSize=10) #, symbolBrush=(100, 100, 255, 50))
        self.u_plot.showGrid(x=True, y=True)



        
        self.update_plot()

        self.w1.slider.valueChanged.connect(self.update_plot)
        self.w2.slider.valueChanged.connect(self.update_plot)

    def update_plot(self):
        ii = int(self.w1.x)
        step = int(self.w2.x)

        x_plot = t_eval[0,:]+t_eval[0,step] 

        self.simpos.setData(t_eval[0,:],x_ep[ii][0,:])
        self.simvel.setData(t_eval[0,:],x_ep[ii][1,:])
        self.simu.setData(t_eval[0,:-1],u_ep[ii][0,:])

        pos_data = [np.hstack([x_ep[ii][0,step],x_en[0,:]]) for x_en in x_th[ii][step]]
        [curve_member.setData(x_plot,data_member) for curve_member, data_member in zip(self.pos_curve,pos_data)]
        self.pos_point.setData(np.array([t_eval[0,step]]),np.array([x_ep[ii][0,step]]))


        vel_data = [np.hstack([x_ep[ii][1,step],x_en[1,:]]) for x_en in x_th[ii][step]]
        [curve_member.setData(x_plot,data_member) for curve_member, data_member in zip(self.vel_curve,vel_data)]
        self.vel_point.setData(np.array([t_eval[0,step]]),np.array([x_ep[ii][1,step]]))

        u_data = np.squeeze(u_th[ii][step][0,:]+T_hover)
        print(u_data.shape)
        print(T_hover)
        self.u_curve.setData(x_plot[:-1],u_data)
        self.u_point.setData(np.array([t_eval[0,step]]),np.array([u_ep[ii][0,step]]))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())