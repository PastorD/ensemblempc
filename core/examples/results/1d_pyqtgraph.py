import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QSlider, QSpacerItem, \
    QVBoxLayout, QWidget

import pyqtgraph as pg
import numpy as np

import scipy as sp
from scipy.io import loadmat
import matplotlib.pyplot as plt


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
ground_altitude = result['ground_altitude']

ii = 0



class Slider(QWidget):
    def __init__(self, minimum, maximum, parent=None):
        super(Slider, self).__init__(parent=parent)
        self.verticalLayout = QVBoxLayout(self)
        self.label = QLabel(self)
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QHBoxLayout()
        spacerItem = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Vertical)
        self.horizontalLayout.addWidget(self.slider)
        spacerItem1 = QSpacerItem(0, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.resize(self.sizeHint())

        self.minimum = minimum
        self.maximum = maximum
        self.slider.valueChanged.connect(self.setLabelValue)
        self.x = None
        self.setLabelValue(self.slider.value())

    def setLabelValue(self, value):
        self.x = self.minimum + (float(value) / (self.slider.maximum() - self.slider.minimum())) * (
        self.maximum - self.minimum)
        self.label.setText("{0:.4g}".format(self.x))


class Widget(QWidget):
    def __init__(self, parent=None):
        super(Widget, self).__init__(parent=parent)

        # Load Data

        self.horizontalLayout = QHBoxLayout(self)
        self.w1 = Slider(0, 1.99)
        self.horizontalLayout.addWidget(self.w1)

        self.w2 = Slider(0, 198)
        self.horizontalLayout.addWidget(self.w2)

        self.w3 = Slider(0, 1.99)
        self.horizontalLayout.addWidget(self.w3)

        self.w4 = Slider(-10, 10)
        self.horizontalLayout.addWidget(self.w4)

        self.win = pg.GraphicsWindow(title="EnMPC Analyzer")
        self.horizontalLayout.addWidget(self.win)
        self.p6 = self.win.addPlot(title="Position(m)")
        self.curve = self.p6.plot(pen='r')
        #self.curve.setXRange(0,t_eval.max()*2)
        self.p6.setRange(xRange=[0,t_eval.max()*2])
        self.p6.setRange(yRange=[x_th.min()-0.1,x_th.max()+0.1])
        self.update_plot()

        self.w1.slider.valueChanged.connect(self.update_plot)
        self.w2.slider.valueChanged.connect(self.update_plot)
        self.w3.slider.valueChanged.connect(self.update_plot)
        self.w4.slider.valueChanged.connect(self.update_plot)

    def update_plot(self):
        ii = int(self.w1.x)
        step = int(self.w2.x)
        en = int(self.w3.x)
        d = self.w4.x
        x = np.linspace(0, 10, 100)
        #data = a + np.cos(x + c * np.pi / 180) * np.exp(-b * x) * d
        x_plot = t_eval[0,:]+t_eval[0,step] 
        data = np.hstack([x_ep[ii][step,0],x_th[ii][step][en][0,:]])
        self.curve.setData(x_plot,data)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())