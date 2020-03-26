import plotly.graph_objects as go
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


#pos_plot.plot([t_eval[0], t_eval[-1]], [ground_altitude, ground_altitude], '--r', lw=2, label='Ground constraint')


# Nt = len(x_th[0])
# ensemble_time_indices = [0,int(Nt/4),int(Nt/3)]

# #b1_lst[ii].fill_between(t_eval, ref[0,:], x_ep[plot_ep[ii], :, 0], alpha=0.2)
# err_norm = (t_eval[-1]-t_eval[0])*np.sum(np.square(x_ep[plot_ep[ii], :, 0].T - ref[0,:]))/x_ep[plot_ep[ii], :, 0].shape[0]
# #b1_lst[ii].text(1.2, 0.5, "$\int (z-z_d)^2=${0:.2f}".format(err_norm))

# pos_plot[ii].set_title('Executed trajectory, ep ' + str(plot_ep[ii]))
# #pos_plot[ii].set_xlabel('Time (sec)')
# pos_plot[ii].set_ylabel('z (m)')
# pos_plot[ii].grid()

# vel_plot.append(f2.add_subplot(gs2[3, ii]))
# vel_plot[ii].plot(t_eval, x_ep[plot_ep[ii], :, 1], label='$\dot{z}$')

# for index_time_ensemble in ensemble_time_indices:
#     for x_th_en in x_th[plot_ep[ii]][index_time_ensemble]: # for every ensemble at time zero 
#         vel_plot[ii].plot(
#             t_eval+t_eval[index_time_ensemble], 
#             np.hstack([x_ep[plot_ep[ii]][index_time_ensemble,1],x_th_en[1,:]]), lw=0.5,c='k', alpha=0.5 )
# vel_plot[ii].set_ylabel('$\dot{z}$ (m/s)')
# vel_plot[ii].grid()


# u_plot.append(f2.add_subplot(gs2[4, ii]))
# u_plot[ii].plot(t_eval[:-1], u_ep[plot_ep[ii], :, 0], label='T')
# u_plot[ii].plot([t_eval[0], t_eval[-2]], [umax+T_hover, umax+T_hover], '--r', lw=2, label='Max thrust')

# for index_time_ensemble in ensemble_time_indices:
#     u_plot[ii].plot(
#             t_eval[:-1]+t_eval[index_time_ensemble], T_hover+u_th[plot_ep[ii]][index_time_ensemble][0,:], lw=0.5,c='k', alpha=0.5 )




# Create figure
fig = go.Figure()

# Add traces, one for each slider step
Nt = len(x_th[0])
Ne = len(x_th[0][0])
en = 0
for step in range(Nt):
    #y_plot = np.zeros(Nt,Ne)
    x_plot = t_eval[0,:]+t_eval[0,step] 
    #for x_th_en in x_th[ii][step]: # for every ensemble at time zero         
    #    y_plot[:,] = np.vstack([y_plot,np.hstack([x_ep[ii][step,0],x_th_en[0,:]]) ])
    y_plot = np.hstack([x_ep[ii][step,0],x_th[ii][step][en][0,:]])
    fig.add_trace( go.Scatter(
            visible=False,
            line=dict(color="#00CED1", width=3),
            name="ensemble " + str(en),
            x=x_plot,
            y=y_plot))
#pos_plot.plot(t_eval, x_ep[ii, :, 0], label='z')

fig.add_trace( go.Scatter(
        visible=True,
        line=dict(color="#AB63FA", width=4),
        name="position ep" + str(ii),
        x= t_eval[0,:],
        y= x_ep[ii][:][0]))


# Make 10th trace visible
fig.data[0].visible = True

# Create and add slider
episode = []
for i in range(len(fig.data)):
    step = dict(
        method="restyle",
        args=["visible", [False] * len(fig.data)],
    )
    step["args"][1][i] = True  # Toggle i'th trace to "visible"
    episode.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Episode: "},
    pad={"t": 20},
    steps=episode
)]

fig.update_layout(
    sliders=sliders,
    xaxis = dict(
      range=[0,t_eval.max()*2],  # sets the range of xaxis
      constrain="domain",  # meanwhile compresses the xaxis by decreasing its "domain"
    ),
    yaxis = dict(
      range=[x_th.min()-0.1,x_th.max()+0.1],  # sets the range of xaxis
      constrain="domain", 
    ),
)

fig.show()

