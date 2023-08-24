
import mc_log_ui
from matplotlib import pyplot as plt 
import numpy as np


logs = ['/home/gabinlembrez/Torque_tracking_logs/Kinova/real/no_compensation.bin',
        '/home/gabinlembrez/Torque_tracking_logs/Kinova/real/NN_compensation.bin',
        '/home/gabinlembrez/Torque_tracking_logs/Kinova/real/stiction_compensation.bin',
        '/home/gabinlembrez/Torque_tracking_logs/Kinova/real/stiction_NN_compensation.bin']
colors = ['teal','lightsalmon', 'blue','red']
legends = ['no compensation', 'NN compensation', 'stiction compensation', 'NN and stiction compensation']

fig1 = plt.figure()
axes = []
for i in range(7):
        axes.append(fig1.add_subplot(7,1,i+1))
for idx,log in enumerate(logs):
    n_steps = 20000
    tau_d_raw = np.zeros((7,10*n_steps))
    tau_m_raw = np.zeros((7,10*n_steps))
    tau_c_raw = np.zeros((7,10*n_steps))
    tau_d = np.zeros((7,n_steps))
    tau_m = np.zeros((7,n_steps))
    tau_c = np.zeros((7,n_steps))
    tau_d_std = np.zeros((7,n_steps))
    tau_m_std = np.zeros((7,n_steps))
    tau_c_std = np.zeros((7,n_steps))
    error = np.zeros((7,n_steps))
    error_std = np.zeros((7,n_steps))
    log = mc_log_ui.read_log(log)

    T = np.linspace(0,20,20000)
    for i in range(7):
        tau_d_raw[i,:] = log["cmdTau_"+str(i)][n_steps:11*n_steps] - np.mean(log["cmdTau_"+str(i)][n_steps:11*n_steps])
        tau_m_raw[i,:] = log["tauIn_"+str(i)][n_steps:11*n_steps] - np.mean(log["tauIn_"+str(i)][n_steps:11*n_steps])
        tau_c_raw[i,:] = log["tauOut_"+str(i)][n_steps:11*n_steps] - np.mean(log["tauOut_"+str(i)][n_steps:11*n_steps])
    for i in range(n_steps):
        error[:,i] = np.mean(tau_d_raw[:,i::n_steps]-tau_m_raw[:,i::n_steps],1)
        tau_d[:,i] = np.mean(tau_d_raw[:,i::n_steps],1)
        tau_m[:,i] = np.mean(tau_m_raw[:,i::n_steps],1)
        tau_c[:,i] = np.mean(tau_c_raw[:,i::n_steps],1)
        error_std[:,i] = np.std(tau_d_raw[:,i::n_steps]-tau_m_raw[:,i::n_steps],1)
        tau_d_std[:,i] = np.std(tau_d_raw[:,i::n_steps],1)
        tau_m_std[:,i] = np.std(tau_m_raw[:,i::n_steps],1)
        tau_c_std[:,i] = np.std(tau_c_raw[:,i::n_steps],1)


    for i in range(7):
        axes[i].fill_between(T,error[i]+error_std[i],error[i]-error_std[i],color = colors[idx],alpha=0.2, edgecolor = None,antialiased=True)
        axes[i].plot(T,error[i],color = colors[idx],linewidth=0.5,label=legends[idx])
        axes[i].legend()

fig1.tight_layout()
plt.show()

