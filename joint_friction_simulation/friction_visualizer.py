import mc_log_ui
from matplotlib import pyplot as plt
import numpy as np

log = mc_log_ui.read_log('/home/gabinlembrez/Torque_tracking_logs/train/mc-control-CustomFrictionController-2023-03-31-15-06-42.bin')

tauIn = log['tauIn_' + repr(3)][10:10000]
cmdTau = log['cmdTau_' + "RKP"][10:10000]
f = cmdTau - tauIn
qIn = log['qIn_'+repr(3)][10:10001]
alpha = np.diff(qIn)/0.001
logtime = log['t'][10:10000]

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(logtime, cmdTau, color = "teal", label = "command torque", linewidth=2)
ax1.plot(logtime, tauIn, color = "lightsalmon", label = "motor torque", linewidth=2)
ax1.set_ylabel("torque [Nm]",fontsize=14)
plt.legend(fontsize=14)
ax2 = fig.add_subplot(312)
ax2.plot(logtime, alpha, color = "teal", label = "joint velocity", linewidth=2)
ax2.set_ylabel("rotation velocity [rad/s]",fontsize=14)
plt.legend(fontsize=14)
ax3 = fig.add_subplot(313)
ax3.plot(logtime, f, color = "teal", label = "torque friction error", linewidth=2)
ax3.set_ylabel("torque [Nm]",fontsize=14)
plt.legend(fontsize=14)
ax3.set_xlabel("time [s]",fontsize=14)
plt.show()

