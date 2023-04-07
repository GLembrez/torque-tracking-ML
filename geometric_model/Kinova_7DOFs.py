import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from random import random

def homogeneous_transform(d,alpha,r,theta):
    # Compute homogeneous tranform i-1 -> i
    # uses modified Denavit Hartenberg parameters :
    # d : translation zi-1 -> zi along xi-1
    # r : translation xi-1 -> xi along zi
    # alpha : rotation zi-1 -> zi around xi-1
    # theta : rotation xi-1 -> xi around zi

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    ct = np.cos(theta)
    st = np.sin(theta)
    T = np.array([
        [ct   , -st  , 0  , d    ],
        [st*ca, ca*ct, -sa, -sa*r],
        [st*sa, ct*sa,  ca,  ca*r],
        [0    , 0    , 0  , 1    ]
    ])
    return T

def forward_kinematics(N,d_list,alpha_list,r_list,q):
    # computes homogeneous tranform base -> effector of a N DOFs robot 
    # d, alpha, r and q are the modified Denavit Hartenberg parameters list

    T = np.eye(4)
    for i in range(N):
        Ti = homogeneous_transform(d_list[i],alpha_list[i],r_list[i],q[i])
        T = np.dot(T,Ti)
    return T

def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.
    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

### GEOMETRIC CONSTANTS ####

r1 = 0.1564 + 0.1284
r3 = 0.2104 + 0.2104
r5 = 0.1059 + 0.2084
r7 = 0.1059 + 0.0615
n_DOFs = 7 
d_list = np.zeros(7)
alpha_list = np.array([0,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2])
r_list = np.array([r1,0,r3,0,r5,0,r7])


### FORWARD KINEMATICS ###

q = np.zeros(n_DOFs)
q_inf = np.array([-np.pi,-2.15,-np.pi,-2.45,-np.pi,-2,-np.pi])
q_sup = np.array([np.pi,2.15,np.pi,2.45,np.pi,2,np.pi])



for i in range(3) :
    for j in range(7) :
        q[j] = random() *  (q_sup[j] - q_inf[j]) + q_inf[j]


    TB1 = homogeneous_transform(d_list[0],alpha_list[0],r_list[0],q[0])
    T12 = homogeneous_transform(d_list[1],alpha_list[1],r_list[1],q[1])
    T23 = homogeneous_transform(d_list[2],alpha_list[2],r_list[2],q[2])
    T34 = homogeneous_transform(d_list[3],alpha_list[3],r_list[3],q[3])
    T45 = homogeneous_transform(d_list[4],alpha_list[4],r_list[4],q[4])
    T56 = homogeneous_transform(d_list[5],alpha_list[5],r_list[5],q[5])
    T6E = homogeneous_transform(d_list[6],alpha_list[6],r_list[6],q[6])

    O = np.array([0,0,0,1])
    z = np.array([0,0,1,0])
    y = np.array([0,1,0,0])
    x = np.array([1,0,0,0])
    OB = O
    O2 = TB1.dot(T12).dot(O)
    O4 = TB1.dot(T12).dot(T23).dot(T34).dot(OB)
    O6 = TB1.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56).dot(OB)
    OE = TB1.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56).dot(T6E).dot(O)
    zE = forward_kinematics(n_DOFs,d_list,alpha_list,r_list,q).dot(z)
    yE = forward_kinematics(n_DOFs,d_list,alpha_list,r_list,q).dot(y)
    xE = forward_kinematics(n_DOFs,d_list,alpha_list,r_list,q).dot(x)



    X = [OB[0],O2[0],O4[0],O6[0],OE[0]]
    Y = [OB[1],O2[1],O4[1],O6[1],OE[1]]
    Z = [OB[2],O2[2],O4[2],O6[2],OE[2]]
    visual = plt.figure()
    ax = visual.add_subplot(111,projection='3d')
    # for i in range(10000) :
    #     for jointId in range(n_DOFs):
    #         q[jointId] = random() * (q_sup[jointId] - q_inf[jointId]) + q_inf[jointId]
    #     T = forward_kinematics(n_DOFs,d_list,alpha_list,r_list,q)
    #     OE = T.dot(O)
    #     ax.plot([OE[0]],[OE[1]],[OE[2]],'.',  color = 'teal')

    ax.plot(X,Y,Z,  color = 'black')
    ax.quiver(OE[0], OE[1], OE[2], zE[0], zE[1], zE[2], length=0.2, normalize=True, color = 'blue')
    ax.quiver(OE[0], OE[1], OE[2], yE[0], yE[1], yE[2], length=0.2, normalize=True, color = 'green')
    ax.quiver(OE[0], OE[1], OE[2], xE[0], xE[1], xE[2], length=0.2, normalize=True, color = 'red')
    print(forward_kinematics(n_DOFs,d_list,alpha_list,r_list,q))
    set_axes_equal(ax)
    plt.show()
