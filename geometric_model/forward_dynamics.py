import numpy as np


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


M = [1.3773,1.1636,1.1636,0.9302,0.6781,0.6781,0.364]
I = np.zeros((3,3*7))
I[:,0:0+3] = np.diag([0.00488868, 0.00457, 0.00135132])
I[:,3:3+3] = np.diag([0.0113017, 0.011088, 0.00102532])
I[:,6:6+3] = np.diag([0.0111633, 0.010932, 0.00100671])
I[:,9:9+3] = np.diag([0.00834839, 0.008147, 0.000598606])
I[:,12:12+3] = np.diag([0.00165901, 0.001596, 0.000346988])
I[:,15:15+3] = np.diag([0.00170087, 0.001641, 0.00035013])
I[:,18:18+3] = np.diag([0.00024027, 0.000222769, 0.000213961])

r1 = 0.1564 + 0.1284
r3 = 0.2104 + 0.2104
r5 = 0.1059 + 0.2084
r7 = 0.1059 + 0.0615
n_DOFs = 7 
d_list = np.zeros(7)
alpha_list = np.array([0,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2,-np.pi/2,np.pi/2])
r_list = np.array([r1,0,r3,0,r5,0,r7])
q_inf = np.array([-np.pi,-2.15,-np.pi,-2.45,-np.pi,-2,-np.pi])
q_sup = np.array([np.pi,2.15,np.pi,2.45,np.pi,2,np.pi])


q = np.zeros(7)

T = np.zeros((4,7*4))

T[:,0*4:0*4+4] = homogeneous_transform(d_list[0],alpha_list[0],r_list[0],q[0])
T[:,1*4:1*4+4] = homogeneous_transform(d_list[1],alpha_list[1],r_list[1],q[1])
T[:,2*4:2*4+4] = homogeneous_transform(d_list[2],alpha_list[2],r_list[2],q[2])
T[:,3*4:3*4+4] = homogeneous_transform(d_list[3],alpha_list[3],r_list[3],q[3])
T[:,4*4:4*4+4] = homogeneous_transform(d_list[4],alpha_list[4],r_list[4],q[4])
T[:,5*4:5*4+4] = homogeneous_transform(d_list[5],alpha_list[5],r_list[5],q[5])
T[:,6*4:6*4+4] = homogeneous_transform(d_list[6],alpha_list[6],r_list[6],q[6])


D = np.zeros((n_DOFs,n_DOFs))
transformi = np.eye(4)
for i in range(n_DOFs):
    transformi = transformi.dot(T[:,i*4:i*4+4])
    Ji = np.zeros((n_DOFs,n_DOFs))
    Oi = transformi[:3,3]
    transformk = np.eye(4)
    for k in range(i+1) : 
        transformk = transformk.dot(T[:,k*4:k*4+4]) 
        Ok = transformk[:3,3]
        zk = transformk[:3,2]
        Ji[:3,k] = zk
        Ji[4:,k] = np.cross(Ok-Oi,zk)
    if i>0 :
        D += (M[i]*Ji[:3,:].T.dot(Ji[:3,:]) + Ji[4:,:].T.dot(transformi[:3,:3]).dot(I[:,i*3:i*3+3]).dot(transformi[:3,:3].T).dot(Ji[4:,:]))
np.set_printoptions(precision=2)
print(D)

