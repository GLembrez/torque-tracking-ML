import numpy as np
import argparse
import mc_log_ui
from matplotlib import pyplot as plt

SAMPLE_RATE = 0.005     #s
LOG_DT = 0.001          #s
samplingStep = int(SAMPLE_RATE/LOG_DT) # sampling step

joints=["RCY", "RCR", "RCP", "RKP", "RAP", "RAR",     # right leg
        "LCY", "LCR", "LCP", "LKP", "LAP", "LAR",     # left leg
        "WP","WR","WY",                               # waist
        "RSC","RSP","RSR","RSY","REP",                # right arm
        "LSC","LSP","LSR","LSY","LEP"]                # left arm

jointIdx = {"RCY" : 0, "RCR" : 1, "RCP" : 2, "RKP" : 3, "RAP" : 4, "RAR" : 5,     
        "LCY" : 6, "LCR" : 7, "LCP" : 8, "LKP" : 9, "LAP" : 10, "LAR" : 11,     
        "WP" : 12,"WR" : 13,"WY" : 14,                               
        "RSC" : 17,"RSP" : 18,"RSR" : 19,"RSY" : 20,"REP" : 21,                
        "LSC" : 35,"LSP" : 36,"LSR" : 37,"LSY" : 38,"LEP" : 39}               


n_DOFs = len(joints)


def parse(args):
    # parse mc-rtc log
    log = mc_log_ui.read_log(args.logpath)

    logtime = log['t']
    sampledTime = logtime[::samplingStep]
    tauIn = np.zeros((n_DOFs,len(sampledTime)))
    cmdTau = np.zeros((n_DOFs,len(sampledTime)))
    qIn = np.zeros((n_DOFs,len(sampledTime)))
    qIn_diff = np.zeros((n_DOFs,len(sampledTime)))

    for jname in joints :

        jId = jointIdx[jname]
        tauIn[joints.index(jname),:] = log['tauIn_' + repr(jId)][::samplingStep]
        cmdTau[joints.index(jname),:] = log['cmdTau_' + jname][::samplingStep]
        qIn[joints.index(jname),:] = log['qIn_' + repr(jId)][::samplingStep]
        qIn_diff[joints.index(jname),1:] = np.diff(log['qIn_' + repr(jId)])[::samplingStep]/LOG_DT


    
    for i in range(len(sampledTime)) :

        qDot = np.zeros(n_DOFs)
        tauRef = np.zeros(n_DOFs)
        error = np.zeros(n_DOFs)


        

        t      =    sampledTime[i]
        qDot   =    qIn_diff[:,i]
        tauRef =    cmdTau[:,i]
        error  =    cmdTau[:,i] - tauIn[:,i]  

        outline =       repr(t)      + \
                ',' + ','.join(map(str, qDot))   + \
                ',' + ','.join(map(str, tauRef)) + \
                ',' + ','.join(map(str, error))

        with open(args.out, 'a') as f:
            f.write(outline + '\n')

    fig = plt.figure()
    jname = "RKP"
    plt.plot(logtime, log['cmdTau_' + jname], color = "lightsalmon", linewidth=1)
    plt.plot(sampledTime,cmdTau[joints.index(jname),:], ".", color = "teal")
    plt.show()
        

    return

if __name__=='__main__':
    # get command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--logpath",
                        required=True,
                        type=str,
                        help="path to mc_log",
    )
    parser.add_argument("--out",
                        required=True,
                        type=str,
                        help="path to output file",
    )

    parser.add_argument('--append',
                        action='store_true'
    )
    args = parser.parse_args()

    # clear outfile
    if not args.append:
        open(args.out, 'w').close()
    parse(args)


