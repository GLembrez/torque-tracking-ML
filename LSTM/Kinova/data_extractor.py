import numpy as np
import argparse
import mc_log_ui
from matplotlib import pyplot as plt

SAMPLE_RATE = 0.005     #s
LOG_DT = 0.001          #s
samplingStep = int(SAMPLE_RATE/LOG_DT) # sampling step

          


n_DOFs = 7


def parse(args):
    # parse mc-rtc log
    log = mc_log_ui.read_log(args.logpath)

    logtime = log['t']
    sampledTime = logtime[::samplingStep]
    tauIn = np.zeros((n_DOFs,len(sampledTime)))
    cmdTau = np.zeros((n_DOFs,len(sampledTime)))
    qIn = np.zeros((n_DOFs,len(sampledTime)))
    qIn_diff = np.zeros((n_DOFs,len(sampledTime)))
    M = np.zeros((len(sampledTime),n_DOFs, n_DOFs))

    for jId in range(n_DOFs) :

        tauIn[jId,:] = log['tauIn_' + repr(jId)][::samplingStep] + np.random.normal(0,0.2, size = len(sampledTime))
        cmdTau[jId,:] = log['cmdTau_' + repr(jId)][::samplingStep] + np.random.normal(0,0.2, size = len(sampledTime))
        qIn[jId,:] = log['qIn_' + repr(jId)][::samplingStep] + np.random.normal(0,0.01, size = len(sampledTime))
        qIn_diff[jId,:] = log['alphaIn_' + repr(jId)][::samplingStep] + np.random.normal(0,0.01, size = len(sampledTime))


    
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

    # fig = plt.figure()
    # jname = "joint_2"
    # plt.plot(logtime, log['cmdTau_1'], color = "lightsalmon", linewidth=1)
    # plt.plot(sampledTime,cmdTau[1,:], ".", color = "teal")
    # plt.show()
        

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


