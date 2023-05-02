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

    for jId in range(n_DOFs) :

        tauIn[jId,:] = log['tauIn_' + repr(jId)][::samplingStep] 
        cmdTau[jId,:] = log['cmdTau_' + repr(jId)][::samplingStep] 
        qIn[jId,:] = log['qIn_' + repr(jId)][::samplingStep] 
        qIn_diff[jId,:] = log['alphaIn_' + repr(jId)][::samplingStep] 


    
    for i in range(len(sampledTime)) :

        qDot = np.zeros(n_DOFs)
        tauRef = np.zeros(n_DOFs)
        error = np.zeros(n_DOFs)


        

        t      =    sampledTime[i]                          # sample Id 
        alpha   =    qIn_diff[:,i]                          # joint velocity 
        tauRef =    cmdTau[:,i]                             # desired torque
        tauReal  =    tauIn[:,i]                            # measured torque
        z = np.random.default_rng().uniform(0,1,7)          # random seed

        outline =       repr(t)      + \
                ',' + ','.join(map(str, alpha))   + \
                ',' + ','.join(map(str, tauRef)) + \
                ',' + ','.join(map(str, tauReal)) + \
                ',' + ','.join(map(str, z))

        with open(args.out, 'a') as f:
            f.write(outline + '\n')

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


