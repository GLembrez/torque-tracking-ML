import numpy as np
import argparse
import mc_log_ui
import pandas as pd

SAMPLE_RATE = 0.005     #s
LOG_DT = 0.001          #s
samplingStep = int(SAMPLE_RATE/LOG_DT) # sampling step

          


n_DOFs = 7


def parse(args):
    # parse mc-rtc log
    log = mc_log_ui.read_log(args.logpath)

    logtime = log['t']
    sampledTime = logtime[::samplingStep]
    dTau = np.zeros((n_DOFs,len(sampledTime)))
    cmdTau = np.zeros((n_DOFs,len(sampledTime)))
    alpha = np.zeros((n_DOFs,len(sampledTime)))
    q = np.zeros((n_DOFs, len(sampledTime)))
    c = np.zeros((n_DOFs, len(sampledTime)))

    for jId1 in range(n_DOFs) :

        q[jId1,:] = log['qIn_'+repr(jId1)][::samplingStep]
        cmdTau[jId1,:] = log['cmdTau_' + repr(jId1)][::samplingStep] 
        alpha[jId1,:] = log['alphaIn_' + repr(jId1)][::samplingStep] 
        dTau[jId1,:] = log['fixedPoint_' + repr(jId1)][::samplingStep]      
        c[jId1,:] = log['c_' + repr(jId1)][::samplingStep]     

    df = pd.DataFrame({"tau_d":list(cmdTau),
                       "q":list(q),
                       "alpha":list(alpha),
                       "f_point":list(dTau),
                       "c":list(c)})
    df.to_pickle(args.out)

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


