import numpy as np
import argparse
import mc_log_ui
from matplotlib import pyplot as plt

SAMPLE_RATE = 0.010     #s
HISTORY_LEN = 0.050     #s
LOG_DT = 0.002          #s
samplingStep = int(SAMPLE_RATE/LOG_DT) # sampling step

joints=["RCY", "RCR", "RCP", "RKP", "RAP", "RAR",
        "LCY", "LCR", "LCP", "LKP", "LAP", "LAR"]

def parse(args):
    # parse mc-rtc log
    log = mc_log_ui.read_log(args.logpath)

    # get start time of 'Demo' state
    exec_main = log['Executor_Main']
    demo_start = exec_main.index('Demo')
    skip = 2 # seconds

    jname = args.joint_name

    jId = joints.index(jname)
    tauIn = log['tauIn_' + repr(jId)][demo_start:][int(skip/LOG_DT):]
    sampled_tauIn = tauIn[::samplingStep]
    cmdTau = log['tauCmd_' + repr(jId)][demo_start:][int(skip/LOG_DT):]
    sampled_cmdTau = cmdTau[::samplingStep]
    qIn = log['qIn_' + repr(jId)][demo_start:][int(skip/LOG_DT):]
    qIn_diff = [0] + (np.diff(qIn)/LOG_DT).tolist()
    sampled_qIn_diff = qIn_diff[::samplingStep]

    logtime = log['t'][demo_start:][int(skip/LOG_DT):]
    sampledTime = logtime[::samplingStep]

    for i in range(len(sampledTime)) :
        #s = np.random.choice(walk_indices)

        t =         sampledTime[i]
        qDot =      sampled_qIn_diff[i]
        tauRef =    sampled_cmdTau[i]
        error =     sampled_cmdTau[i] - sampled_tauIn[i]  

        outline =       repr(t)      + \
                  ',' + repr(qDot)   + \
                  ',' + repr(tauRef) + \
                  ',' + repr(error)

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
    parser.add_argument("--joint-name",
                        required=True,
                        type=str,
                        help="Name of joint for which to extract logs",
    )
    parser.add_argument('--append',
                        action='store_true'
    )
    args = parser.parse_args()

    # clear outfile
    if not args.append:
        open(args.out, 'w').close()
    parse(args)


