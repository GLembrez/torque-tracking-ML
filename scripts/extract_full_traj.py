import numpy as np
import random
import argparse
import mc_log_ui
from matplotlib import pyplot as plt

SAMPLE_RATE = 0.010 #s
HISTORY_LEN = 0.050 #s
LOG_DT = 0.002 #s

POLICY_IN = 80

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
    cmdTau = log['tauCmd_' + repr(jId)][demo_start:][int(skip/LOG_DT):]

    qIn = log['qIn_' + repr(jId)][demo_start:][int(skip/LOG_DT):]
    qIn_diff = [0] + (np.diff(qIn)/LOG_DT).tolist()

    mode = log['WalkingStatePolicy_inputs_' + repr(POLICY_IN)][demo_start:][int(skip/LOG_DT):-int(skip/LOG_DT)]
    stand_indices = np.where(mode==0.)[0].tolist()
    walk_indices = np.where(mode==1.)[0].tolist()
    stand_indices = np.random.choice(stand_indices, int(args.num_clips/2)).tolist()
    walk_indices = np.random.choice(walk_indices, int(args.num_clips/2)).tolist()

    logtime = log['t'][demo_start:][int(skip/LOG_DT):]

    # Segment into clips of len HISTORY_LEN
    #for clip_id in range(args.num_clips):
    collectTauIn = []
    collectT = []

    for t in range(5,len(logtime)) :
            qDot = [qIn_diff[i] for i in range(t-5,t)]  #store velocities
            tauRef = [cmdTau[i] for i in range(t-5,t)]  #store command torque
            gt = cmdTau[t]  - tauIn[t]          # target = actual real motor torque
            outline = ','.join(map(str, qDot)) + ',' + \
                    ','.join(map(str, tauRef))  +\
                    ',' + repr(gt)
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
    parser.add_argument("--num-clips",
                        required=False,
                        type=int,
                        help="number of clips to generate for each joint",
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


