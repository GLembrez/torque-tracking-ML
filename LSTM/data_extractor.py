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

jointIdx = {"RCY" : 7, "RCR" : 8, "RCP" : 9, "RKP" : 10, "RAP" : 11, "RAR" : 12,     
        "LCY" : 1, "LCR" : 2, "LCP" : 3, "LKP" : 4, "LAP" : 5, "LAR" : 6,     
        "WP" : 13,"WR" : 14,"WY" : 15,                               
        "RSC" : 50,"RSP" : 51,"RSR" : 52,"RSY" : 53,"REP" : 54,                
        "LSC" : 32,"LSP" : 33,"LSR" : 34,"LSY" : 35,"LEP" : 36}               


n_DOFs = len(joints)


def parse(args):
    # parse mc-rtc log
    log = mc_log_ui.read_log(args.logpath)


    jname = args.joint_name

    jId = joints.index(jname)
    tauIn = log['tauIn_' + repr(jId)]
    sampled_tauIn = tauIn[::samplingStep]
    cmdTau = log['cmdTau_' + jname] 
    sampled_cmdTau = cmdTau[::samplingStep]
    qIn = log['qIn_' + repr(jId)]
    qIn_diff = [0] + (np.diff(qIn)/LOG_DT).tolist()
    sampled_qIn_diff = qIn_diff[::samplingStep]

    logtime = log['t']
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


