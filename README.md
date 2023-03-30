# Torque tracking - notice

## HRP5 joints name

* LCY -> left vertical hip pivot
* LCR -> left lateral hip joint
* LCP -> left forward/backward hip joint
* LKP -> left knee pivot 
* LAP -> left forward/backward ankle joint 
* LAR -> left lateral ankle joint
* RCY -> right vertical hip pivot
* RCR -> right lateral hip pivot
* RCP -> right forward/backward hip joint
* RKP -> right knee pivot
* RAP -> right forward/backward ankle joint
* RAR -> left lateral ankle joint
* WP -> forward/backward abdominal pivot
* WR -> lateral abdominal pivot 
* WY -> vertical abdominal pivot
* HY -> vertical head pivot
* HP -> horizontal head pivot
* LSC -> left shoulder vertical joint
* LSP -> left shoulder forward/backward pivot
* LSR -> left shoulder lateral joint
* LSY -> left elbow lateral joint
* LEP -> left elbow joint
* LWRY -> left wrist axial joint
* LWRR -> left wrist up/down joint
* LWRP -> left wrist lateral joint
* LHDY -> left effector
* RSC -> right shoulder vertical joint
* RSP -> right shoulder forward/backward pivot
* RSR -> right shoulder lateral joint
* RSY -> right elbow lateral joint
* REP -> right elbow joint
* RWRY -> right wrist axial joint
* RWRR -> right wrist up/down joint
* RWRP -> right wrist lateral joint
* RHDY -> right effector

[info] [mc_mujoco] RCY, pgain = 5400.0, dgain = 54.0
[info] [mc_mujoco] RCR, pgain = 12925.0, dgain = 103.4
[info] [mc_mujoco] RCP, pgain = 16000.0, dgain = 160.0
[info] [mc_mujoco] RKP, pgain = 36000.0, dgain = 360.0
[info] [mc_mujoco] RAP, pgain = 8000.0, dgain = 80.0
[info] [mc_mujoco] RAR, pgain = 2244.0, dgain = 22.44
[info] [mc_mujoco] LCY, pgain = 5400.0, dgain = 54.0
[info] [mc_mujoco] LCR, pgain = 12925.0, dgain = 103.4
[info] [mc_mujoco] LCP, pgain = 16000.0, dgain = 160.0
[info] [mc_mujoco] LKP, pgain = 36000.0, dgain = 360.0
[info] [mc_mujoco] LAP, pgain = 8000.0, dgain = 80.0
[info] [mc_mujoco] LAR, pgain = 2244.0, dgain = 22.44
[info] [mc_mujoco] WP, pgain = 8000.0, dgain = 80.0
[info] [mc_mujoco] WR, pgain = 8000.0, dgain = 80.0
[info] [mc_mujoco] WY, pgain = 8000.0, dgain = 80.0
[info] [mc_mujoco] HY, pgain = 1311.0, dgain = 13.11
[info] [mc_mujoco] HP, pgain = 1311.0, dgain = 13.11
[info] [mc_mujoco] RSC, pgain = 3235.0, dgain = 32.35
[info] [mc_mujoco] RSP, pgain = 3235.0, dgain = 32.35
[info] [mc_mujoco] RSR, pgain = 5586.0, dgain = 55.86
[info] [mc_mujoco] RSY, pgain = 1429.0, dgain = 10.83
[info] [mc_mujoco] REP, pgain = 8705.0, dgain = 87.05
[info] [mc_mujoco] RWRY, pgain = 1386.0, dgain = 13.87
[info] [mc_mujoco] RWRR, pgain = 1386.0, dgain = 13.87
[info] [mc_mujoco] RWRP, pgain = 1492.0, dgain = 11.19
[info] [mc_mujoco] RHDY, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RTMP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RTPIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RTDIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RIMP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RIPIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RIDIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RMMP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RMPIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] RMDIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LSC, pgain = 3235.0, dgain = 32.35
[info] [mc_mujoco] LSP, pgain = 3235.0, dgain = 32.35
[info] [mc_mujoco] LSR, pgain = 5586.0, dgain = 55.86
[info] [mc_mujoco] LSY, pgain = 1429.0, dgain = 10.83
[info] [mc_mujoco] LEP, pgain = 8705.0, dgain = 87.05
[info] [mc_mujoco] LWRY, pgain = 1386.0, dgain = 13.87
[info] [mc_mujoco] LWRR, pgain = 1386.0, dgain = 13.87
[info] [mc_mujoco] LWRP, pgain = 1492.0, dgain = 11.19
[info] [mc_mujoco] LHDY, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LTMP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LTPIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LTDIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LIMP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LIPIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LIDIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LMMP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LMPIP, pgain = 693.0, dgain = 6.93
[info] [mc_mujoco] LMDIP, pgain = 693.0, dgain = 6.93




## Change mujoco simulation parameters

default joint friction values for HRP5P :  
damping = 0.2
frictionLoss = 0 
Both values are set to 0 to enable custom joint friction model.

## Creation of training and validation datasets

    V=$(find ~/Torque_tracking_logs/train -name *.bin);
    for i in {RCY,RCR,RCP,RKP,RAP,RAR,LCY,LCR,LCP,LKP,LAP,LAR}; 
        do echo $i; 
        mkdir ../data/$i; 
        for b in $V; 
            do echo $b; 
            python data_extractor.py --logpath $b --out ../data/$i/train.txt  --joint-name $i --append ; 
        done; 
    done

## Traing the MLP using the datasets

    python train.py --dataset data/ --batch_size 256 --outdir exps/train1 --epochs 100
