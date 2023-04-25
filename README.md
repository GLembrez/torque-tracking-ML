# Torque tracking - notice

## HRP5 joints name

| name | descritpion |
|:--------:|:-----:|
| LCY | left vertical hip pivot |
| LCR | left lateral hip joint |
| LCP | left forward/backward hip joint |
| LKP | left knee pivot |
| LAP | left forward/backward ankle joint | 
| LAR | left lateral ankle joint |
| RCY | right vertical hip pivot |
| RCR | right lateral hip pivot |
| RCP | right forward/backward hip joint |
| RKP | right knee pivot |
| RAP | right forward/backward ankle joint |
| RAR | left lateral ankle joint | 
| WP | forward/backward abdominal pivot |
| WR | lateral abdominal pivot  |
| WY | vertical abdominal pivot |
| HY | vertical head pivot |
| HP | horizontal head pivot |
| LSC | left shoulder vertical joint |
| LSP | left shoulder forward/backward pivot |
| LSR | left shoulder lateral joint |
| LSY | left elbow lateral joint |
| LEP | left elbow joint | 
| LWRY | left wrist axial joint |
| LWRR | left wrist up/down joint |
| LWRP | left wrist lateral joint |
| LHDY | left effector |
| RSC | right shoulder vertical joint |
| RSP | right shoulder forward/backward pivot |
| RSR | right shoulder lateral joint |
| RSY | right elbow lateral joint |
| REP | right elbow joint |
| RWRY | right wrist axial joint |
| RWRR | right wrist up/down joint |
| RWRP | right wrist lateral joint |
| RHDY | right effector |





## Dataset generation using mc-mujoco

### Change mujoco simulation parameters

default joint friction values for HRP5P :  
damping = 0.2
frictionLoss = 0 
Both values are set to 0 to enable custom joint friction model.

### Create and export logs

Ensure that the correct controller is loaded in mc-rtc.yaml config file. The following command execute the simulation $N$ time. Each execution takes $T$ seconds. the logs are stored in /tmp/. $T$ should never exceed 60 to ensure the log is small enough to be processed.

    for i in {1..10} ;
        do (mc_mujoco --torque-control) & sleep 30 ; kill $! ;
    done ;

Next the files are moved from /tmp to the log folder

    V=$(find /tmp -name *.bin 2>/dev/null);
    for b in $V;
        do cp $b /home/gabinlembrez/Torque_tracking_logs/Kinova/train;
    done;

Extract the dataset from the logs

    V=$(find ~/Torque_tracking_logs/Kinova/train -name *.bin);
    for b in $V;
        do echo $b; 
        python /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/data_extractor.py --logpath $b --out /home/gabinlembrez/data/Kinova/train.txt --append ; 
    done ;

Repeat the operation for valid dataset.

    python /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/train.py --dataset /home/gabinlembrez/GitHub/torque-tracking-ML/data/Kinova --batch_size 64 --outdir /home/gabinlembrez/trained_nets/Kinova_multivariate_v0/ --epochs 100



