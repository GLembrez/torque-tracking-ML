#!/bin/bash


## Generate train dataset

for i in {1..5} ;
    do (mc_mujoco --torque-control --without-visualization) & sleep 30 ; kill $! ;
done ;

V=$(find /tmp -name *.bin -not -name "*latest.bin" 2>/dev/null);
for b in $V;
    do mv $b /home/gabinlembrez/Torque_tracking_logs/Kinova/train;
done;

V=$(find ~/Torque_tracking_logs/Kinova/train -name *.bin);
for b in $V;
    do echo $b; 
    python /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/data_extractor.py --logpath $b --out /home/gabinlembrez/data/Kinova/train.txt --append ; 
done ;


## Generate validation dataset

for i in {1..2} ;
    do (mc_mujoco --torque-control --without-visualization) & sleep 10 ; kill $! ;
done ;

V=$(find /tmp -name *.bin -not -name "*latest.bin" 2>/dev/null);
for b in $V;
    do mv $b /home/gabinlembrez/Torque_tracking_logs/Kinova/valid;
done;

V=$(find ~/Torque_tracking_logs/Kinova/valid -name *.bin);
for b in $V;
    do echo $b; 
    python /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/data_extractor.py --logpath $b --out /home/gabinlembrez/data/Kinova/valid.txt --append ; 
done ;


## train

python /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/train.py --dataset /home/gabinlembrez/data/Kinova --batch_size 64 --outdir /home/gabinlembrez/trained_nets/Kinova_multivariate_v3/ --epochs 100


## Validation on real data

rm /home/gabinlembrez/data/Kinova/real.txt ;
V=$(find ~/Torque_tracking_logs/Kinova/real -name *.bin);
for b in $V;
    do echo $b; 
    python /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/data_extractor.py --logpath $b --out /home/gabinlembrez/data/Kinova/real.txt --append ; 
done ;

python /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/predict.py --model /home/gabinlembrez/trained_nets/Kinova_multivariate_v3/trained_50.model --dataset /home/gabinlembrez/data/Kinova/real.txt --visualize
