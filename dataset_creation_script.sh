#!/bin/bash


## Generate train dataset

for i in {1..20} ;
    do (mc_mujoco --torque-control --without-visualization) & sleep 400 ; kill $! ;
done ;

V=$(find /tmp -name *.bin -not -name "*latest.bin" 2>/dev/null);
for b in $V;
    do mv $b /home/gabinlembrez/Torque_tracking_logs/Kinova/train;
done;

V=$(find ~/Torque_tracking_logs/Kinova/train -name *.bin);
for b in $V;
    do echo $b; 
    python3 /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/fixed_point_solver/data_extractor.py --logpath $b --out /home/gabinlembrez/data/Kinova/train.txt --append ; 
    rm $b
done ;


## Generate validation dataset

# for i in {1..3} ;
#     do (mc_mujoco --torque-control --without-visualization) & sleep 20 ; kill $! ;
# done ;

# V=$(find /tmp -name *.bin -not -name "*latest.bin" 2>/dev/null);
# for b in $V;
#     do mv $b /home/gabinlembrez/Torque_tracking_logs/Kinova/valid;
# done;

# V=$(find ~/Torque_tracking_logs/Kinova/valid -name *.bin);
# for b in $V;
#     do echo $b; 
#     python3 /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/data_extractor.py --logpath $b --out /home/gabinlembrez/data/Kinova/valid.txt --append ; 
#     rm $b
# done ;

# ## send data to remote pc

# cd
# cd data/Kinova/
# # beware: scp will overwrite the file if they already exist
# sshpass -p 83xb96dc scp train.txt valid.txt  glembrez@150.82.172.124:torque_tracking_ML/data/
# rm train.txt valid.txt


# ## ssh connect to remote pc

# sshpass -p 83xb96dc ssh glembrez@150.82.172.124
# cd torque_tracking_ML/
# source torch_env/bin/activate
# tmux
# python3 scripts/train.py --dataset data/ --batch_size 64 --outdir models/june5_3/ --epochs 100 --model models/may22/trained.model



###########################################################################################
#
#                                   LOCAL TRAINING
#
###########################################################################################



## train

# python3 /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/train.py --dataset /home/gabinlembrez/data/Kinova --batch_size 64 --outdir /home/gabinlembrez/trained_nets/Kinova_multivariate_v4/ --epochs 2



## Validation on real data

# rm /home/gabinlembrez/data/Kinova/real.txt ;
# V=$(find ~/Torque_tracking_logs/Kinova/real -name *.bin);
# for b in $V;
#     do echo $b; 
#     python3 /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/data_extractor.py --logpath $b --out /home/gabinlembrez/data/Kinova/real.txt --append ; 
# done ;

# python3 /home/gabinlembrez/GitHub/torque-tracking-ML/LSTM/Kinova/predict.py --model /home/gabinlembrez/trained_nets/Kinova_multivariate_v3/trained.model --dataset /home/gabinlembrez/data/Kinova/real.txt --visualize
