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