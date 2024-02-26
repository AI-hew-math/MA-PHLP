#!/bin/bash

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")
EXPERIMENT_DIR="$PARENT_DIR/src"

# # Original WalkPooling
# for name in Celegans Power USAir NS Celegans Router Yeast PB Ecoli 
# do
#   for i in 1 2 3 4 5 6 7 8 9 10
#   do
#     python $EXPERIMENT_DIR/main.py --data-name $name --num-cpu 32 --epoch-num 10 --PH False --seed $i
#   done
# done

# WalkPooling with Persistent homology (Angle-multi focus)
for name in Power Celegans USAir NS Router Yeast PB Ecoli 
do
  for i in 1 2 3 4 5 6 7 8 9 10
  do
    python $EXPERIMENT_DIR/main.py --data-name $name --num-cpu 32 --epoch-num 10 --PH True --multi-angle True --seed $i
  done
done


# # WalkPooling with Persistent homology (Angle)
# for name in Celegans Power USAir NS Celegans Router Yeast PB Ecoli 
# do
#   for i in 1 2 3 4 5 6 7 8 9 10
#   do
#     python $EXPERIMENT_DIR/main.py --data-name $name --num-cpu 32 --epoch-num 10 --PH True --multi-angle False --seed $i
#   done
# done
