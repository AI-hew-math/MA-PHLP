#!/bin/bash

# Multi PHLP (0dim) 
for name in USAir NS PB Yeast Celegans Power Router Ecoli
do
  for i in 1 2 3 4 5 6 7 8 9 10
  do
    python main.py --data-name $name --seed $i --num-cpu 32 --onedim-PH False --multi-angle True --multiprocess True
  done
done

# Multi PHLP
for name in USAir NS PB Yeast Celegans Power Router Ecoli 
do
  for i in 1 2 3 4 5 6 7 8 9 10
  do
    python main.py --data-name $name --seed $i --num-cpu 32 --onedim-PH True --multi-angle True --multiprocess True
  done
done

# # PHLP (0dim)
# for name in USAir NS PB Yeast Celegans Power Router Ecoli 
# do
#   for i in 1 2 3 4 5 6 7 8 9 10
#   do
#     python main.py --data-name $name --seed $i --num-cpu 32 --onedim-PH False --multi-angle False --multiprocess True
#   done
# done

# # PHLP
# for name in USAir NS PB Yeast Celegans Power Router Ecoli 
# do
#   for i in 1 2 3 4 5 6 7 8 9 10
#   do
#     python main.py --data-name $name --seed $i --num-cpu 32 --onedim-PH True --multi-angle False --multiprocess True
#   done
# done