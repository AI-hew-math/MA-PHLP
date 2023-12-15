#!/bin/bash

# Multi PHLP
for name in NS USAir Power Celegans Router Yeast PB Ecoli 
do
  for i in 1
  do
    python main.py --data-name $name --seed $i --num-cpu 32 --onedim-PH True --multi-angle True --multiprocess True
  done
done

# Multi PHLP (0dim)
for name in NS USAir Power Celegans Router Yeast PB Ecoli 
do
  for i in 1
  do
    python main.py --data-name $name --seed $i --num-cpu 32 --onedim-PH False --multi-angle True --multiprocess True
  done
done

# Multi PHLP
for name in NS USAir Power Celegans Router Yeast PB Ecoli 
do
  for i in 1
  do
    python main.py --data-name $name --seed $i --num-cpu 32 --onedim-PH True --multi-angle False --multiprocess True
  done
done

# Multi PHLP (0dim)
for name in NS USAir Power Celegans Router Yeast PB Ecoli 
do
  for i in 1
  do
    python main.py --data-name $name --seed $i --num-cpu 32 --onedim-PH False --multi-angle False --multiprocess True
  done
done