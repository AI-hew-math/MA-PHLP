#!/bin/bash

for name in NS USAir Power Celegans Router Yeast PB Ecoli 
do
  for i in 1 2 3 4 5 6 7 8 9 10
  do
    python PHSEAL.py --data-name $name --seed $i --graph-feature None --multi-angle True #SEAL
    python PHSEAL.py --data-name $name --seed $i --graph-feature True --multi-angle False #PHSEAL(0dim)
    python PHSEAL.py --data-name $name --seed $i --graph-feature True --multi-angle True  #MultiPHSEAL(0dim)
  done
done
