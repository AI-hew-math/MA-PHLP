# Multi_PHLP + SEAL


## About

This is the source code for paper _Multi Persistent Homology for Link Prediction_ for part Multi_PHLP + SEAL.  

The code is adapted from SEAL's original implementation. [link](https://github.com/muhanzhang/SEAL)

## Requirements

## Run

### Quick start

~~~
python ./Python/PHSEAL.py --data-name USAir --graph-feature True --multi-angle True
~~~

### Parameters for PHLP

`--graph-feature` : You can add PHLP tp SEAL if it is not `None` and only use SEAL if `False`. Defalut is `None`.

`--multi-angle` : You can add Multi-Angle PHLP if `True` and PHLP if `False`. Defalut is `False`. This works only if `--graph-feature` is not `None`.