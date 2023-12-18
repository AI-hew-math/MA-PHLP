# Multi_PHLP + SEAL


## About

This is the source code for the paper Multi Persistent Homology for Link Prediction, specifically for the part Multi_PHLP + SEAL.

The code is adapted from the original implementation of SEAL, which can be found [here](https://github.com/muhanzhang/SEAL).

## Requirements
[Pytorch_DGCNN](https://github.com/muhanzhang/pytorch_DGCNN/tree/master) should be git cloned into the Python directory.


## Run

### Quick start

~~~
python ./Python/PHSEAL.py --data-name USAir --graph-feature True --multi-angle True
~~~

### Parameters for adding PHLP

`--graph-feature`: Set this parameter to a value other than `None` to add PHLP to SEAL. Use only SEAL by setting it to `None`. The default value is `None`.

`--multi-angle`: Enable Multi-Angle PHLP by setting this to `True`, or use standard PHLP by setting it to `False`. This parameter only takes effect if `--graph-feature` is not `None`. The default value is `False`.
