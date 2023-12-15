# Multi_PHLP


## About

This is the source code for paper _Multi Persistent Homology for Link Prediction_.

## Requirements

## Run

### Quick start

~~~
python main.py --data-name USAir --num-cpu 32
~~~

### Parameters

#### Data and Persistent Homology

`--data-name` : supported data

`--test-ratio` and `--val-ratio`: Test ratio and validation ratio of the data. Defaults are `0.1` and `0.05` respectively.

`--practical-neg-sample`: whether only see the train positive edges when sampling negative.

`--Max-hops`: The number of maximum hops in sampling subgraph. Defalut is `3`.

`--max-nodes-per-hop`: When the graph is too large or too dense, we need max node per hop threshold to avoid OOM. Default is `100`.

`--starting-hop-restric` : The list of hop at which the applying 'max nodes per hop' begins and the number of max nodes. Default is `[3,100]`

`--node-label` : node labeling option `drnl`, `degdrnl`. Default is `degdrnl`.

`--deg-cut` : When using `degdrnl`, maximum number of degrees to count.

`--onedim-PH` : You can only use 0 dimensional homology if `False` and 0 dimension and 1 dimension together if `True`. Defalut is `False`. 

`--multi-angle` : You can use Multi-Angle PHLP if `True` and PHLP if `False`. Defalut is `False`. 

`--angle-hop` : angle hop of PHLP. Defalut is `[3,1]`. If we use Multi-angle PHLP, it does not needed.


#### CPU Multiprocessing setting

`--num-cpu` : The number of cpus for multiprocessing. Defalut is `1`.

`--multiprocess` : You can turn off cpu multiprocessing if set to `False`. Defalut is `True`


#### Model Hyperparameters

`--seed`: random seed. Default is `1`.

`--lr`: learning rate. Default is `0.0005`.

`--dropout` : dropout ratio. Default is `0.5`.

`--num-layers`: The number of fully connected layers. Default is `3`.

`--batch-size`: Default is `1024`.

`--epoch-num`: Default is `10000`.

`--patience`: Patience of early stopping. Default is `20`.

