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


#### CPU Multiprocessing setting

`--num-cpu` : The number of cpus for multiprocessing. Defalut is `1`.

`--multiprocess` : You can turn off cpu multiprocessing if set to `False`. Defalut is `True`


#### Model Hyperparameters

`--seed`: random seed. Default is `1`.

`--lr`: learning rate. Default is `0.00005`.

`--dropout` : dropout ratio. Default is `0.5`.

`--hidden-channels`: Default is `1024`.

`--num-layers`: The number of fully connected layers. Default is `3`.

`--batch-size`: Default is `1024`.

`--epoch-num`: Default is `10000`.

`--patience`: Patience of early stopping. Default is `7`.

