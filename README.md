# Multi_PHLP


## About

This is the source code for paper _Multi Persistent Homology for Link Prediction_.

## Requirements

## Run

### Quick start

~~~
python main.py --data-name USAir
~~~

### Parameters

#### Data and Persistent Homology

`--data-name` : supported data

`--test-ratio` and `--val-ratio`: Test ratio and validation ratio of the data. Defaults are `0.1` and `0.05` respectively.

`--practical-neg-sample`: whether only see the train positive edges when sampling negative.

`--Max-hops`: number of maximum hops in sampling subgraph. Defalut is `3`.

`--max-nodes-per-hop`: When the graph is too large or too dense, we need max node per hop threshold to avoid OOM. Default is `100`.

`--starting-hop-of-max-nodes` : The hop at the start of applying max nodes.

`--node-label` : node labeling option `drnl`, `degdrnl`. Default is `degdrnl`.

`--deg-cut` : When using `degdrnl`, maximum number of degrees to count.


#### Model Hyperparameters

`--seed`: random seed. Default is `1`.

`--lr`: learning rate. Default is `0.00005`.

`--dropout` : dropout ratio. Default is `0.5`.

`--activation` : activation function. Default is `relu`.

`--batch-nomalize` : whether to use batch nomalization. Default is `False`.

`--weight-initialization` : whether to use batch nomalization. Default is `False`.

`--hidden-channels`: Default is `1024`.

`--num-layers`: The number of fully connected layers. Default is `3`.

`--batch-size`: Default is `1024`.

`--epoch-num`: Default is `10000`.

`--patience`: Patience of early stopping. Default is `7`.

