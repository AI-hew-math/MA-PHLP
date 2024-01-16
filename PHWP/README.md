# Multi_PHLP + WalkPooling (WP)

## About

This is the source code for the paper Multi Persistent Homology for Link Prediction, specifically for the part Multi_PHLP + WalkPooling.

The code is adapted from the original implementation of WalkPooling, which can be found [here](https://github.com/DaDaCheng/WalkPooling).

#### Quick start

~~~ run python file 
python ./src/main.py --data-name Power --PH True --multi-angle True --num-cpu 32
~~~

OR

~~~ run shell script file
./experiment/run.sh
~~~

### Parameters for adding PHLP

`--num-cpu`: Set this parameter to specify the number of CPUs used for Multiprocess in PH calculations. The default value is `1`."

`--PH`: Set this parameter to a value other than `False` to add PHLP to WP. Use only WP by setting it to `False`. The default value is `False`.

`--multi-angle`: Enable Multi-Angle PHLP by setting this to `True`, or use standard PHLP by setting it to `False`. This parameter only takes effect if `--PH` is not `False`. The default value is `False`.

# WalkPooling

## Requirements of walkpooling

python>=3.3.7

torch>=1.9.0

torch-cluster>=1.5.9

torch-geometric>=2.0.0

torch-scatter>=2.0.8

torch-sparse>=0.6.11

tqdm

This code was tested on macOS and Linux.


### Parameters of walkpooling

`--data-name`: supported data:

1. Without node attributes: USAir NS Power Celegans Router PB Ecoli Yeast

2. With node attributes: cora citeseer pubmed

`--use-splitted`: when it is `True`, we use the splitted data from [SEAL](https://github.com/muhanzhang/SEAL). When it is `False`, we will randomly split train, validation and test data.

`--data-split-num`: the index of splitted data when `--use-splitted` is `True`. From 1 to 10.

`--test-ratio` and `--val-ratio`: Test ratio and validation ratio of the data set when `--use-splitted` is False. Defaults are `0.1` and `0.05` respectively.

`--observe-val-and-injection`: whether to contain the validation set in the observed graph and apply injection trick.

`--practical-neg-sample`: whether only see the train positive edges when sampling negative.

`--num-hops`: number of hops in sampling subgraph. Default is `2`.

`--max-nodes-per-hop`: When the graph is too large or too dense, we need max node per hop threshold to avoid OOM. Default is `None`.


#### Hyperparameters of walkpooling

`--init-attribute`: the initial attribute for graphs without node attributes. options: `n2v`, `one_hot`, `spc`, `ones`, `zeros`, `None`. Default is `ones`.

`--init-representation`: node feature representation . options:  `gic`, `vgae`, `argva`, `None`. Default is `None`.

`--drnl`: whether to use drnl labeling. Default is `False`.

`--seed`: random seed. Default is `1`.

`--lr`: learning rate. Default is `0.00005`.

`-heads`: using multi-heads in the attention link weight encoder. Default is `2`.

`--hidden-channels`: Default is `32`.

`--batch-size`: Default is `32`.

`--epoch-num`: Default is `50`.