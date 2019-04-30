# On Evaluating the Generalization of LSTMs in Formal Languages
* This repository includes a PyTorch implementation of [On Evaluating the Generalization of LSTMs in Formal Languages](https://arxiv.org/abs/1811.01001). 
* Our paper appeared in the Proceedings of the Society for Computation in Linguistics (SCiL) 2019.

## Requirements
The code is written in Python, and requires PyTorch and a couple of other dependencies. If you would like to run the code locally, please install PyTorch by following the instructions on http://pytorch.org and then run the following command to install the other required packages, which are listed inside `requirements.txt`:
```
pip install -r requirements.txt
```

## Usage
### Training/Testing options
#### Experiment options
* `exp_type`: The experiment type. The choices are `single`, `distribution`, `window`, `hidden_units`. 
* `distribution`: The distribution regime(s). The choices are `uniform`, `u-shaped`, `left-tailed`, `right-tailed`.
* `window`: The training length window. It should be a single (or a list of) integer-pair(s) in the form of a and b, where <a href="https://www.codecogs.com/eqnedit.php?latex=a&space;\leq&space;b" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a&space;\leq&space;b" title="a \leq b" /></a>.
* `lstm_huints`: The number of hidden units in the LSTM model. It should be a single (or a list of) integer(s).

#### Language options
* `language`: The language in consideration. The choices are `ab`, `abc`, `abcd`, representing the languages <a href="https://www.codecogs.com/eqnedit.php?latex=a^n&space;b^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^n&space;b^n" title="a^n b^n" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=a^n&space;b^n&space;c^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^n&space;b^n&space;c^n" title="a^n b^n c^n" /></a>, and <a href="https://www.codecogs.com/eqnedit.php?latex=a^n&space;b^n&space;c^n&space;d^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^n&space;b^n&space;c^n&space;d^n" title="a^n b^n c^n d^n" /></a>, respectively.

#### Model and data options
* `lstm_hlayers`: The number of hidden layers in the LSTM model. It should be a single positive integer. 
* `n_trials`: The number of trials. It should be a single positive integer.
* `n_epochs`: The number of epochs per trial. It should be a single positive integer.
* `sample_size`: The number of training samples. It should be a single positive integer.
* `disp_err_n`: The total number of <a href="https://www.codecogs.com/eqnedit.php?latex=e_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?e_i" title="e_i" /></a> values in consideration. It should be a single positive integer.

## Three Simple Examples
#### Experiment I: Different Trials with the Same Experiment Setup
Suppose we would like to investigate the influence of weight initialization on the inductive capabilities of LSTM models in the task of learning the CSL <a href="https://www.codecogs.com/eqnedit.php?latex=a^n&space;b^n&space;c^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^n&space;b^n&space;c^n" title="a^n b^n c^n" /></a>. We may then run the following command:
```
python main.py --exp_type single --language abc --distribution uniform --window 1 50 --lstm_hunits 3 --disp_err_n 5
```

#### Experiment II: Different Distribution Regimes
Suppose we would like to investigate the influence of various distribution regimes on the inductive capabilities of LSTM models in the task of learning the CSL <a href="https://www.codecogs.com/eqnedit.php?latex=a^n&space;b^n&space;c^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^n&space;b^n&space;c^n" title="a^n b^n c^n" /></a>. We may then run the following command:
```
python main.py --exp_type distribution --language abc --distribution uniform u-shaped left-tailed right-tailed --window 1 50 --lstm_hunits 3 --disp_err_n 5
```

#### Experiment III: Different Training Windows
Suppose we would like to investigate the influence of the training window on the inductive capabilities of LSTM models in the task of learning the CSL <a href="https://www.codecogs.com/eqnedit.php?latex=a^n&space;b^n&space;c^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^n&space;b^n&space;c^n" title="a^n b^n c^n" /></a>. Assuming that we are considering three training windows `[1, 30]`, `[1,50]`, and `[50, 100]`, we may then run the following command:
```
python main.py --exp_type window --language abc --distribution uniform --window 1 30 1 50 50 100 --lstm_hunits 3 --disp_err_n 5
```

## Citation
If you would like to cite our work, please use the following BibTeX format:
```
@InProceedings{suzgun2019evaluating,
  title={On Evaluating the Generalization of LSTM Models in Formal Languages},
  author={Suzgun, Mirac and Belinkov, Yonatan and Shieber, Stuart M.},
  journal={Proceedings of the Society for Computation in Linguistics (SCiL)},
  pages={277--286},
  year={2019},
  month={January}
}
```

Thanks!

## Acknowledgement
We thank Sebastian Gehrmann of Harvard SEAS for his insightful comments and discussions.
