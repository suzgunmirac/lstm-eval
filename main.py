import numpy as np
import argparse
import sys
import random

import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

from sample_generator import SampleGenerator
from model import MyLSTM


MAX_INT = sys.maxsize

first_k_errors = 5

# EPSILON VALUE
epsilon = 0.5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## GPU 

def get_args ():
    parser = argparse.ArgumentParser(description='Training an LSTM model')

    # experiment type
    parser.add_argument ('--exp_type', required=True, type=str, default='single', choices=['single', 'distribution', 'window', 'hidden_units'], help='The experiment type.')

    # required params
    parser.add_argument ('--language', type=str, default='abc', choices=['ab', 'abc', 'abcd'], help='The language in consideration.')
    parser.add_argument ('--distribution', type=str, default=['uniform'], nargs='*', choices = ['uniform', 'u-shaped', 'left-tailed', 'right-tailed'], help='A list of distribution regimes for our training set (e.g. \'uniform\' \'u_shaped\' \'left_tailed\' \'right_tailed\').')
    parser.add_argument ('--window', type=int, default=[1,50], nargs='*',help='A list of length windows for our training set (e.g. 1 30 1 50 50 100 for 1-30, 1-50, 50-100).')
    parser.add_argument ('--lstm_hunits', type=int, default=[3], nargs='*', help='A list of hidden units for our LSTM for a given language (e.g. 4 10 36).')
    
    # optional params
    parser.add_argument ('--lstm_hlayers', type=int, default=1, help='The number of layers in the LSTM network.')

    parser.add_argument ('--sample_size', type=int, default=1000, help='The total number of training samples.')
    parser.add_argument ('--n_epochs', type=int, default=150, help='The total number of epochs.')
    parser.add_argument ('--n_trials', type=int, default=10, help='The total number of trials.') 

    parser.add_argument ('--disp_err_n', type=int, default=5, help='The first k error values.')
    
    params, _ = parser.parse_known_args ()
    
    # Print the entire list of params
    print(params)

    return params


def plot_graphs (lang, info, labels, accuracy_vals, loss_vals, window, filename):
    accuracy_vals = np.array (accuracy_vals)

    domain = list(range(1, accuracy_vals.shape[2] + 1))

    # for plotting purposes
    # max_y = np.max(accuracy_vals) + 10
    
    e_nums = [1, first_k_errors] 

    border1 = np.ones(len(domain)) * window[0]
    border2 = np.ones(len(domain)) * window[1]

    for err_n in e_nums:
        plt.figure ()
        for i in range(len(labels)):
            acc = np.array (accuracy_vals[i])
            acc_avg = np.average (acc, axis=0).T
            loss = np.array (loss_vals[i])
            plt.plot (domain, acc_avg[err_n-1], '.-', label=labels[i])

        plt.legend(loc='upper left')

        if labels != 'window':
            plt.plot (domain, border1, 'c-', label='Threshold$_1$')
            plt.plot (domain, border2, 'c-', label='Threshold$_2$')

        lang_str = '^n'.join(lang + ' ')[:-1]

        plt.title('Generalization Graph for ${}$'.format(lang_str))
        plt.xlabel ('Epoch Number')
        plt.ylabel ('$e_{}$ Value'.format(str(err_n)))
        # for plotting purposes
        # plt.ylim ([0, max_y])
        plt.savefig('./figures/{}_error_{}'.format(filename, str(err_n)),dpi=256)
    return

def single_investigation (lang, distrib, h_layers, h_units, window, sample_size, n_epochs, exp_num):
    acc_per_d = []
    loss_per_d = []

    generator = SampleGenerator(lang)
    for _ in range(exp_num):
        inputs, outputs, s_dst = generator.generate_sample (sample_size, window[0], window[1], distrib, False)
        e_vals, loss_vals = train (generator, distrib, h_layers, h_units, inputs, outputs, n_epochs, 1) # each experiment is unique
        acc_per_d.append (e_vals)
        loss_per_d.append(loss_vals)

    filename = '{}_{}_{}_{}_{}_{}_{}'.format(lang, 'single', distrib, h_layers, h_units, window[0], window[1])

    # Uncomment the following line if you would like to save the e_i and loss values.
    # np.savez('./results/result_{}.npz'.format(filename), errors = np.array(e_vals), losses = np.array (loss_vals))

    trials_label = ['Experiment {}'.format(elt) for elt in range (1, exp_num + 1)]
    plot_graphs (lang, 'trials', trials_label, acc_per_d, loss_per_d, window, filename)

    return acc_per_d, loss_vals

def hidden_units_investigation (lang, distrib, h_layers, h_units, window, sample_size, n_epochs, exp_num):
    acc_per_d = []
    loss_per_d = []

    generator = SampleGenerator(lang)
    for hidden_dim in h_units:
        inputs, outputs, s_dst = generator.generate_sample (sample_size, window[0], window[1], distrib, False)
        e_vals, loss_vals = train (generator, distrib, h_layers, hidden_dim, inputs, outputs, n_epochs, exp_num)
        acc_per_d.append (e_vals)
        loss_per_d.append(loss_vals)

    filename = '{}_{}_{}_{}_{}_{}'.format(lang, 'hidden', distrib, h_layers, window[0], window[1])
    hunits_label = ['{} Hidden Units'.format(val) for val in h_units]
    plot_graphs (lang, 'hiddenunits', hunits_label, acc_per_d, loss_per_d, window, filename)

    return acc_per_d, loss_vals


def window_investigation (lang, distrib, h_layers, h_units, windows, sample_size, n_epochs, exp_num):
    acc_per_d = []
    loss_per_d = []

    generator = SampleGenerator(lang)
    for window in windows:
        inputs, outputs, s_dst = generator.generate_sample (sample_size, window[0], window[1], distrib, False)
        e_vals, loss_vals = train (generator, distrib, h_layers, h_units, inputs, outputs, n_epochs, exp_num)
        acc_per_d.append (e_vals)
        loss_per_d.append(loss_vals)

    filename = '{}_{}_{}_{}_{}'.format(lang, 'window', distrib, h_layers, h_units)
    window_label = ['Window [{}, {}]'.format(elt[0], elt[1]) for elt in windows]
    plot_graphs (lang, 'window', window_label, acc_per_d, loss_per_d, [1, 30], filename) # [1, 30] is a random window. We'll ignore it later on.

    return acc_per_d, loss_vals


def distribution_investigation (lang, distribution, h_layers, h_units, window, sample_size, n_epochs, exp_num):
    acc_per_d = []
    loss_per_d = []

    generator = SampleGenerator(lang)
    for distrib in distribution:
        inputs, outputs, s_dst = generator.generate_sample (sample_size, window[0], window[1], distrib, False)
        e_vals, loss_vals = train (generator, distrib, h_layers, h_units, inputs, outputs, n_epochs, exp_num)
        acc_per_d.append (e_vals)
        loss_per_d.append(loss_vals)

    filename = '{}_{}_{}_{}_{}_{}'.format(lang, 'distrib', h_layers, h_units, window[0], window[1])
    distrib_label = [elt.capitalize() for elt in distribution]
    plot_graphs (lang, 'distrib', distrib_label, acc_per_d, loss_per_d, window, filename)

    return acc_per_d, loss_vals

def test (generator, lstm):
    first_errors = []

    with torch.no_grad():
        for num in range (1, MAX_INT):
            temp = generator.generate_sample (1, num, num)
            inp, out = temp[0][0], temp[1][0]
            input_size = len(inp)

            hidden = lstm.init_hidden()

            letter_count = 0
            for i in range (input_size):
                output, hidden = lstm (generator.lineToTensorInput(inp[i]).to(device), (hidden[0].to(device), hidden[1].to(device)))
                output = output.cpu()

                prediction = np.int_ (output.numpy()[0] >= epsilon)

                actual = np.int_ ((generator.lineToTensorOutput(out[i]).to(device)).numpy()[0])

                if np.all(np.equal(np.array(prediction), np.array(actual))): 
                    letter_count += 1

            if letter_count != input_size:
                first_errors.append(num)

                if len(first_errors) == first_k_errors:
                    return first_errors


def train (generator, distrib, h_layers, h_units, inputs, outputs, n_epochs, exp_num):
    lang = generator.get_vocab()
    vocab_size = len (lang)
    training_size = len (inputs)

    loss_arr_per_iter = []
    first_errors_per_iter = []

    for exp in range (exp_num):
        print ('Experiment Number: {}'.format(exp+1))

        # create the model
        lstm = MyLSTM(h_units, vocab_size, h_layers).to(device) 
        learning_rate = .01 ## learning rate
        
        criterion = nn.MSELoss() ## MSE Loss
        optim = torch.optim.RMSprop(lstm.parameters(), lr = learning_rate) ## RMSProp optimizer

        loss_arr = []
        first_errors = []

        for it in range(1, n_epochs + 1):
            for i in range (training_size):
                lstm.zero_grad ()
                h0, c0 = lstm.init_hidden()
                output, hidden = lstm (generator.lineToTensorInput(inputs[i]).to(device), (h0.to(device), c0.to(device)))
                target = generator.lineToTensorOutput(outputs[i]).to(device) 
                loss = criterion (output, target)
                loss.backward ()
                optim.step ()

                if i == training_size - 1: ## one full pass of the training set
                    loss_arr.append (loss.item()) ## add loss val
                    first_errors.append(test(generator, lstm)) ## add e_i vals

        loss_arr_per_iter.append (loss_arr)
        first_errors_per_iter.append (first_errors)

        # print ('Loss array: ', loss_arr)
        # print ('Max Gen: ', first_errors)

        # We can save the models as we train.
        # rnn_path = './lstm_lang{}_distrib_{}_expn_{}.pth'.format(lang, distrib, str(exp))
        # torch.save (lstm, rnn_path)

    return first_errors_per_iter, loss_arr_per_iter

def main(args):
    global first_k_errors

    investigation = args.exp_type
    lang = args.language
    distrib = args.distribution
    window = [] 
    for i in range (int(len(args.window)/2)):
        window.append([args.window[2 * i], args.window[2 * i + 1]])
    
    n_units = args.lstm_hunits
    n_layers = args.lstm_hlayers
    s_size = args.sample_size
    n_epochs = args.n_epochs
    n_trials = args.n_trials
    first_k_errors = args.disp_err_n

    if investigation == 'distribution':
        distribution_investigation (lang, distrib, n_layers, n_units[0], window[0], s_size, n_epochs, n_trials)
    elif investigation == 'window':
        window_investigation (lang, distrib[0], n_layers, n_units[0], window, s_size, n_epochs, n_trials)
    elif investigation == 'hidden_units':
        hidden_units_investigation (lang, distrib[0], n_layers, n_units, window[0], s_size, n_epochs, n_trials)
    elif investigation == 'single':
        single_investigation (lang, distrib[0], n_layers, n_units[0], window[0], s_size, n_epochs, n_trials)
    else:
        print ('Sorry, couldn\'t process your input; please try again.')

    print ('\nGoodbye...\n')
    return

if __name__ == "__main__":
    args = get_args ()
    main(args)