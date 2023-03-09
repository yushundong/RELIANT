import argparse
import yaml
from pathlib import Path
import numpy as np
import os
import torch
import pandas as pd
import scipy.sparse as sp
from scipy.spatial import distance_matrix
from data.make_dataset import get_dataset_and_split_planetoid, get_dataset, get_train_val_test_split
from data.preprocess import binarize_labels
from keras.utils import to_categorical


def get_experiment_config(config_path):
    with open(config_path, 'r') as conf:
        return yaml.load(conf, Loader=yaml.FullLoader)


def generate_data_path(dataset, dataset_source):
    if dataset_source == 'planetoid':
        return '../../../../data/planetoid'
    elif dataset_source == 'npz':
        return '../../../../data/npz/' + dataset + '.npz'
    else:
        print(dataset_source)
        raise ValueError(f'The "dataset_source" must be set to "planetoid" or "npz"')


def load_dataset_and_split(labelrate, dataset):
    # _config = get_experiment_config(config_file)
    _config = {
        'dataset_source': 'npz',
        'seed': 0,
        'train_config': {
            'split': {
                'train_examples_per_class': labelrate,  # 20
                'val_examples_per_class': 30
            },
            'standardize_graph': True
        }
    }
    # print('_config', _config)
    _config['data_path'] = generate_data_path(dataset, _config['dataset_source'])
    if _config['dataset_source'] == 'planetoid':
        return get_dataset_and_split_planetoid(dataset, _config['data_path'])
    else:
        adj, features, labels, sens = get_dataset(dataset, _config['data_path'],
                                            _config['train_config']['standardize_graph'],
                                            _config['train_config']['split']['train_examples_per_class'],
                                            _config['train_config']['split']['val_examples_per_class'])
        random_state = np.random.RandomState(_config['seed'])
        idx_train, idx_val, idx_test = get_train_val_test_split(dataset, labels, random_state,
                                                                **_config['train_config']['split'])

        return adj, features, labels, idx_train, idx_val, idx_test, sens


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='load data', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        default='dataset.conf.yaml',
                        help='Path to the YAML configuration file for the experiment.')
    args = parser.parse_args()
    adj, features, labels, idx_train, idx_val, idx_test = load_dataset_and_split(5,'bail')
    print(features)
    print(labels)
