import argparse
import copy
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.append('../')
from get_raw_data import getUTKdata, getMORPHdata, getAPPAdata, getMegaasianData
from utils.data_utils import get_min_max_sample, update


def flip_image(image_path):
    im = Image.open(image_path)
    im_flipped = im.transpose(method=Image.FLIP_LEFT_RIGHT)
    save_path = image_path + '_flip.jpg'
    im_flipped.save(save_path, 'JPEG')
    return save_path
    #jpg_train.tsv files are created in data/original after running python3 data_preprocess.py. Why?

def get_balanced_data(data_folder, train_save_path='../data/original/train_new.tsv', test_save_path='../data/original/test_new.tsv'):
    # dataset_names = ['UTKdata', 'Megaasian', 'APPA', 'MORPH']
    dataset_names = ['UTKdata']
    races = ['caucasian', 'afroamerican', 'asian']
    # Get all four datasets
    #tx: all_datasets is a dict, contains the path, age, gender and race information from all datasets, categorized by the name of each dataset
    all_datasets = {
        'UTKdata': getUTKdata(data_folder),
        # 'Megaasian': getMegaasianData(data_folder),
        # 'APPA': getAPPAdata(data_folder),
        # 'MORPH': getMORPHdata(data_folder)
    }
    # Store the number of samples of each age of each race of each dataset
    # Organized as age->race->dataset
    # For sorting the datasets for choosing data
    #tx: UTKFace will give us something like: { caucasian: {UTKFace: 0}, afroamerican: {UTKFace: 0}, asian: {UTKFace: 0}}
    #if we have other datasets than just UTKFace, we will have: num_samples_tmp = {
    # caucasian: {UTKFace: 0 WIKI: 0 IMDB: 0}, afroamerican: {UTKFace: 0 WIKI: 0 IMDB: 0}, asian: {UTKFace: 0  WIKI: 0 IMDB: 0}}
    num_samples_tmp = {
        race: {i: 0 for i in dataset_names} for race in races
    }
    #tx:copy num_samples_tmp for 100 times, 0~100 represents the ages we care about
    dataset_samples = {
        i: copy.deepcopy(num_samples_tmp) for i in range(0, 101)
    }
    # Store the samples, organized by dataset->race->age
    # For sampling the balanced data
    #tx: defaultdict will not raise key error. If a race doesn't exist, all_sample_tmp will return an empty list.
    #tx: all_samples_tmp = {caucasian: [], afroamerican: [], asian: []}
    all_samples_tmp = {
        race: defaultdict(list) for race in races
    }
    #tx: every dataset will have a dictionary with races as its keys
    all_samples = {
        dataset: copy.deepcopy(all_samples_tmp) for dataset in dataset_names
    }
    # Number of samples for each ethnicity in each age
    # For getting the max, min and threshold
    #tx: num_sample will return something like: num_sample = {caucasian : {0: 0, 1: 0, 2: 0, 3: 0 ...101: 0} afroamerican: caucasian : {0: 0, 1: 0, 2: 0, 3: 0 ...101: 0} asisan: {0: 0, 1: 0, 2: 0, 3: 0 ...101: 0}}
    #tx: num_sample is used to show the number of people at different age (from 0 to 100) for each race
    num_sample = {
        race: {i: 0 for i in range(0, 101)} for race in races
    }
    # Store and organize the original data from the raw data
    for dataset in all_datasets:
        for samples in tqdm(all_datasets[dataset]):
            if 0 <= samples['age'] <= 100 and samples['race'] in ['caucasian', 'afroamerican', 'asian']:
                #tx: 'image_path':folder+'/UTKFace/'+image
                #tx: ??? are we going to store every picture that satisfies if conditions to a new path?
                #tx: replace won't rise errors if it doesn't find "OriDatasets".
                file_path = samples['image_path'].replace('OriDatasets', 'AliDatasets_new')
                #tx: check if the file_path really exists in the operating system, maybe it can select effective pictures?
                if not os.path.exists(file_path):
                    continue
                #tx: every dataset will have a dictionary with races as its keys, read each sample and complement the information with sample path
                all_samples[dataset][samples['race']][samples['age']].append(
                    [file_path, samples['race'], samples['age']])
                #tx: for each age and race, we update the count of people from different datasets
                dataset_samples[samples['age']][samples['race']][dataset] += 1
                #tx: update the number of people for each race and age combination
                num_sample[samples['race']][samples['age']] += 1
                #tx:flip this image and save it to a new path
                try:
                    save_path = flip_image(file_path)
                except Exception as e:
                    print(file_path, e)
                    continue
                #tx:update the following dictionary and counters
                all_samples[dataset][samples['race']][samples['age']].append(
                    [save_path, samples['race'], samples['age']])
                dataset_samples[samples['age']][samples['race']][dataset] += 1
                num_sample[samples['race']][samples['age']] += 1
    # Sort the number of samples of each race
    for key in num_sample:
        #tx:samples is age counter (age as the key) for each race, the key below is a race
        samples = copy.deepcopy(num_sample[key])
        #tx: sort age by count for every race
        num_sample[key] = dict(sorted(samples.items(), key=lambda samples: samples[1]))
    #tx:???find a region of age that all races are populated
    min_sample, max_sample = get_min_max_sample(num_sample)

    # Store the train data and test data
    balanced_train_data = []
    balanced_test_data = []
    #tx: make an age counter for every race
    train_data_num = {
        race: {i: 0 for i in range(0, 101)} for race in races
    }

    for age in range(1, 101):

        # Get threshold
        # xi: ???what's the purpose of having a threshold
        threshold = np.inf
        for race in num_sample:
            #tx: this threshold is the smallest number of count for all ages for a given race
            threshold = min(threshold, num_sample[race][age])
        #xi: why do we define threshold this way?
        threshold = int(min(max_sample, max(min_sample, threshold)))

        # Get select_size
        #xi: ds_num shows how many datasets are being used
        ds_num = len(all_datasets)
        #xi: This select size  will ensure that for a race and age combination, data from every dataset will be choosen
        select_size = math.ceil(threshold * 1.0 / ds_num)

        for race in num_sample:
            # Copy threshold and threshold for update for this race
            race_threshold = threshold
            race_select_size = select_size
            ds_num = len(all_datasets)

            # Sort the dataset according to the number of samples about each race at this age 
            race_num_sample = copy.deepcopy(dataset_samples[age][race])
            dataset_samples[age][race] = dict(
                sorted(race_num_sample.items(), key=lambda race_num_sample: race_num_sample[1]))

            # Begin sampling data
            for dataset in dataset_samples[age][race]:

                # get the number of the samples of this dataset in this age and this race
                num = dataset_samples[age][race][dataset]
                if num > race_select_size:
                    train_size = math.ceil(num * 0.8)
                    # Sampling train data
                    for index in range(train_size):
                        balanced_train_data.append(all_samples[dataset][race][age][index])
                        train_data_num[race][age] += 1
                    # Sampling test data
                    for index in range(train_size, num):
                        balanced_test_data.append(all_samples[dataset][race][age][index])
                    race_select_size, race_threshold, ds_num = update(race_select_size, race_threshold, num, ds_num)
                else:

                    # random sample from dataset
                    indices = np.random.choice(len(all_samples[dataset][race][age]), race_select_size, replace=False)
                    train_size = math.floor(race_select_size * 0.8)
                    ds_num -= 1
                    for index in range(train_size):
                        balanced_train_data.append(all_samples[dataset][race][age][indices[index]])
                        train_data_num[race][age] += 1
                    for index in range(train_size, len(indices)):
                        balanced_test_data.append(all_samples[dataset][race][age][indices[index]])

    balanced_test_data = pd.DataFrame(balanced_test_data)
    balanced_train_data = pd.DataFrame(balanced_train_data)
    balanced_test_data.to_csv(test_save_path, header=None, index=None, sep='\t')
    balanced_train_data.to_csv(train_save_path, header=None, index=None, sep='\t')
    return


def get_separate_data(file_path):
    f = open(file_path, 'r')
    test_train = file_path.split('/')[-1].split('_')[0]
    folder = '/'.join(file_path.split('/')[:-1])
    lines = f.readlines()
    goals = defaultdict(list)
    for line in lines:
        goal = line.strip().split('\t')[0].split('/')[3]
        goals[goal].append(line)

    for keys in goals:
        f = open('{}/{}_{}.tsv'.format(folder, keys, test_train), 'w')
        for i in goals[keys]:
            f.write(i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dir', type=str, default='../data/')
    parser.add_argument('-train_save_path', type=str, default='../data/original/train_new.tsv')
    parser.add_argument('-test_save_path', type=str, default='../data/original/test_new.tsv')
    args = parser.parse_args()

    data_folder = args.dir
    train_save_path = args.train_save_path
    test_save_path = args.test_save_path

    get_balanced_data(data_folder, train_save_path, test_save_path)

    get_separate_data(train_save_path)
    get_separate_data(test_save_path)
