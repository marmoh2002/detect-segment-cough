#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Python wraper to detect cough """

import os
import sys
sys.path.append(os.path.abspath('.../src'))
sys.path.append(os.path.abspath('.../src/cough_detection'))
sys.path.append(os.path.abspath('./src/'))
sys.path.append(os.path.abspath('./models/'))
# sys.path.append('.../models/')
# sys.path.append('../models/')
sys.path.append('./models/')
from src.feature_class import features
# from models import cough_classifier, cough_classification_scaler
from src.DSP import classify_cough
from scipy.io import wavfile
import pickle
import argparse
import xgboost as xgb

def main(input_file):
    # print("Current working directory:", os.getcwd())
    # print("Script location:", os.path.abspath(__file__))
    # print("Script directory:", os.path.dirname(os.path.abspath(__file__)))
    """
    Detect cough in a given audio file
    Inputs:
        input_file: (str) path to audio file
    Outputs:
        result: (float) probability that a given file is a cough
    """
    # data_folder = './sample_recordings'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # model = pickle.load(open(os.path.join(script_dir, 'models', 'cough_classifier'),
    #     'rb'))
    # scaler = pickle.load(open(os.path.join(script_dir, '/models',
        'cough_classification_scaler'), 'rb'))
    model_path = os.path.join(script_dir, 'models', 'cough_classifier.json')  # or .ubj or .bin
    model = xgb.Booster()
    model.load_model(model_path)
    # Load the scaler
    scaler_path = os.path.join(script_dir, 'models', 'cough_classification_scaler.pkl')
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
        
    fs, x = wavfile.read(input_file)
    prob = classify_cough(x, fs, model, scaler)
    print(f"{input_file} has probability of cough: {prob}")
    return prob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        help='Path to input audio file',
                        required=True)
    args = parser.parse_args()
    main(args.input)

