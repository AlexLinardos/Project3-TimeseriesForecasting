# detect_ui.py

# Import the argparse library
import argparse

import os
import sys

class Ui:    

    def __init__(self, *args):
        # Create the parser
        my_parser = argparse.ArgumentParser()
        # Add the arguments
        my_parser.add_argument('-d', metavar='<dataset>', type=str, action='store', help='the path to the dataset', required=True)
        my_parser.add_argument('-n', metavar='<number of time series selected>', type=int, action='store', help='number of time series selected', required=True)
        my_parser.add_argument('-mae', metavar='<error value as double>', type=float, action='store', help='error value as double', required=True)
        my_parser.add_argument('--retrain', action='store_true', help='if given, model will be re-train (ATTENTION! This will take a lot of time)', required=False)

        # Execute the parse_args() method
        args = my_parser.parse_args()

        print(vars(args))

        if not os.path.isfile(args.d):
            print('The dataset_path specified does not exist')
            sys.exit()
        
        self.dataset_path = args.d
        self.ts_number = args.n
        self.mae = args.mae
        self.retrain = args.retrain
