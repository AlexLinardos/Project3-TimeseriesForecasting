# forecast_ui.py

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

        # Execute the parse_args() method
        args = my_parser.parse_args()

        print(vars(args))

        if not os.path.isfile(args.d):
            print('The dataset_path specified does not exist')
            sys.exit()
        
        self.dataset_path = args.d
        self.ts_number = args.n




