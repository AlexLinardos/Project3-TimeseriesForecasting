# reduce_ui.py

# Import the argparse library
import argparse

import os
import sys

# –d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>

# Create the parser
my_parser = argparse.ArgumentParser()

# Add the arguments
my_parser.add_argument('-d', metavar='<dataset>', type=str, action='store', help='the path to the dataset', required=True)
my_parser.add_argument('-q', metavar='<queryset>', type=str, action='store', help='the path to the queryset', required=True)
my_parser.add_argument('-od', metavar='<output_dataset_file>', type=str, action='store', help='the output_dataset_file', required=True)
my_parser.add_argument('-oq', metavar='<output_query_file>', type=str, action='store', help='the output_query_file', required=True)

# Execute the parse_args() method
args = my_parser.parse_args()

print(vars(args))

dataset_path = args.d
queryset_path = args.q

if not os.path.isfile(dataset_path):
    print('The dataset_path specified does not exist')
    sys.exit()

if not os.path.isfile(queryset_path):
    print('The dataset_path specified does not exist')
    sys.exit()
