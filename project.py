# CSCI 4502

import argparse
import pandas as pd
import matplotlib as plt
import scipy

def get_data_frame(file_name):
    df = pd.read_csv(file_name)
    # Data formatting
    return df

def main(file_name):
    df = get_data_frame(file_name)
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Mining Project')
    parser.add_argument('-f', type=str,
                            help="Location of file.",
                            required=True)

    args = parser.parse_args()
    main(args.f)