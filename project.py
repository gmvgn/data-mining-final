# CSCI 4502

import argparse
import pandas as pd
import matplotlib as plt
import scipy

def get_data_frame(file_name, nrows=None):
    df = pd.read_csv(file_name, nrows=nrows)
    # Data formatting
    return df

def main(file_name):
    df_working = get_data_frame(file_name, 100000)
    df_working['Date'] = pd.to_datetime(df_working['Date'])
    filtered = df_working[df_working['Primary Type'] == 'HOMICIDE']
    test = filtered.groupby([(filtered.Date.dt.year), (filtered.Date.dt.month), 'Community Area']).agg(['count'])
    print(test.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Mining Project')
    parser.add_argument('-f', type=str,
                            help="Location of file.",
                            required=True)

    args = parser.parse_args()
    main(args.f)