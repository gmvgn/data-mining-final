# CSCI 4502
# python project.py -f config.json

# Datasets:
# https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# https://data.cityofchicago.org/Public-Safety/Chicago-Police-Department-Illinois-Uniform-Crime-R/c7ck-438e

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os.path
import glob
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from scipy import stats

class DataMineChicago:
    def __init__(self, config):
        self.config = config
        self.main_dataset = config['crime_report_file']
        self.nrows_main = config['crime_report_rows']
        self.iucr_dataset = config['crime_iucr_file']
        self.export_dataset = config['export_dataset']

        self.year_range = list(range(2001, 2019))
        self.index_codes = {}
        self.index_code_names = {}


    def save_plt(self, file, legend=True):
        if (legend):
            plt.legend()
        plt.gcf().set_size_inches(10, 5.5)
        plt.savefig(file)
        plt.close()


    def get_main_data_frame(self):
        nrows = None if self.nrows_main == False else self.nrows_main
        print("\nStarting read main dataset...")
        df = pd.read_csv(self.main_dataset, nrows=nrows)
        print("Finished read")
        # Data formatting
        print("\nDate conversion")
        df['Date'] = pd.to_datetime(df['Date'])
        df.columns = [col.replace(" ", "") for col in df.columns]
        print("Finished")
        return df

    def iucr_crimes(self):
        print("\nReading IUCR data...")
        df = pd.read_csv(self.iucr_dataset)
        df['IUCR'] = df['IUCR'].apply(lambda x: "0" + str(x) if len(x) < 4 else str(x) )
        # Get list of index and non-index crimes
        for key in df['INDEX CODE'].unique():
            self.index_codes[key] = df[df['INDEX CODE'] == key]['IUCR']
        # Get text descriptions for IUCR codes
        for key in df['PRIMARY DESCRIPTION'].unique():
            self.index_code_names[key] = df[df['PRIMARY DESCRIPTION'] == key]['IUCR']
        print("Finished IUCR")

    def one_hot_encode(self, df, columns):
        uniq_col_values = {}
        for col in columns:
            uniq_col_values[col] = df[col].unique()
            for val in uniq_col_values[col]:
                key = "{} {}".format(col, val)
                df[key] = 0
                df.loc[df[col] == val, key] = 1
        return uniq_col_values

    def encode_index_crimes(self, df):        
        # Mark index crimes
        df['IndexCrime'] = 0
        df.loc[df['IUCR'].isin(self.index_codes['I']), 'IndexCrime'] = 1

    def export_time_stats(self):        
        print("\nExporting time community stats...")
        self.iucr_crimes()

        df_working = self.get_main_data_frame()
        unique_years = df_working['Date'].dt.year.unique()
        export_file_pre = self.export_dataset.replace('.csv', '')
        print("\nExport files: {}_*.csv, years: {}".format(export_file_pre, unique_years))
        
        print("\nStarting setup...")
        # Index crimes
        self.encode_index_crimes(df_working)

        # Dates
        date_col = 'YearMonthDay'
        df_working[date_col] = df_working['Date'].apply(lambda x : "{}-{}-{}".format(x.year, x.month, x.day) )

        # Get distinct column values
        columns = self.config['export_columns']
        df_encode = pd.get_dummies(df_working, columns=columns, prefix=columns, prefix_sep=" ")

        aggs = {
            'ID' : 'count',
            'IndexCrime' : 'sum'
        }

        for c in df_encode.columns:
            s = c.split(" ")
            if c[0] in columns:
                aggs[c] = 'sum'

        print(aggs)

        print("Finished setup")

        # Select columns
        selection = [date_col, 'CommunityArea'] + list(aggs.keys())

        # Do by year
        for year in unique_years:
            df_selection = df_encode[df_encode['Date'].dt.year == year][selection]
            export_file = "{}_{}.csv".format(export_file_pre, year)

            print("\nStarting group by and aggregation {}...".format(year))
            # Group by + aggregations
            group_by = df_selection.groupby([
                date_col,
                'CommunityArea'
            ]).agg(aggs)

            total_df = group_by.reset_index().sort_values(date_col, ascending=True)
            print("Finished")

            print("Starting export {}...".format(year))
            # Write to CSV
            total_df.to_csv(export_file, index=False)
            print("Finished export")
            del total_df
            del df_selection

        print("\nFinished exports")

    def plot_years(self):
        print("\nPlotting years...")
        for year in range(2001, 2019):
            datums = pd.read_csv('exports/day_type_description_{}.csv'.format(year))
            if (self.config['plot_years_overall']):
                img_file = "{}/all_crimes_{}.png".format(self.config['plot_years_overall_path'], year)
                print("Overall: ", img_file)
                
                aggs_df = datums.groupby(['YearMonthDay']).agg({ 
                    'ID' : 'sum', 
                    'IndexCrime' : 'sum' 
                }).reset_index()
                
                aggs_df['YearMonthDay'] = pd.to_datetime(aggs_df['YearMonthDay'])
                aggs_df.sort_values('YearMonthDay', ascending=True, inplace=True)
                days = list(range(len(aggs_df)))

                plt.scatter(days, aggs_df['ID'].values, label="Total Crime")
                plt.scatter(days, aggs_df['IndexCrime'].values, label="Index Crime")
                plt.title("Scatter Plot of Crimes ({})".format(year))
                plt.xlabel("Day (indexed)")
                plt.ylabel("Number of Crimes")
                self.save_plt(img_file)
            
            if (self.config['plot_years_type']):
                print("Year: ", year)
                for i, group in enumerate(self.config['plot_years_types']):
                    img_file = "{}/types_{}_{}.png".format(self.config['plot_years_type_path'], year, i)
                    print("Group: ", img_file)
                    dic = {'PrimaryType {}'.format(x) : 'sum' for x in group}
                    aggs_df = datums.groupby(['YearMonthDay']).agg(dic).reset_index()
                    aggs_df['YearMonthDay'] = pd.to_datetime(aggs_df['YearMonthDay'])
                    aggs_df.sort_values('YearMonthDay', ascending=True, inplace=True)
                    aggs_df.columns = ['Date'] + group
                    days = list(range(len(aggs_df)))

                    for series in group:
                        plt.scatter(days, aggs_df[series].values, label=series)
                    plt.title("Scatter Plot of Crime Types ({})".format(year))
                    plt.xlabel("Day (indexed)")
                    plt.ylabel("Number of Crimes")
                    self.save_plt(img_file)

        print("Done years")

    def add_ah_data(self, df):
        print("Adding affordable housing data...")
        
        datums = pd.read_csv('data/Affordable_Rental_Housing_Developments.csv')
        aggs = datums.groupby(['Community Area Number']).agg({ 
            'Community Area Name' : 'count'
        }).reset_index()
        aggs.columns = ['CommunityArea', 'Count']

        for ci in aggs['CommunityArea'].unique():
            count = aggs[aggs['CommunityArea'] == ci]['Count']
            df.loc[df['CommunityArea'] == ci, 'AH'] = count
        # df['AH'].fillna(0, inplace=True)

        print("Finished")

    def add_grocery_data(self, df):
        datums = pd.read_csv('data/Business_Licenses.csv')

        # print(datums.head())
        print(datums[datums['LICENSE ID'] == 1243125]['WARD PRECINCT'])

    def add_lead_data(self, df):
        print("Adding blood lead level data...")
        datums = pd.read_csv('data/Public_Health_Statistics_-_Screening_for_elevated_blood_lead_levels_in_children_aged_0-6_years_by_year__Chicago__1999_-_2013.csv')
        years = range(1999, 2014)

        for index, row in datums.iterrows():
            ci = row['Community Area Number']
            for year in years:
                key = 'Percent Elevated {}'.format(year)
                df.loc[(df['CommunityArea'] == ci) & (df['Year'] == year), 'Lead'] = row[key]
        print("Finished")

    def add_prenatal_data(self, df):
        print("Adding prenatal care data...")
        datums = pd.read_csv('data/Public_Health_Statistics_-_Prenatal_care_in_Chicago__by_year__1999___2009.csv')
        years = range(1999, 2010)
        keys = {
            '1ST TRIMESTER' : 'PrenatalTrimester1',
            '2ND TRIMESTER' : 'PrenatalTrimester2',
            '3RD TRIMESTER' : 'PrenatalTrimester3',
            'NO PRENATAL CARE' : 'PrenatalNone'
        }

        for index, row in datums.iterrows():
            ci = row['Community Area Number']
            trim = row['Trimester Prenatal Care Began']
            if (trim == 'NOT GIVEN'):
                continue
            col = keys[trim]
            for year in years:
                key = 'Percent {}'.format(year)
                df.loc[(df['CommunityArea'] == ci) & (df['Year'] == year), col] = row[key]
        print("Finished")

    def add_ses_data(self, df):
        print("Adding SES data...")
        datums = pd.read_csv('data/Census_Data_-_Selected_socioeconomic_indicators_in_Chicago__2008___2012.csv')

        keys = {
            'PERCENT OF HOUSING CROWDED' : 'SESCrowded',
            'PERCENT HOUSEHOLDS BELOW POVERTY' : 'SESBelowPovertyLine',
            'PERCENT AGED 16+ UNEMPLOYED' : 'SESUnemployed',
            'PERCENT AGED 25+ WITHOUT HIGH SCHOOL DIPLOMA' : 'SESNoDiploma',
            'PERCENT AGED UNDER 18 OR OVER 64' : 'SESYoungOld',
            'PER CAPITA INCOME ' : 'SESPerCapitaIncome', # This has a space in the column title
            'HARDSHIP INDEX' : 'SESIndex'
        }
        for index, row in datums.iterrows():
            ci = row['Community Area Number']
            for k, v in keys.items():
                df.loc[df['CommunityArea'] == ci, v] = row[k]
        print("Finished")

    def get_compiled_data(self):
        export_file = 'exports/combined_data.csv'
        if (os.path.isfile(export_file)):
            return pd.read_csv(export_file)

        df = pd.DataFrame(data=[], columns=['Year', 'ID', 'CommunityArea', 'IndexCrime'])
        for year in self.year_range:
            datums = pd.read_csv('exports/day_type_description_{}.csv'.format(year))
            aggs = datums.groupby(['CommunityArea']).agg({ 'ID' : 'sum', 'IndexCrime' : 'sum' })
            aggs_df = aggs.reset_index()
            aggs_df['Year'] = year
            df = pd.concat([df, aggs_df])

        # Rename columns
        new_names = {v: v for v in df.columns}
        new_names['ID'] = 'CrimeCount'
        new_names['IndexCrime'] = 'IndexCrimeCount'
        df.columns = new_names.values()

        self.distinct_communities = df['CommunityArea'].unique()

        self.add_ah_data(df)
        # self.add_grocery_data(df)
        self.add_lead_data(df)
        self.add_prenatal_data(df)
        self.add_ses_data(df)

        print("Filling missing data...")
        for col in df.columns:
            if (df[col].isnull().sum() > 0):
                df[col].fillna(df[col].mean(), inplace=True)
        print("Finished")

        print("Scaling data...")
        df['CrimeCountOrig'] = df['CrimeCount']
        df['IndexCrimeCountOrig'] = df['IndexCrimeCount']
        scale_targets = [
            'CrimeCount', 'IndexCrimeCount', 'AH', 'Lead',
            'PrenatalTrimester1', 'PrenatalTrimester2', 'PrenatalTrimester3',
            'PrenatalNone', 'SESCrowded', 'SESBelowPovertyLine', 'SESUnemployed',
            'SESNoDiploma', 'SESYoungOld', 'SESPerCapitaIncome', 'SESIndex'
        ]
        for col in scale_targets:
            df[col] = pd.to_numeric(df[col])

        scaler = MinMaxScaler()
        matrix = df[scale_targets].as_matrix()
        new_matrix = scaler.fit_transform(matrix)

        for i, col in enumerate(scale_targets):
            df[col] = new_matrix[:,i]

        print("Finished")

        # print("Encoding community area...")
        # for ci in self.distinct_communities:
        #     col = 'CA_{}'.format(int(ci))
        #     df[col] = 0
        #     df.loc[df['CommunityArea'] == ci, col] = 1
        # print("Finished")

        df.to_csv(export_file, index=False)
        return df

    def community_data(self):
        df = self.get_compiled_data()

        print("Doing correlations...")
        df.corr().to_csv('exports/correlations.csv')
        print("Finished")

        print("Starting regression...")
        x_cols = []
        exclude = [
            'CommunityArea', 'CrimeCount', 'IndexCrimeCount', 
            'Year', 'CrimeCountOrig', 'IndexCrimeCountOrig'
        ]
        for col in df.columns:
            if (col not in exclude):
                x_cols.append(col)
        x_data = df[x_cols].as_matrix()
        y_data = df['CrimeCountOrig'].values
        
        svr_lin = SVR(kernel='linear', C=1e3)
        y_lin = svr_lin.fit(x_data, y_data).predict(x_data)
        print("Linear kernel r2: {}".format(r2_score(y_data, y_lin)))

        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        y_rbf = svr_rbf.fit(x_data, y_data).predict(x_data)
        print("RBF kernel r2: {}".format(r2_score(y_data, y_rbf)))

        svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        y_poly = svr_poly.fit(x_data, y_data).predict(x_data)
        print("Poly kernel r2: {}".format(r2_score(y_data, y_poly)))

        print("Finished")


    def multi_class_test(self):
        # Given PrimaryType, LocationDescription, Arrest, Domestic, Year predict CommunityArea

        pkl_file = 'multiclass_model.pkl'
        if (os.path.isfile(pkl_file)):
            print("\nLoaded from pickle file...")
            ovr = joblib.load(pkl_file)
            print("Finished")
        else:
            print("\nStarting read main dataset...")
            test_rows = None
            df = pd.read_csv(self.main_dataset, nrows=test_rows)
            df.columns = [col.replace(" ", "") for col in df.columns]
            print("Rows: ", len(df))
            print("Finished")
            print("\nEncoding...")
            slice_cols = ['PrimaryType', 'LocationDescription', 'Arrest', 'Domestic', 'CommunityArea', 'Year']
            encode_cols = ['PrimaryType', 'LocationDescription']
            df_slice = df[slice_cols]
            result = self.one_hot_encode(df_slice, encode_cols)
            df_slice['Arrest'] = df_slice['Arrest'].apply(lambda x: int(x))
            df_slice['Domestic'] = df_slice['Domestic'].apply(lambda x: int(x))
            df_slice['Year'] = df_slice['Year'].apply(lambda x: x - 2000)
            print("Finished")

            print("\nStarting multiclass ML...")
            y = df_slice['CommunityArea'].values.astype('int')
            df_slice.drop(['PrimaryType', 'LocationDescription', 'CommunityArea'], axis=1, inplace=True)
            x = df_slice.as_matrix()

            # clf = LinearSVC(random_state=0)
            clf = RandomForestClassifier(n_estimators=100, random_state=1)
            ovr = OneVsRestClassifier(clf)
            ovr.fit(x, y)
            y_pred = ovr.predict(x)
            print("Accuracy: ", accuracy_score(y, y_pred))
            print("Finished")

            print("\nSaving model...")
            joblib.dump(clf, pkl_file) 
            print("Finished")
        print()


    def get_year_data(self, begin=2001, end=2018):
        df = pd.DataFrame(data=[])
        for year in range(begin, end + 1):
            datums = pd.read_csv('exports/day_type_description_{}.csv'.format(year), index_col=0)
            df = pd.concat([df, datums])
            print("Year: {}, Length: {}".format(year, len(df)))
        return df


    def corr_data(self):
        df = self.get_year_data()

        print("Encoding day of year...")
        df['DayOfYear'] = pd.to_datetime(df['YearMonthDay']).dt.dayofyear
        df['CommunityArea'] = df['CommunityArea'].astype(int)

        print("Encoding booleans...")
        df_encode = pd.get_dummies(df, columns=['CommunityArea', 'DayOfYear'])
        print(df_encode.columns)

        df_encode.drop(['YearMonthDay', 'ID', 'IndexCrime', 'CommunityArea', 'DayOfYear'], inplace=True)
        
        print("Doing correlations...")
        corr_df = df_encode.corr()
        print(corr_df.head())
        corr_df.to_csv('exports/correlations_year.csv')
        print("Finished")

        # Correlation between community area and crime type?
        # Correlation between day of year area and crime type?


    def regression(self):
        print("\nRegression method...")
        img_file = self.config['regression_img']
        type_files = self.config['regression_type_imgs']
        sums = { 'ID' : 'sum' }
        for k in type_files.keys():
            sums["PrimaryType {}".format(k)] = 'sum'

        df = self.get_year_data(begin=2003)
        aggs = df.groupby(['YearMonthDay']).agg(sums)
        aggs_df = aggs.reset_index()
        aggs_df['YearMonthDay'] = pd.to_datetime(aggs_df['YearMonthDay'])
        df_now = aggs_df.sort_values(by=['YearMonthDay'], ascending=True)

        cols = list(type_files.keys()) + ['ID']
        for col in cols:
            img = type_files[col] if col in type_files else img_file
            label = "All Crimes" if col == "ID" else col
            col = col if col == "ID" else "PrimaryType {}".format(col)
            print()
            print(label)
            print(img)
            print(col)

            Y = df_now[col].values
            Y_len = len(Y)
            X = list(range(Y_len))

            slope, intercept, r_value, p_value, std_err = stats.linregress(X, Y)
            y_pred = [slope * x + intercept for x in X]

            print("Linear regression: slope = {}, intercept = {}".format(slope, intercept))

            z = np.polyfit(X, Y, 10)
            p = np.poly1d(z)
            y_pred2 = [p(x) for x in X]

            plt.plot(X, Y, label=label)
            plt.plot(X, y_pred, label="Linear regression")
            plt.plot(X, y_pred2, label="Polynomial regression")
            plt.title("Regression on {} (2003-2018)".format(label))
            plt.xlabel("Day (indexed)")
            plt.ylabel("Number of Crimes")
            self.save_plt(img)


    def regression_type(self):
        print("\nRegression on crime type...")
        use_types = ['NARCOTICS', 'ROBBERY', 'THEFT', 'BURGLARY', 'STALKING', 'ASSAULT', 'INTIMIDATION']
        predict_type = 'CRIM SEXUAL ASSAULT'

        df = self.get_year_data()
        df['DayOfYear'] = pd.to_datetime(df['YearMonthDay']).dt.dayofyear

        slice_cols = ['PrimaryType {}'.format(c) for c in use_types] + ['CommunityArea', 'DayOfYear']
        x_data = df[slice_cols]
        y_data = df['PrimaryType {}'.format(predict_type)]

        print("\nRegression on {} rows...".format(len(y_data)))

        print("\nBeginning neural net...")
        nn = MLPRegressor(hidden_layer_sizes=(200, 100))
        y_nn = nn.fit(x_data, y_data).predict(x_data)
        print("Neural network r2: {}".format(r2_score(y_data, y_nn)))

        print("\nBeginning RBF SVR...")
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        y_rbf = svr_rbf.fit(x_data, y_data).predict(x_data)
        print("RBF kernel r2: {}".format(r2_score(y_data, y_rbf)))

        print("\nBeginning polynomial SVR...")
        svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        y_poly = svr_poly.fit(x_data, y_data).predict(x_data)
        print("Poly kernel r2: {}".format(r2_score(y_data, y_poly)))
        
        # svr_lin = SVR(kernel='linear', C=1e3)
        # y_lin = svr_lin.fit(x_data, y_data).predict(x_data)
        # print("Linear kernel r2: {}".format(r2_score(y_data, y_lin)))


    def find_outliers(self):
        start, end = self.config['outlier_years']
        types = self.config['outlier_types']
        devs = self.config['outlier_deviations']
        do_comm = self.config['outlier_community']

        cross_years_df = pd.DataFrame(data=[], columns=['Date', 'Count', 'Type'])

        print("Using {} deviations for years {}-{}".format(devs, start, end))

        for year in range(start, end + 1):
            print("\n******************************************************************************************************************\nYear: {}\n".format(year))
            datums = pd.read_csv('exports/day_type_description_{}.csv'.format(year))
            dic = {'PrimaryType {}'.format(x) : 'sum' for x in types}
            aggs_df = datums.groupby(['YearMonthDay']).agg(dic).reset_index()
            aggs_df['YearMonthDay'] = pd.to_datetime(aggs_df['YearMonthDay'])
            aggs_df.sort_values('YearMonthDay', ascending=True, inplace=True)
            
            # Crime type outliers
            for t in types:
                lbl = 'PrimaryType {}'.format(t)
                col = 'Outlier {}'.format(t)
                mean = aggs_df[lbl].mean()
                std = aggs_df[lbl].std()
                lower = mean - devs * std
                upper = mean + devs * std
                aggs_df.loc[:, col] = 0
                aggs_df.loc[(aggs_df[lbl] > upper), col] = 1 #  | (aggs_df[lbl] < lower)
                
                out_slice = aggs_df[aggs_df[col] == 1]
                if (len(out_slice) == 0):
                    continue

                print("------------------------------------------------------")
                print("{}: [{:.2f}, {:.2f} (mean), {:.2f}] (std = {:.2f})\n".format(t, lower, mean, upper, std))
                for i, row in out_slice.iterrows():
                    print("{}: {}".format(row['YearMonthDay'].strftime('%m-%d'), row[lbl]))

                    cross_years_df = cross_years_df.append({
                        'Date': row['YearMonthDay'],
                        'Count': row[lbl],
                        'Type' : t
                    }, ignore_index=True)
                print()

            # Index crime outliers
            if do_comm:
                df2 = datums.copy()
                df2['Month'] = pd.to_datetime(df2['YearMonthDay']).dt.month
                aggs_df2 = df2.groupby(['Month', 'CommunityArea']).agg({
                    'ID' : 'sum',
                    'IndexCrime' : 'sum'
                }).reset_index()
                aggs_df2.sort_values('Month', ascending=True, inplace=True)
                print("------------------------------------------------------")
                print("Index Crime Outliers by Month:\n")
                for month in aggs_df2.Month.unique():
                    month_slice = aggs_df2[aggs_df2['Month'] == month]
                    idx_series = month_slice['IndexCrime']
                    mean = idx_series.mean()
                    upper = mean + devs * idx_series.std()
                    out_slice = month_slice[month_slice['IndexCrime'] >= upper]
                    if len(out_slice) > 0:
                        for i, row in out_slice.iterrows():
                            print("Month {} area {}: {}".format(month, int(row['CommunityArea']), row['IndexCrime']))
                    print()


        print("*********************************************************************************************************************")
        print("Sum of Outliers by Type ({}-{})".format(start, end))
        print("*********************************************************************************************************************")

        cross_years_df.loc[:,'MonthDate'] = pd.to_datetime(cross_years_df['Date']).dt.strftime('%m-%d')
        cross_years_df.loc[:,'DayOfYear'] = pd.to_datetime(cross_years_df['Date']).dt.dayofyear
       
        # List
        for t in types:
            out_slice = cross_years_df[cross_years_df['Type'] == t]
            aggs_df = out_slice.groupby(['MonthDate']).agg({
                'Count' : 'sum'
            }).reset_index()
            aggs_df.sort_values('MonthDate', ascending=True, inplace=True)
            print()
            print("Crime type: `{}`".format(t))
            for i, row in aggs_df.iterrows():
                print("{}: {}".format(row['MonthDate'], row['Count']))
        
        # Plot
        days = list(range(1, 366))
        for t in types:
            out_slice = cross_years_df[cross_years_df['Type'] == t]
            aggs_df = out_slice.groupby(['DayOfYear']).agg({
                'Count' : 'sum'
            }).reset_index()
            Y = [0] * 365
            for i, row in aggs_df.iterrows():
                Y[row['DayOfYear'] - 1] = row['Count']
            img_file = "{}/outlier_{}.png".format(self.config['outlier_imgs'], t)
            plt.scatter(days, Y)
            plt.title("Outliers for `{}` Crimes".format(t))
            plt.xlabel("Day (indexed)")
            plt.ylabel("Number of Crimes")
            self.save_plt(img_file, legend=False)


    def start(self):
        print()
        if (self.config['jobs']['breakout']):
            self.export_time_stats()

        if (self.config['jobs']['community']):
            self.community_data()

        if (self.config['jobs']['multiclass']):
            self.multi_class_test()

        if (self.config['jobs']['correlations']):
            self.corr_data()

        if (self.config['jobs']['plot_years']):
            self.plot_years()

        if (self.config['jobs']['regression']):
            self.regression()

        if (self.config['jobs']['regression_type']):
            self.regression_type()

        if (self.config['jobs']['outliers']):
            self.find_outliers()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Mining Project')
    parser.add_argument('-f', type=str,
                            help="Location of file.",
                            required=True)

    args = parser.parse_args()

    config = json.load(open(args.f))
    
    inst = DataMineChicago(config)
    inst.start()

