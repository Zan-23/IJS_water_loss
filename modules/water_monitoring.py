import pandas as pandas
import numpy as np

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor


class WaterMonitoringInstance:
    """"
    Class for loading and analysis data from Alicante and Braila. The data is
    related to plumbing/water loses.

    file_name - name of the file that the data was read from
    """
    file_name = ""
    nan_data = None
    data = None

    def read_data(self, file_name, separator):
        """
        Method for reading data.

        :param file_name: File fromn which to read.
        :param separator: separator can take next values: "\t", ",", ";", "json". This list can be expanded...
        """
        self.file_name = file_name

        if separator == "\t":
            self.data = pandas.read_csv(file_name, sep="\t")
        elif separator == ",":
            self.data = pandas.read_csv(file_name, sep=",")
        elif separator == ";":
            self.data = pandas.read_csv(file_name, sep=";")
            # , dtype="string", low_memory=False
        elif separator == "json":
            self.data = pandas.read_json(file_name)

        else:
            raise Exception("Unknown type of separator, please remodel read_data function !")

    def transform_data(self, city, del_cols=False):
        """
        Alicante - Removes all rows where the first value is NaN, and transforms other data into a logical structure
        |->

        :param city: "alicante" or "braila", set it to to the place where the data originated
        :param del_cols: This is a flag which deletes unnecessary rows -> saving RAM
        """
        if city == "braila":
            self.nan_data = self.data[(self.data.tot1 == 0) & (self.data.analog2 == 0)]
            self.data = self.data[(self.data.tot1 != 0) & (self.data.analog2 != 0)]

            self.data.iloc[:, 0] = pandas.to_datetime(self.data.iloc[:, 0], format="%Y-%m-%d %H:%M:%S")

        elif city == "alicante":
            # saving rows with NaN in the first column
            self.nan_data = self.data[self.data.iloc[:, 0].isnull()]
            # removing all unknown dates from dataframe and converting first column to datetime
            self.data = self.data[self.data.iloc[:, 0].notna()]

            self.data.iloc[:, 0] = pandas.to_datetime(self.data.iloc[:, 0], format="%d/%m/%Y %H:%M:%S")
            first_col = self.data.iloc[:, 1].str.replace(',', '.').astype(float)
            date_1_col = self.data.iloc[:, 0]

            if not del_cols:
                date_2_col = self.data.iloc[:, 2]
                second_col = self.data.iloc[:, 3].str.replace(',', '.').astype(float)

                new_data_f = pandas.concat([date_1_col, first_col, date_2_col, second_col], axis=1)
                new_data_f = new_data_f.fillna(-150)
                new_data_f.rename(columns={new_data_f.columns[1]: "Data-1(Sensor1)",
                                           new_data_f.columns[3]: "Data-1(Sensor2)"}, inplace=True)
                self.data = new_data_f
            else:
                # If flag is set, only the first 3 columns are returned as specified on drive
                new_data_f = pandas.concat([date_1_col, first_col], axis=1)
                new_data_f = new_data_f.fillna(-150)
                new_data_f.rename(columns={new_data_f.columns[1]: "Data-1(Sensor1)"}, inplace=True)
                self.data = new_data_f
        else:
            raise Exception("Unknown city, modify transform_data to process other cities")

    def plot_two_columns(self, name_x, name_y, x_time=False):
        """
        Method for plotting two columns.
        :param name_x: Column name for the values that you want to plot on x
        :param name_y: Column name for the values that you want to plot on y
        :param x_time: Boolean attribute which if True, plots numbers in range instead of x values
        """
        x_data = None
        y_data = self.data[name_y].to_numpy()

        if x_time:
            x_data = [i for i in range(len(y_data))]
        else:
            x_data = self.data[name_x].to_numpy()

        fig = plt.figure(figsize=(18, 8), dpi=100, facecolor='w')
        plt.plot(x_data, y_data, color="red")
        # plt.xticks([])
        plt.ylabel(name_y)
        plt.xlabel(name_x)
        plt.title(name_x + " - " + name_y)
        plt.show()

    def interactive_two_columns_plot(self, name_x, name_y):
        """
        Method for an interactive visualization. Be carefull with it because it can take up a lot of memory.

        :param name_x: Column name for the values that you want to plot on x
        :param name_y: Column name for the values that you want to plot on y
        """
        fig = px.line(self.data, x=name_x, y=name_y, height=600)
        fig.show()

    def make_time_features(self, column_to_process):
        """
        This method divides specified column into multiple columns which can be used to feature engineer predictions
        :param column_to_process: Name of the column, which to split
        :param format: Format in which the date is specified in the column, format="%d/%m/%Y %H:%M:%S"
        """
        self.data['Minute'] = [i.minute for i in self.data[column_to_process]]
        self.data['Hour'] = [i.hour for i in self.data[column_to_process]]
        self.data['Day'] = [i.day for i in self.data[column_to_process]]
        self.data['Month'] = [i.month for i in self.data[column_to_process]]
        self.data['Year'] = [i.year for i in self.data[column_to_process]]

    def calculate_correlation(self, col_name_1, col_name_2=""):
        """
        This method can calculate correlation between all the attributes and one column.
        Or just between two columns.

        :param col_name_1: If only this one is given calculate correlation from this to all.
        :param col_name_2: If none is given calculate for all, if given calculate just for the two
        :return: Returns an array of tuples. First element in a tuple is the column name and the second is Pearson coef.
        """
        coef_arr = []

        if col_name_1 is not None and col_name_2 == "":
            column_names = list(self.data.columns)
            column_names.remove(col_name_1)
            max_correlation = [0, "none"]

            for col_name in column_names:
                try:
                    pearson_coef, p_value = pearsonr(self.data[col_name_1], self.data[col_name])
                    coef_arr.append((col_name, pearson_coef))

                    if abs(pearson_coef) > abs(max_correlation[0]):
                        max_correlation[0] = pearson_coef
                        max_correlation[1] = col_name
                except Exception as e:
                    print("Attribute " + col_name + " can't be correlated. " + e.__str__())

            print("\nColumn with the most correlation is: " + max_correlation[1] +
                  ". Pearson coef: {:.4f}".format(max_correlation[0]))

        elif col_name_1 is not None and col_name_2 != "":
            pearson_coef, p_value = pearsonr(self.data[col_name_1], self.data[col_name_2])
            spear_coef, p_value_s = spearmanr(self.data[col_name_1], self.data[col_name_2])
            kendall_tau, p_value_s = kendalltau(self.data[col_name_1], self.data[col_name_2])
            coef_arr.append((col_name_2, pearson_coef))

            print("Columns '" + col_name_1 + "' and '" + col_name_2 +
                  "' are correlated with a Pearson coef:\n{:.4f}".format(pearson_coef))
            print("Columns '" + col_name_1 + "' and '" + col_name_2 +
                  "' are correlated with a Spearman coef:\n{:.4f}".format(spear_coef))
            print("Columns '" + col_name_1 + "' and '" + col_name_2 +
                  "' are correlated with a Kendall tau coef:\n{:.4f}".format(kendall_tau))

        else:
            raise Exception("Wrong parameters given !")

        # sorts coefficient array and returns it
        return sorted(coef_arr, key=lambda tup: tup[1])

    def take_data_in_range(self, start_date, end_date, col_name, date_format="%Y-%m-%d %H:%M:%S"):
        """
        This method returns a subset of the data frame between two dates.

        :param start_date: Starting date, data will be filtered from this day forward
        :param end_date: End data, date will be filtered to this day
        :param col_name: Column in which the timestamp are located
        :param date_format: Format of the date, a default format is given
        :return: Returns subset of the data between ranges
        """
        start_date = pandas.to_datetime(start_date, format=date_format)
        end_date = pandas.to_datetime(end_date, format=date_format)

        return self.data.loc[(self.data[col_name] >= start_date) & (self.data[col_name] <= end_date)]

    def get_nan_rows(self):
        return self.nan_data

    def get_data_frame(self):
        """
        Returns instance dataframe on which all operations are performed
        """
        return self.data

    def set_data_frame(self, data, file_name="Manually set"):
        """
        Sets the self.data to the new dataframe
        """
        self.data = data
        self.file_name = file_name

    def set_df_index(self, column_name):
        """ TODO implement this method when all is finished and we have decided where the timestamp will be
        or in a column or in index """
        self.data = self.data.set_index(column_name)


class Analyzer:
    dataframe = None

    def __init__(self, dataframe):
        self.dataframe = dataframe

    def linear_regression(self, matrix_X, vector_Y):
        # TODO split into learn and test, automatic trying of linear, ridge and lasso
        # TODO Return the best model and its predictions
        pass
        return None

    def random_forest(self, matrix_cols, vector_col, time_col_name=""):
        """
        This method builds a random forest model based on the sklearn library.

        :param matrix_cols: Columns which you want to include in matrix X
        :param vector_col: Column which you want to predict
        :param time_col_name: Column in which the time series is stored(used for plotting predictions)
        :return: Returns the actual y values, predicted values of y and
        the timeframe for which the values were generated
        """
        matrix_x, vector_y, time_col = self.generate_matrix_and_vector(matrix_cols, vector_col, time_col_name)
        if len(matrix_x) != len(vector_y):
            raise Exception("Matrix X and vector Y must be of the same length !!")

        # Splitting on learn and test
        len_of_test = round(len(matrix_x) * 0.7)
        learn_x = matrix_x[:len_of_test]
        learn_y = vector_y[:len_of_test]

        test_x = matrix_x[len_of_test:]
        test_y = vector_y[len_of_test:]
        if time_col is not None:
            time_col = time_col[len_of_test:]

        # RANDOM FOREST - 50 trees
        regressor_rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        regressor_rf.fit(learn_x, learn_y.ravel())
        y_predicted = regressor_rf.predict(test_x)

        r2_random_f = r2_score(test_y, y_predicted)
        print("Random forest regression R2: ", r2_random_f)
        diff = mean_squared_error(test_y, y_predicted)
        print("MSE result:", diff)

        return test_y, y_predicted, time_col

    def hierarhical_clustering(self):
        # TODO grouping based on all attributes, try out different sets of data
        # TODO returns different classes depending on row number
        pass
        return None

    def generate_matrix_and_vector(self, matrix_cols, vector_col, time_col):
        """
        This method generate matrix X and vector Y which are used in most of the methods of this for building models.

        :param matrix_cols: Columns which you want to include in matrix X
        :param vector_col: Column which you want to predict
        :param time_col:  Column in which the time series is stored(used for plotting predictions)
        :return: Returns three arrays, matrix X and vector Y and timestamp array
        """
        matrix_x = self.dataframe[matrix_cols].to_numpy()
        vector_col = self.dataframe[vector_col].to_numpy()
        timestamp_col = None

        if time_col != "":
            timestamp_col = self.dataframe[time_col].to_numpy()

        return matrix_x, vector_col, timestamp_col


"""
        elif city == "alicante":
            # saving rows with NaN in the first column
            self.nan_data = self.data[self.data.iloc[:, 0].isnull()]

            # removing all unknown dates from dataframe and converting first column to datetime
            self.data = self.data[self.data.iloc[:, 0].notna()]
            self.data.iloc[:, 0] = pandas.to_datetime(self.data.iloc[:, 0], format="%d/%m/%Y %H:%M:%S")

            first_col_splitted = self.data.iloc[:, 1].str.split(',', expand=True)
            first_col_splitted.columns = ['Data-1(Sensor1)', 'Data-2(Sensor1)']
            date_1_col = self.data.iloc[:, 0]

            if not del_cols:
                date_2_col = self.data.iloc[:, 2]
                second_col_splitted = self.data.iloc[:, 3].str.split(',', expand=True)
                second_col_splitted.columns = ['Data-1(Sensor2)', 'Data-2(Sensor2)']

                new_data_f = pandas.concat([date_1_col, first_col_splitted, date_2_col, second_col_splitted], axis=1)
                new_data_f = new_data_f.fillna(-150)
                new_data_f = new_data_f.astype({'Data-1(Sensor1)': 'int64', 'Data-2(Sensor1)': 'int64',
                                                'Data-1(Sensor2)': 'int64', 'Data-2(Sensor2)': 'int64'})
                self.data = new_data_f
            else:
                # If flag is set, only the first 3 columns are returned as specified on drive
                new_data_f = pandas.concat([date_1_col, first_col_splitted], axis=1)
                new_data_f = new_data_f.fillna(-1)
                new_data_f = new_data_f.astype({'Data-1(Sensor1)': 'int64', 'Data-2(Sensor1)': 'int64'})
                self.data = new_data_f

"""

