import pandas as pandas
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
# TODO implement saving of NaN data as it can be useful
# TODO implement random forest method for the class
# TODO implement method for calculating of correlation between attributes and then printing it


class WaterMonitoringInstance:
    """"
    Class for loading and analysis data from Alicante and Braila. The data is
    related to plumbing/water loses.

    file_name - name of the file that the data was read from
    """
    file_name = ""
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
            self.data = self.data[(self.data.tot1 != 0) & (self.data.analog2 != 0)]
        elif city == "alicante":
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
                new_data_f = new_data_f.fillna(-1)
                new_data_f = new_data_f.astype({'Data-1(Sensor1)': 'int64', 'Data-2(Sensor1)': 'int64',
                                                'Data-1(Sensor2)': 'int64', 'Data-2(Sensor2)': 'int64'})
                self.data = new_data_f
            else:
                # If flag is set, only the first 3 columns are returned as specified on drive
                new_data_f = pandas.concat([date_1_col, first_col_splitted], axis=1)
                new_data_f = new_data_f.fillna(-1)
                new_data_f = new_data_f.astype({'Data-1(Sensor1)': 'int64', 'Data-2(Sensor1)': 'int64'})
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

    def make_time_features(self, column_to_process, format="%d/%m/%Y %H:%M:%S"):
        """
        This method divide specified column into multiple columns which can be used to feature engineer predictions
        :param column_to_process: Name of the column, which to split
        :param format: Format in which the date is specified in the column
        :return:
        """
        # TODO implement from existing code

        return None

    def calculate_correlation(self, col_name_1="", col_name_2=""):
        """
        This method calculates correlation between all the atributtes in the dataframe or just between two column.
        :param col_name_1: If only this one is given calculate correlation from this to all,
        if none is given calculate for all
        :param col_name_2: If only this one is given calculate correlation from this to all,
        if none is given calculate for all
        """
        if col_name_1 == "" and col_name_2 == "":
            # TODO implement calculation of correlation for all attributes and print it
            print()

        else:
            # TODO implement calculation of correlation just for the two given column names


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


class Analayzer:
    dataframe = None

    def __int__(self, dataframe):
        self.dataframe = dataframe

    def linear_regression(self, matrix_X, vector_Y):
        # TODO split into learn and test, automatic trying of linear, ridge and lasso
        # TODO Return the best model and its predictions
        pass
        return None

    def random_forest(self, matrix_X, vector_Y):
        # TODO copy random forest from the existing implementations
        # TODO return predictions
        pass
        return None

    def hierarhical_clustering(self):
        # TODO grouping based on all attributes, try out different sets of data
        # TODO returns different classes depending on row number
        pass
        return None
