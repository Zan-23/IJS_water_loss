from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt


class Analyzer:
    dataframe = None
    random_f_regressor = None

    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.random_f_regressor = RandomForestRegressor(n_estimators=50, random_state=42)

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
        regressor_rf = RandomForestRegressor(n_estimators=50, random_state=42)
        regressor_rf.fit(learn_x, learn_y.ravel())
        y_predicted = regressor_rf.predict(test_x)

        r2_random_f = r2_score(test_y, y_predicted)
        print("Random forest regression R2: ", r2_random_f)
        diff = mean_squared_error(test_y, y_predicted)
        print("MSE result:", diff)

        return test_y, y_predicted, time_col

    def random_forest_train(self, dataframes, matrix_cols, vector_col, time_col_name=""):
        """
        This method builds a random forest regressor but doesnt't yet predict anything.
        The models is built on multiple dataframes.
        The regressor is stored in the class variable and can be used in random_forest_train method to predict data.

        :param dataframes: Array of dataframes...
        :param matrix_cols: Columns which you want to include in matrix X
        :param vector_col: Column which you want to predict
        :param time_col_name: Column in which the time series is stored(used for plotting predictions)
        :return: Returns the actual y values, predicted values of y and
        the timeframe for which the values were generated
        """
        regressor_rf = None

        for dataframe in dataframes:
            print(dataframe.head(3))
            matrix_x, vector_y, time_col = self.generate_matrix_and_vector(matrix_cols, vector_col, time_col_name, dataframe)
            if len(matrix_x) != len(vector_y):
                raise Exception("Matrix X and vector Y must be of the same length !!")
            learn_x = matrix_x[:len(matrix_x)]
            learn_y = vector_y[:len(vector_y)]

            # RANDOM FOREST - 50 trees
            self.random_f_regressor.fit(learn_x, learn_y.ravel())

    def random_forest_test(self, matrix_cols, vector_col, time_col_name=""):
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

        test_x = matrix_x[:len(matrix_x)]
        test_y = vector_y[:len(vector_y)]

        y_predicted = self.random_f_regressor.predict(test_x)
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

    def generate_matrix_and_vector(self, matrix_cols, vector_col, time_col, data_f=None):
        """
        This method generate matrix X and vector Y which are used in most of the methods of this for building models.

        :param matrix_cols: Columns which you want to include in matrix X
        :param vector_col: Column which you want to predict
        :param time_col:  Column in which the time series is stored(used for plotting predictions)
        :param data_f: if given, perform operations on this dataframe
        :return: Returns three arrays, matrix X and vector Y and timestamp array
        """
        data_frame = self.dataframe

        if data_f is not None:
            data_frame = data_f

        matrix_x = data_frame[matrix_cols].to_numpy()
        vector_col = data_frame[vector_col].to_numpy()
        timestamp_col = None

        if time_col != "":
            timestamp_col = data_frame[time_col].to_numpy()

        return matrix_x, vector_col, timestamp_col


def plot_real_and_predicted(real_y, predicted_y, time_x=None):
    if len(real_y) != len(predicted_y):
        raise Exception("Test data and predicted data must be of the same length !!")
    if time_x is None:
        time_x = [i for i in range(len(real_y))]

    fig = plt.figure(figsize=(18, 8), dpi=100, facecolor='w')
    plt.plot(time_x, real_y, color="red", label="actual")
    plt.plot(time_x, predicted_y, color="blue", label="predicted")
    plt.legend(loc="lower left")
    plt.show()