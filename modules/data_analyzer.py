from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.metrics import silhouette_score
from math import floor

# TODO predict states that lead into anomalies


class Analyzer:
    dataframe = None

    def __init__(self, dataframe):
        self.dataframe = dataframe

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
        This method builds a random forest regressor on multiple dataframes.
        Method can be modified so that the regressor is stored and can me used to predict data without train set.

        :param dataframes: Array of dataframes...
        :param matrix_cols: Columns which you want to include in matrix X
        :param vector_col: Column which you want to predict
        :param time_col_name: Column in which the time series is stored(used for plotting predictions)
        :return: Returns the actual y values, predicted values of y and
        the timeframe for which the values were generated
        """
        random_f_regressor = RandomForestRegressor(n_estimators=50, random_state=42)

        for dataframe in dataframes:
            print(dataframe.head(3))
            matrix_x, vector_y, time_col = self.generate_matrix_and_vector(matrix_cols, vector_col, time_col_name, dataframe)

            learn_x = matrix_x[:len(matrix_x)]
            learn_y = vector_y[:len(vector_y)]

            # RANDOM FOREST - 50 trees
            random_f_regressor.fit(learn_x, learn_y.ravel())

        matrix_x, vector_y, time_col = self.generate_matrix_and_vector(matrix_cols, vector_col, time_col_name)
        test_x = matrix_x[:len(matrix_x)]
        test_y = vector_y[:len(vector_y)]

        y_predicted = random_f_regressor.predict(test_x)
        r2_random_f = r2_score(test_y, y_predicted)
        print("Random forest regression R2: ", r2_random_f)
        diff = mean_squared_error(test_y, y_predicted)
        print("MSE result:", diff)

        return test_y, y_predicted, time_col

    def hierarhical_clustering(self, matrix_cols, vector_col, num_of_clus, time_col_name=""):
        """

        :param num_of_clus:
        :param matrix_cols:
        :param vector_col:
        :param num_of_clusters:
        :param time_col_name:
        :return:
        """
        matrix_x, vector_y, time_col = self.generate_matrix_and_vector(matrix_cols, vector_col, time_col_name)
        # print(len(matrix_x))
        linkage_method = sch.linkage(matrix_x[:, :1], method="complete", metric="cityblock")
        """plt.figure(figsize=(25, 15))

        d = sch.dendrogram(linkage_method, leaf_font_size=15, orientation="left",
                           show_contracted=True,
                           truncate_mode='lastp',  # last p merged clusters
                           p=num_of_clus  # show only the last p merged clusters
                           )
        plt.xlabel("Razdalja")
        plt.ylabel("Datum")
        plt.show()"""
        print("Done")
        arr_, t_best = find_optimal_t_cluster(linkage_method, matrix_x, num_of_clus)
        predictions = sch.fcluster(linkage_method, t=t_best, criterion="distance").ravel()
        print("Done 2")
        class_dict = dict()
        for index, combined in enumerate(zip(time_col, predictions)):
            value, predicted_class = combined
            if predicted_class not in class_dict:
                class_dict[predicted_class] = [(index, value)]
            else:
                class_dict[predicted_class].append((index, value))

        draw_classes(time_col, vector_y, class_dict)
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
        vector_y = data_frame[vector_col].to_numpy()
        timestamp_col = None

        if time_col != "":
            timestamp_col = data_frame[time_col].to_numpy()

        if len(matrix_x) != len(vector_y):
            raise Exception("Matrix X and vector Y must be of the same length !!")

        return matrix_x, vector_y, timestamp_col


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


def find_optimal_t_cluster(L, matrix_fun, num_of_clusters):
    max_silhuete = [-1, 0]
    arr = []

    max_val = (floor(max(sch.maxdists(L)) * 100) / 100.0) * 0.66
    t_f = max_val

    while True:
        predictions = sch.fcluster(L, t=t_f, criterion="distance").ravel()
        score = round(silhouette_score(matrix_fun, predictions, metric="cityblock"), 3)
        arr.append((score, round(t_f, 4)))

        num_of_c = len(set(predictions))
        if num_of_c == num_of_clusters:
            max_silhuete[0] = score
            max_silhuete[1] = round(t_f, 4)
            break
        elif num_of_c < num_of_clusters:
            t_f = t_f - (t_f / 2)
        else:
            t_f = t_f + (t_f / 2)
        # print(t_f)

    print("The best value for t is t =", max_silhuete[1])
    return arr, t_f


def draw_classes(time_x, y_actual, class_dict):
    color_array = ["yellowgreen", "coral", "dodgerblue", "red", "springgreen", "aqua",
                   "mistyrose", "deeppink", "bisque", "indigo"]
    size_arr = generate_sizes(class_dict)

    fig = plt.figure(figsize=(18, 12), dpi=100, facecolor='w')
    plt.plot(time_x, y_actual, color="pink")

    for num, element in enumerate(class_dict):
        label_name = "Class: " + str(element) + " - Num of elements: " + str(len(class_dict[element]))
        plt.scatter([x[1] for x in class_dict[element]], [y_actual[x[0]] for x in class_dict[element]],
                    color=color_array[num], s=size_arr[num],
                    label=label_name)

    plt.ylabel('Pressure in bar ?')
    plt.xlabel('Date')
    plt.legend()
    plt.title('Time')
    plt.show()


def generate_sizes(class_dict):
    class_len_arr = []
    for i in class_dict:
        class_len_arr.append([i, len(class_dict[i])])

    class_sorted_arr = sorted(class_len_arr, key=lambda x: x[1])

    size_arr = []
    anomaly_koef = get_anomaly_coeficient(class_sorted_arr)

    for i in class_len_arr:
        if i[1] <= anomaly_koef:
            size_arr.append(200)

        elif i[1] <= (anomaly_koef * 2):
            size_arr.append(100)
        else:
            size_arr.append(20)

    return size_arr


def get_anomaly_coeficient(sorted_arr):
    return sorted_arr[len(sorted_arr) - 1][1] * 0.05