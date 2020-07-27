
import pandas as pandas


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
            # , dtype="string"
        elif separator == "json":
            self.data = pandas.read_json(file_name)

        else:
            raise Exception("Unknown type of separator, please remodel read_data function !")

    def transform_data(self, city, del_cols=False):
        """
        :param city: "alicante" or "braila", set it to to the place where the data originated
        :param del_cols: This is a flag which deletes unnecessary rows -> saving RAM
        """

        if city == "braila":
            self.data = self.data[(self.data.tot1 != 0) & (self.data.analog2 != 0)]
        elif city == "alicante":
            first_col_splitted = self.data.iloc[:, 1].str.split(',', expand=True)
            first_col_splitted.columns = ['Water flow - 1(Sensor 1)', 'Water flow - 2(Sensor 1)']
            date_1_col = self.data.iloc[:, 0]

            if not del_cols:
                date_2_col = self.data.iloc[:, 2]
                second_col_splitted = self.data.iloc[:, 3].str.split(',', expand=True)
                second_col_splitted.columns = ['Water flow - 1(Sensor 2)', 'Water flow - 2(Sensor 2)']

                new_data_f = pandas.concat([date_1_col, first_col_splitted, date_2_col, second_col_splitted], axis=1)
                self.data = new_data_f
            else:
                # If flag is set, only the first 3 columns are returned as specified on drive
                new_data_f = pandas.concat([date_1_col, first_col_splitted], axis=1)
                self.data = new_data_f
        else:
            raise Exception("Unknown city, modify transform_data to process other cities")

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
