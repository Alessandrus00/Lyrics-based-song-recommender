import pandas as pd
from random import seed, sample
import pickle
import glob
import os

class Loader():

    def __init__(self, in_path, out_path):
        """
        Args:
            in_path (str): csv input path.
            out_path (str): Output directory path to store the pickles.
            chunksize (int, optional): Chunksize for DataFrame reader. Defaults to 10**6. 
        """

        self.__in_path = in_path
        self.__out_path = out_path
        self.__chunksize = 10**6

    def __produce_pickles(self):
        """produce pickles by reading csv by chunksize
        """
        with pd.read_csv(self.__in_path, chunksize = self.__chunksize) as reader:
            try:
                os.makedirs(self.__out_path)
            except FileExistsError:
                # directory already exists
                pass
            for i, chunk in enumerate(reader):
                out_file = self.__out_path + "/data_{}.pkl".format(i+1)
                with open(out_file, "wb") as f:
                    pickle.dump(chunk, f, pickle.HIGHEST_PROTOCOL)
    
    def load_pickle(self, pickle_id):
        """load a pickle file by id

        Args:
            pickle_id (int): pickle id.

        Raises:
            Exception: The path of the given id isn't a file

        Returns:
            obj: DataFrame
        """
        # produce the pickles if the directory not exists or
        # if the directory is empty 
        if (not os.path.exists(self.__out_path)) or \
              (len(os.listdir(self.__out_path)) == 0):
            self.__produce_pickles()
        
        # get the file path following the pickle_id
        # given in parameter
        file_path = self.__out_path + \
            "/data_" + str(pickle_id) + ".pkl"

        if os.path.isfile(file_path):
            df = pd.read_pickle(file_path)
        else:
            raise Exception("The pickle file data_{}.pkl doesn't exist".format(pickle_id))
        return df
        

    def random_pickles(self, n_pickles = 3, init = 42, verbose = True):
        """random reader over pickles files

        Args:
            n_pickles (int, optional): number of pickles to load. Defaults to 3.
            init (int, optional): Integer given to the random seed. Defaults to 42.
            verbose (bool, optional): Print the loaded files. Defaults to True

        Raises:
            Exception: Stop the process if n_pickles exceed pickle files number.

        Returns:
            obj: pd.Dataframe
        """

        # produce the pickles if the directory not exists or
        # if the directory is empty 
        if (not os.path.exists(self.__out_path)) or \
              (len(os.listdir(self.__out_path)) == 0):
            self.__produce_pickles()

        pickle_files = [name for name in
                        glob.glob(self.__out_path + "/data_*.pkl")]
        # draw p_files        
        seed(init)

        if n_pickles <= 6:
            random_p_files = sample(pickle_files, n_pickles)
        else:
            raise Exception("The parameter n_pickles (" +
                            "{}) exceed the numbers of pickle files ({})"\
                                .format(n_pickles, len(pickle_files)))
        # print the drawed files
        if verbose:
            print("Loaded pickles:")
            for p in random_p_files:
                print(p)

        # load random pickles file
        df_list = [pd.read_pickle(p) for p in random_p_files]

        # create the dataframe by concatenate the previous
        # dataframes list
        df = pd.concat(df_list, ignore_index = True)
        return df

loader = Loader(in_path = './input/song_lyrics.csv', out_path = './input/data')
df_orig = loader.load_pickle(1)
df_sampled = df_orig.sample(frac=0.01, random_state=1)
df_sampled.to_csv("./data/song_lyrics_sampled.csv", header='true', index=False)