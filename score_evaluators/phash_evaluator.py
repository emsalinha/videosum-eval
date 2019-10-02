import glob
import os
import numpy as np
import h5py
import pandas as pd
from scipy.spatial.distance import cdist

from phash_plotter import Plot
from summ_evaluator import SummaryEvaluator


class PhashEvaluator(SummaryEvaluator):
    def __init__(self, ds_name, model_name, num_classes):
        super().__init__(ds_name=ds_name, model_name=model_name, num_classes=num_classes)

        self.root = '/movie-drive/phashing/hashes/{}'.format(self.ds_name)

        self.df_hashes = self.__load_hashes__()

        self.df_true = self.__join_hashes__(self.y_true)
        self.df_pred = self.__join_hashes__(self.y_pred)
        self.df_rand = self.__join_hashes__(self.y_rand)




    def get_metrics(self, metric='hamming', random=False, threshold=None, print_output=False):

        if threshold != None:
            y = self.get_y(random, threshold)
            df = self.__join_hashes__(y)

        if random:
            df = self.df_rand
        else:
            df = self.df_pred

        matched_hashes = np.vstack(df[df.y == 1].hash.values).astype(np.int)
        not_matches_hashes = np.vstack(df[df.y == 0].hash.values).astype(np.int)
        true_hashes = np.vstack(self.df_true[self.df_true.y == 1].hash.values).astype(np.int)

        self.distances = cdist(matched_hashes, true_hashes, metric).min(axis=0)

        self.distances_mean = self.distances.mean()
        print('mean distances predicted and true trailer: ', self.distances_mean)
        self.distances_std = self.distances.std()
        self.distances_min = self.distances.min()
        self.distances_max = self.distances.max()

        self.non_distances = cdist(not_matches_hashes, true_hashes, metric).min(axis=0)

        self.non_distances_mean = self.non_distances.mean()
        print('mean distances not predicted and true trailer: ', self.non_distances_mean)
        self.non_distances_std = self.non_distances.std()
        self.non_distances_min = self.non_distances.min()
        self.non_distances_max = self.non_distances.max()

        return self.distances_mean, self.non_distances_mean

    def plot(self, random=False):
        plotter = Plot(window_size=1, run_name=self.model_name)
        if random:
            self.get_metrics(random=True)
            pred_paths = self.df_rand[self.df_rand.y == 1].path.values
            non_pred_paths = self.df_rand[self.df_rand.y == 0].path.values
        else:
            pred_paths = self.df_pred[self.df_pred.y == 1].path.values
            non_pred_paths = self.df_pred[self.df_pred.y == 0].path.values

        true_paths = self.df_true[self.df_true.y == 1].path.values
        plotter.plot(self.distances, pred_paths, true_paths)
        plotter.plot(self.non_distances, non_pred_paths, true_paths)

    def __join_hashes__(self, y):
        df = pd.DataFrame()
        df['y'] = y.flatten()
        df['frame_nr'] = self.fns
        df['vid'] = self.vids
        df = pd.merge(df, self.df_hashes, how='left', left_on=['vid', 'frame_nr'], right_on=['vid', 'frame_nr'])
        return df

    def __load_hashes__(self):
        hashes = []
        frame_nrs = []
        vids = []
        for hash_dir in glob.glob(self.root + '/*'):
            if self.ds_name == 'moviesum':
                vid = int(os.path.basename(hash_dir).split('_')[0])
                ds_name_hash = 'unaugmented/DCT_hash/12/hashes'
                ds_name_frame_nrs = 'unaugmented/DCT_hash/12/frames'
            else:
                vid = os.path.basename(hash_dir)
                ds_name_hash = 'unaugmented/DCTHash/12/hashes'
                ds_name_frame_nrs = 'unaugmented/DCTHash/12/frames'

            hash_file = glob.glob(hash_dir + '/*')[0]
            hash_store = h5py.File(hash_file, 'r')
            stored_hashes = hash_store[ds_name_hash][:]
            stored_frame_nrs = hash_store[ds_name_frame_nrs][:]
            hashes += [list(hash) for hash in list(stored_hashes)]
            frame_nrs += [int(x) for x in list(stored_frame_nrs)]
            vids += [vid] * len(stored_frame_nrs)

        df = pd.DataFrame()
        df['hash'] = hashes
        df['frame_nr'] = frame_nrs
        df['vid'] = vids
        df['path'] = df.apply(lambda x: self.__fn_to_path__(x.frame_nr, x.vid), axis=1)
        return df

    def __fn_to_path__(self, fn, vid):
        if self.ds_name == 'moviesum':
            path = os.path.join(self.root, str(vid), 'frame_' + str(fn).zfill(6) + '.jpg')
        else:
            path = os.path.join(self.root, str(fn).zfill(5) + '.png')
        return path

