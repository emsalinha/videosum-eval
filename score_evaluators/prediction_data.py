import abc
import h5py
import numpy as np
import os


class PredictionData(abc.ABC):
    def __init__(self, ds_name, num_classes, model_name, local=False):
        self.ds_name = ds_name
        self.num_classes = num_classes
        self.model_name = model_name
        self.file_name = self.get_file(local)
        self.y_true, self.y_pred, self.y_prob, self.fns, self.vids = self.get_data()
        self.y_rand = self.get_random_pred()
        self.y_thres = None

    def get_file(self, local):
        file_name = '/movie-drive/new_models/{}/y_pred_{}.hdf5'.format(self.model_name, self.model_name)
        if local:
            loc_to = os.getcwd()
            local_file_name = os.path.join(loc_to, os.path.basename(file_name))

            if os.path.exists(local_file_name):
                pass
            else:
                command = 'scp emma@52.142.213.14:{} {}'.format(file_name, loc_to)
                os.system(command)

            file_name = local_file_name
        return file_name

    def get_data(self) -> tuple:
        hdf5_store = h5py.File(self.file_name, 'r')

        y_true, y_pred = hdf5_store['y_true'][:], hdf5_store['y_pred'][:]

        y_prob = None
        try:
            y_prob = hdf5_store['y_prob'][:len(y_true)]
        except:
            print('probabilities of predictions of {} not saved'.format(self.ds_name))

        if self.num_classes == 2:
            y_pred = y_pred[:, 1]
            y_true = y_true[:, 1]

        elif self.num_classes == 1:
            y_pred = y_pred.flatten()
            y_true = y_true.flatten()

            if y_prob is not None:
                y_prob = y_prob.flatten()
                # y_prob = np.vstack(y_prob[:]).astype(np.float).flatten()
                # y_prob = y_prob[:len(y_true)]

        fns = hdf5_store['frame_nr'][::64][:int(len(y_true)/64)]
        fns = np.vstack(fns[:]).astype(np.int).flatten()

        vids = hdf5_store['vid'][:len(y_true)]

        if self.ds_name == 'moviesum':
            vids = np.vstack(vids[:]).astype(np.int).flatten()
        else:
            vids = np.vstack(vids[:]).astype(np.str).flatten()

        return y_true, y_pred, y_prob, fns, vids

    def get_random_pred(self):
        # n_pred = self.y_pred.sum()
        # rand_pred = np.hstack((np.ones((n_pred)), np.zeros(len(self.y_pred) - n_pred)))
        # np.random.shuffle(rand_pred)
        if self.ds_name.lower() == 'moviesum':
            rand_pred = np.random.randint(2, size=len(self.y_pred))
        else:
            rand_pred = np.random.randint(2, size=len(self.y_pred))
            #rand_pred = np.random.rand((len(self.y_pred)))
        return rand_pred

    def get_y(self, random, threshold):

        if threshold != None:
            y = np.where(self.y_prob < threshold, 0.0, 1.0)
        else:
            y = self.y_pred
        if random:
            y = self.y_rand

        self.y_thres = y
        return y
