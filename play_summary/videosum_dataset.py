import pandas as pd
import numpy as np
import pickle

import sys
sys.path.append('/home/emma/summary_evaluation/score_evaluators')
from prediction_data import PredictionData

class VideosumDataset(PredictionData):
    def __init__(self, name, num_classes, local=True):
        self.repr_name = name
        self.name = name.lower()
        self.ds_loc = '/movie-drive/created_summaries/{}/'.format(self.name)
        if num_classes == 1:
            self.model_name = self.name + '_one'
        else:
            self.model_name = self.name + '_two'
        print(self.model_name)
        super().__init__(self.name, num_classes, self.model_name, local)

        self.meta_data = self.get_meta_data()
        self.fps_df = self.get_fps()

        self.data_frame = self.get_data_set()
        self.segments = self.get_segments()

    def get_meta_data(self):
        file_path = self.name + '_ds.pickle'
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def get_fps(self):
        fps_df = pd.DataFrame()

        fps = []
        vids = []
        movie_names = []

        for movie_name, data in self.meta_data.items():
            fps.append(data['fps'])

            if self.name == 'moviesum':
                vid = int(movie_name.split('_')[0])
            else:
                vid = movie_name

            movie_names.append(movie_name)
            vids.append(vid)

        fps_df['movie_name'] = movie_names
        fps_df['vid'] = vids
        fps_df['fps'] = fps
        return fps_df

    def get_frame_second(self, fn, fps):
        frame_time = fn / fps
        return frame_time

    def get_data_set(self):
        data_frame = pd.DataFrame()
        data_frame['y_pred'] = self.y_pred
        data_frame['y_true'] = self.y_true
        if self.y_prob is not None:
            data_frame['y_prob'] = self.y_prob
        data_frame['vid'] = self.vids
        data_frame['fns'] = self.fns
        data_frame = pd.merge(data_frame, self.fps_df, how='left', left_on=['vid'], right_on=['vid'])
        data_frame['f_time'] = data_frame.apply(lambda x: self.get_frame_second(x.fns, x.fps), axis=1)
        data_frame = data_frame.sort_values(by=['vid', 'f_time'], ascending=True)
        data_frame['concat_pred'] = self.calc_concatenated_predictions(data_frame.y_true.values)
        return data_frame

    def calc_concatenated_predictions(self, y_pred):
        """ implement concat heuristics, such that two or one zeros between two ones also become one
        the method below does so by adding a padding of len=1 to all positive predictions
        the method below is chosen because it does not require iteration through the predictions"""
        add_1 = [int(x) for x in y_pred]
        add_1 = [0] + add_1[:-1]
        add_2 = [int(x) for x in y_pred]
        add_2 = add_2[1:] + [0]

        concat_pred = np.maximum(np.array(add_1), np.array(add_2))
        concat_pred = np.maximum(concat_pred, y_pred)
        return concat_pred

    def get_segments(self):

        frame_times = {}
        for video_name, video_data in self.data_frame.groupby(by='movie_name'):
            video_segments_data = video_data[video_data.y_pred == 1]
            frame_times[video_name] = list(video_segments_data.f_time.values)

        segments = {}
        for video_name, video_frame_times in frame_times.items():
            segments[video_name] = []
            segment = []
            for i, _ in enumerate(video_frame_times[1:]):
                diff = video_frame_times[i] - video_frame_times[i - 1]
                if abs(diff) < 1.1:
                    segment.append(video_frame_times[i-1])
                else:
                    if len(segment) > 1:
                        start_time = segment[0]
                        duration = segment[-1] - start_time
                        segments[video_name].append((start_time, duration, segment))
                    else:
                        segment = [video_frame_times[i]]
                        start_time = video_frame_times[i]
                        duration = 1
                        segments[video_name].append((start_time, duration, segment))
                    segment = []

        return segments


if __name__ == '__main__':
    moviesum = VideosumDataset(name='MovieSum', num_classes=2, local=False)
    tvsum = VideosumDataset(name='TVSum', num_classes=1, local=False)
    summe = VideosumDataset(name='SumMe', num_classes=1, local=False)


    t = 0.4
    # moviesum.get_y(threshold=t, random=False)
    # moviesum.y_thresh
    tvsum.get_y(threshold=t, random=False)
    tvsum.y_thres
    summe.get_y(threshold=t, random=False)
    summe.y_thres

