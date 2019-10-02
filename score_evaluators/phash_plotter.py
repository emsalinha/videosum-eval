import logging
import abc
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clr

class Plotter(abc.ABC):

    def __init__(self, window_size: int, run_name: str):
        self.run_name = run_name
        self.window_size = window_size

    def plot(self, distances, list_of_img_file_paths_y: list, list_of_img_file_paths_x: list, normalize=False):
        self.distances = distances
        self.list_of_img_file_paths_y = list_of_img_file_paths_y
        self.list_of_img_file_paths_x = list_of_img_file_paths_x

        f= plt.figure()
        ax = f.add_subplot(111)

        title, plot_name = self.__get_title_and_name_graph__(list_of_img_file_paths_y, list_of_img_file_paths_x)
        ax.set_title(title)
        ax.set_ylabel('frame')
        ax.set_xlabel('frame')

        if normalize:
            norm = clr.Normalize(vmin=0.9, vmax=1, clip=False)
            im = ax.imshow(self.distances, norm = norm)
        else:
            im = ax.imshow(self.distances)

        plt.colorbar(im)

        cid = f.canvas.mpl_connect('button_press_event',
                                   lambda event: self.__onclick__(event))

        plot_path = os.path.join(os.getenv("HOME"), 'plots')

        plt.savefig(plot_path + '/plot_{}{}.pdf'.format(plot_name, self.run_name), orientation="landscape")
        plt.show()

    def __onclick__(self, event):
        if event.dblclick:
            self.clicked_x = int(np.round(event.xdata))
            self.clicked_y = int(np.round(event.ydata))
            self.distance_of_clicked_x_y = self.distances[self.clicked_y, self.clicked_x]
            print('distance: {}, of {} & {}'.format(self.distance_of_clicked_x_y, self.clicked_y, self.clicked_x))
            self.__plot_images__()

    @abc.abstractmethod
    def __plot_images__(self):
        pass

    def __get_title_and_name_graph__(self, list_of_img_file_paths_y, list_of_img_file_paths_x):
        path_y = os.path.dirname(list_of_img_file_paths_y[0])
        path_x = os.path.dirname(list_of_img_file_paths_x[0])
        label_y = '-'.join(path_y.split('/')[5:])
        label_x = '-'.join(path_x.split('/')[5:])
        title = '{} graph {} & {}'.format(self.run_name, label_y, label_x)

        plot_name = '-'.join(path_y.split('/')[-3:-2])
        return title, plot_name


class Plot(Plotter):
    """use for plotting when windowsize = 1, meaning no more than one image is used per embedding"""
    def __init__(self, window_size: int, run_name: str):
        super().__init__(window_size, run_name)
        self.run_name = run_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def __load_clicked_img__(self, list_of_img_file_paths, clicked):
        path = os.path.dirname(list_of_img_file_paths[0])
        os.chdir(path)
        name_img = os.path.basename(list_of_img_file_paths[clicked])
        img = cv2.imread(name_img, 1)
        return img, name_img

    def __plot_images__(self):
        # called when clicking a point in the heatgraph

        img_y, name_img_y = self.__load_clicked_img__(self.list_of_img_file_paths_y, self.clicked_y)
        img_x, name_img_x = self.__load_clicked_img__(self.list_of_img_file_paths_x, self.clicked_x)

        merged_img = np.hstack((img_y, img_x))
        merged_img = cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)

        # show
        f = plt.figure()
        ax = f.add_subplot(111)

        ax.imshow(merged_img)

        title_subplot = 'distance: {}, Y:{}, X:{}'.format(self.distance_of_clicked_x_y, name_img_y, name_img_x)
        ax.set_title(title_subplot)
        plt.axis('off')
        plt.savefig("/tmp/plot_{}.pdf".format(title_subplot), orientation="landscape")
        plt.show()


class PlotAggregatedEmbeddings(Plotter):
    """use for plotting when windowsize > 1, meaning an aggregation of more than one image is used per embedding"""

    def __init__(self, window_size:int, run_name: str):
        super().__init__(window_size, run_name)
        self.run_name = run_name
        self.logger = logging.getLogger(self.__class__.__name__)

    def __load_clicked_images__(self, list_lists_of_img_file_paths, clicked):
        clicked_img_file_paths = list_lists_of_img_file_paths[int(np.round(clicked))]
        path = os.path.dirname(clicked_img_file_paths[0])
        os.chdir(path)
        names_images = [os.path.basename(filename_path) for filename_path in clicked_img_file_paths]
        images = [cv2.imread(filename, 1) for filename in names_images]
        return images, names_images

    def __vertically_stack_images__(self, images):
        init = np.vstack((images[0], images[1]))
        if self.window_size > 2:
            stacked_images = init
            for i in range(2, len(images)):
                stacked_images = np.vstack((stacked_images, images[i]))
            return stacked_images
        else:
            return init

    def __plot_images__(self):

        images_y, names_images_y = self.__load_clicked_images__(self.list_of_lists_of_img_file_paths_y, self.clicked_y)
        images_x, names_images_x = self.__load_clicked_images__(self.list_of_lists_of_img_file_paths_x, self.clicked_x)

        stacked_images_y = self.__vertically_stack_images__(images_y)
        stacked_images_x = self.__vertically_stack_images__(images_x)

        merged_img = np.hstack((stacked_images_y, stacked_images_x))
        merged_img = cv2.cvtColor(merged_img, cv2.COLOR_BGR2RGB)

        # show
        f = plt.figure()
        ax = f.add_subplot(111)

        ax.imshow(merged_img)
        title_subplot = 'distance: {}, Y:{}, X:{}'.format(self.distance_of_clicked_x_y, names_images_x, names_images_y)

        ax.set_title(title_subplot)
        plt.axis('off')
        plt.savefig("/tmp/plot_{}.pdf".format(title_subplot), orientation="landscape")
        plt.show()

