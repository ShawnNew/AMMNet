import importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


class WriterTensorboardX():
    def __init__(self, writer_dir, logger, enable):
        self.writer = None
        if enable:
            log_path = writer_dir
            try:
                self.writer = importlib.import_module('tensorboardX').SummaryWriter(log_path)
            except ModuleNotFoundError:
                message = "Warning: TensorboardX visualization is configured to use, but currently not installed on this machine. " + \
                          "Please install the package by 'pip install tensorboardx' command or turn off the option in the 'config.json' file."
                logger.warning(message)
        self.step = 0
        self.mode = ''

        self.tensorboard_writer_ftns = ['add_scalar', 'add_scalars', 'add_image', 'add_audio', 'add_text', 'add_histogram', 'add_pr_curve', 'add_embedding']

    def set_step(self, step, mode='train'):
        self.mode = mode
        self.step = step

    def __getattr__(self, name):
        """
        If visualization is configured to use:
            return add_data() methods of tensorboard with additional information (step, tag) added.
        Otherwise:
            return blank function handle that does nothing
        """
        if name in self.tensorboard_writer_ftns:
            add_data = getattr(self.writer, name, None)
            def wrapper(tag, data, *args, **kwargs):
                if add_data is not None:
                    add_data('{}/{}'.format(self.mode, tag), data, self.step, *args, **kwargs)
            return wrapper
        else:
            # default action for returning methods defined in this class, set_step() for instance.
            try:
                attr = object.__getattr__(name)
            except AttributeError:
                raise AttributeError("type object 'WriterTensorboardX' has no attribute '{}'".format(name))
            return attr


def decode_segmap(temp, n_classes):
    
    label_colours = np.asarray([
        [0,0,0],
        [128,128,128],
        [255,255,255]
        ])
    rgb = np.zeros((temp.shape[0], 3, temp.shape[1], temp.shape[2]))
    # decode_list = []
    for i in range(len(temp)):
        img = temp[i]
        r = img.copy()
        g = img.copy()
        b = img.copy()
        for l in range(0, n_classes):
            r[img == l] = label_colours[l, 0]
            g[img == l] = label_colours[l, 1]
            b[img == l] = label_colours[l, 2]
        
        rgb[i, 0, :, :] = r
        rgb[i, 1, :, :] = g
        rgb[i, 2, :, :] = b


    return rgb