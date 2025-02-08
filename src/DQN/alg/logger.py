"""Logger script for saving the model and the training stats

"""

import csv
import os.path as osp, os
from datetime import datetime

import yaml

class Logger:
    """
    A class used to represent the stats Logger
    """
    
    def __init__(self, metrics, out_fname, out_dir, args):
        """Gets the training details and initiate the Logger

        Args:
            metrics: list of parameters name to store

        """

        dir_name = 'log/'

        dt = datetime.now().strftime('%d-%m-%Y--%H.%M.%S') + '/'   # dd/mm/YY H:M:S

        out_dir = dir_name + out_dir + dt
        if not osp.exists(dir_name): os.makedirs(out_dir)

        self.metric_dir = out_dir + "metric/"
        os.makedirs(self.metric_dir)

        self.model_dir = out_dir + "model/"
        os.makedirs(self.model_dir)

        self.metric_file = open(self.metric_dir + out_fname + '.csv', mode='w')
        self.writer = csv.DictWriter(self.metric_file, fieldnames=metrics, lineterminator='\n')
        self.writer.writeheader()

        # Save algorithm configuration
        with open(out_dir + 'config.yaml', 'w') as cfg:
            yaml.dump(args, cfg)

    def write(self, stats):
        self.writer.writerow(stats)
        self.metric_file.flush()

    def log_close(self):
        self.metric_file.close()

    def save_model(self, model, e):
        """Save the model

        Args:
            model (Model): model to save
            epoch (int): epoch of the saved model
            success (int): performance of the saved model
                       
        Returns:
            None
        """

        model.save(self.model_save + '_epoch' + str(e) + '.h5')