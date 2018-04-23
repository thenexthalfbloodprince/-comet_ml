# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2015-2019 Comet ML INC
#  This file can not be copied and/or distributed without the express
#  permission of Comet ML Inc.
# *******************************************************

import json
from copy import deepcopy
from os.path import isfile

DEBUG = False

# Global experiment placeholder. Should be set by the latest call of Experiment.init()
experiment = None

# Global Optimizer
optimizer = None



# TODO: Allow the configuration file path to be set by a environment variable?
CONFIG_PATH = ".comet_ml.json"

DEFAULT_CONFIG = {
    "uploaded_extensions": [".py", ".txt", ".json"],
}




def read_config_file(config, config_path):
    """ read and parse the JSON comet_ml configuration file """
    if isfile(config_path):
        with open(config_path) as config_file:
            raw_config = json.load(config_file)

            config.update(raw_config)

    return config


def get_config():
    """ return a copy of the default configuration
    """
    return deepcopy(DEFAULT_CONFIG)
