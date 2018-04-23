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

import sys

from comet_ml import config


def fit_logger(real_fit):
    def wrapper(*args, **kwargs):
        if config.experiment and config.experiment.disabled_monkey_patching is False:
            callback = config.experiment.get_keras_callback()
            if 'callbacks' in kwargs and kwargs['callbacks'] is not None:
                # Only append the callback if it's not there.
                if not any(x.__class__ == callback.__class__
                           for x in kwargs['callbacks']):
                    kwargs['callbacks'].append(callback)
            else:
                kwargs['callbacks'] = [callback]

        return real_fit(*args, **kwargs)

    return wrapper


def patch(module_finder):
    module_finder.register('keras.models', 'Model.fit', fit_logger)
    module_finder.register('keras.models', 'Model.fit_generator', fit_logger)


if "keras" in sys.modules:
    raise SyntaxError("Please import Comet before importing any keras modules")
