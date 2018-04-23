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

'''
Author: Boris Feld

This module contains logging configuration for Comet

'''

import logging

MSG_FORMAT = "COMET %(levelname)s: %(message)s"

INTERNET_CONNECTION_ERROR = ("Failed to establish connection to Comet server. Please check your internet connection. "
                "Your experiment would not be logged")

IPYTHON_NOTEBOOK_WARNING = (
    "Comet.ml support for Ipython Notebook is limited at the moment,"
    " automatic monitoring and stdout capturing is deactivated")


METRIC_ARRAY_WARNING = (
    "Cannot safely convert %r object to a scalar value, using it string"
    " representation for logging."
)

EXPERIMENT_OPTIMIZER_API_KEY_MISMTACH_WARNING = (
    "WARNING: Optimizer and Experiments API keys mismatch. Please use"
    " the same API key for both.")


PARSING_ERR_MSG = """We failed to parse your parameter configuration file.

Type casting will be disabled for this run, please fix your configuration file.
"""

CASTING_ERROR_MESSAGE = """Couldn't cast parameter %r, returning raw value instead.
Please report it to comet.ml and use `.raw(%r)` instead of `[%r]` in the meantime."""

NOTEBOOK_MISSING_ID = (
    "We detected that you are running inside a Ipython/Jupyter notebook environment but we cannot save your notebook source code."
    " Please be sure to have installed comet_ml as a notebook server extension by running:\n"
    "jupyter comet_ml enable"
)


def setup(level):
    root = logging.getLogger("comet_ml")

    root.setLevel(level)

    # Don't send comet-ml to the application logger
    root.propagate = False

    # Add handler
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(MSG_FORMAT))
    root.addHandler(console)
