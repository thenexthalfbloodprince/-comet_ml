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

"""
Author: Gideon Mendels

This module contains the functions needed for Jupyter extensions in case we
are not in Jupyter server extension environment
"""

import copy
import hashlib
import json
import os
import sys


def _jupyter_server_extension_paths():
    return []


def _jupyter_nbextension_paths():
    return []


def load_jupyter_server_extension(nbapp):
    return


def get_notebook_id():
    return None
