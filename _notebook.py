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

This module contains the functions related to Ipython/Jupyter notebook support
"""

import copy
import hashlib
import json
import os
import sys

NOTEBOOK_MAPPING = {}

try:
    import notebook
    import IPython
    from notebook.utils import url_path_join
    from notebook.base.handlers import IPythonHandler
    from IPython.core.magics.namespace import NamespaceMagics
    from IPython import get_ipython

    from ._real_notebook import (
        _jupyter_server_extension_paths,
        _jupyter_nbextension_paths,
        load_jupyter_server_extension,
        get_notebook_id,
    )

    HAS_NOTEBOOK = True
except ImportError:
    HAS_NOTEBOOK = False

    from ._not_notebook import (
        _jupyter_server_extension_paths,
        _jupyter_nbextension_paths,
        load_jupyter_server_extension,
        get_notebook_id,
    )


def in_notebook_environment():
    return "ipykernel" in sys.modules
