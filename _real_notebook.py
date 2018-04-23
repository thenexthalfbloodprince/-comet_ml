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
import errno
import hashlib
import json
import logging
import os
from base64 import b64encode
from os.path import abspath, expanduser, isdir, isfile, join

from IPython import get_ipython
from notebook.base.handlers import IPythonHandler
from notebook.utils import url_path_join

from .connection import notebook_source_upload

LOGGER = logging.getLogger(__name__)
CACHE_DIR = expanduser("~/.comet_notebook_cache/")
NOTEBOOK_MAPPING = {}


def _jupyter_server_extension_paths():
    return [{"module": "comet_ml"}]


def _jupyter_nbextension_paths():
    return []


def load_jupyter_server_extension(nbapp):
    nbapp.log.info("Comet.ml extension activated!")

    contents_manager = nbapp.contents_manager
    contents_manager.pre_save_hook = pre_save_hook

    web_app = nbapp.web_app

    # make sure our static files are available
    static_files_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "static")
    )
    LOGGER.debug("Editing nbextensions path to add %s", static_files_path)
    if static_files_path not in web_app.settings["nbextensions_path"]:
        web_app.settings["nbextensions_path"].append(static_files_path)

    host_pattern = ".*$"
    web_app.add_handlers(
        host_pattern,
        [
            (
                url_path_join(web_app.settings["base_url"], r"/comet_ml"),
                NotebookHashMapperHandler,
            )
        ],
    )


def get_notebook_id():
    _Jupyter = get_ipython()
    if _Jupyter is None:
        return None

    try:
        return _Jupyter.kernel.shell.user_ns["NOTEBOOK_ID"]

    except KeyError:
        return None


def parse_jupyter_server_model(model):
    api_key = None

    for cell in model["content"]["cells"]:
        if cell["cell_type"] != "code":
            continue

        # Clean up the execution count in order to have stable hashes
        cell["execution_count"] = None

        for output in cell["outputs"]:
            output["execution_count"] = None

        for line in cell["source"].splitlines():
            if line.startswith("COMET_ML_API_KEY="):
                api_key = line[len("COMET_ML_API_KEY=") + 1:-1]

    return model, api_key


def get_hash_content(model):
    json_model = json.dumps(model, sort_keys=True)
    content_hash = hashlib.sha1(json_model.encode("utf-8")).hexdigest()

    return content_hash, json_model


def create_cache_dir():
    try:
        os.makedirs(CACHE_DIR)
    except (IOError, OSError) as error:
        if error.errno == errno.EEXIST and isdir(CACHE_DIR):
            return

        raise


def pre_save_hook(model, **kwargs):
    local_path = kwargs.pop("path", None)

    if local_path is None:
        LOGGER.warning("No notebook path given, cannot save it")
        return

    path = abspath(local_path)

    if model["type"] != "notebook":
        return

    # only run on nbformat v4
    if model["content"]["nbformat"] != 4:
        return

    new_model, api_key = parse_jupyter_server_model(copy.deepcopy(model))
    if api_key is None:
        LOGGER.info("Couldn't detect an api key in the notebook")
        return

    content_hash, json_model = get_hash_content(new_model)

    NOTEBOOK_MAPPING[path] = content_hash

    cache_file_name = "%s_%s" % (b64encode(path.encode("utf-8")), content_hash)
    cache_file_path = join(CACHE_DIR, cache_file_name)
    if isfile(cache_file_path):
        return

    notebook_source_upload(content_hash, json_model, api_key, path)

    try:
        create_cache_dir()

        # Create the empty cache file
        with open(cache_file_path, "a"):
            pass
    except IOError:
        # raise
        LOGGER.warning("Failed to create a cache file", exc_info=True)


class NotebookHashMapperHandler(IPythonHandler):

    def get(self):
        local_path = self.get_argument("jupyter_path", "")
        path = abspath(local_path)
        msg_json = json.dumps({"NOTEBOOK_HASH_ID": NOTEBOOK_MAPPING.get(path)})
        self.write(msg_json)
        self.flush()
