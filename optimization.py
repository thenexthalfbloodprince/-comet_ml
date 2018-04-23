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
#
'''
Author: Boris Feld

This module contains the various helpers for the Optimization API
'''
import logging
from comet_ml import config

from .exceptions import PCSCastingError, PCSParsingError, OptimizationMissingExperiment
from ._logging import CASTING_ERROR_MESSAGE

LOGGER = logging.getLogger(__name__)


class Suggestion(object):
    """ A suggestion is a single proposition of hyper-parameters values.

    You can use it like a dict:

    ```python
    suggestion["x"] # Returns the value for hyper-parameter x
    ```

    Suggestion is automatically casting values for hyper-parameter declared as
    `integer` or `real`. It will returns `int` for `integer` and `float` for
    `real`. For `categorical` and `ordinal`, it will returns `str`.

    In case casting is failing, it will print a warning message and return a `str`.

    For accessing the raw value without casting, you can use the `raw` method:

    ```python
    suggestion.raw("x") # Always returns a str
    ```
    """

    def __init__(self, suggestion, optimizer, types):
        """ You shouldn't directly instantiate Suggestion objects, use
        [Optimizer.get_suggestion](/Optimizer/#optimizerget_suggestion)
        instead.
        """
        self.suggestion = suggestion
        self.run_id = suggestion["run_id"]
        self.params = suggestion["params"]
        self.optimizer = optimizer
        self.types = types

    def __iter__(self):
        return iter(self.params)

    def __getitem__(self, name):
        """ Return the casted value for this hyper-parameter.
        Args:
            name: The hyper-parameter name
        """
        raw_value = self.params[name]
        try:
            return cast_parameter(raw_value, self.types[name])
        except (KeyError, PCSCastingError):
            LOGGER.warning(CASTING_ERROR_MESSAGE, name, name, name)
            return self.raw(name)

    def raw(self, name):
        """ Return the raw not-casted value for this hyper-parameter.
        Args:
            name: The hyper-parameter name
        """
        return self.params[name]

    def report_score(self, name, score):
        """ Send back the score for this suggestion.
        Args:
            score: A float representing the score
        """
        self._report_params_to_experiment(self.params, name, score)

        self.optimizer._report_score(self.run_id, score)

    def _report_params_to_experiment(self, suggestion, name, score):
        if config.experiment is None:
            raise OptimizationMissingExperiment

        exp = config.experiment

        exp.log_multiple_params(suggestion)
        exp.log_metric(name, score)




PCS_TYPES = {"integer", "real", "ordinal", "categorical"}


def parse_pcs(pcs_content):
    parsed = {}
    for line in pcs_content.splitlines():
        # Clean line
        line = line.strip()

        # Ignore empty lines
        if line == "":
            continue

        # Ignore commented lines
        if line.startswith("#"):
            continue

        # Ignore conditions as they doesn't influence parameters type
        if "|" in line:
            continue

        # Ignore forbidden parameter syntax
        if line.startswith("{") and line.endswith("}"):
            continue

        # Check that the line looks valid
        if "}" not in line and "]" not in line:
            raise PCSParsingError(line)

        splitted = line.split(" ")

        if len(splitted) < 2:
            raise PCSParsingError(line)

        param_name = splitted[0]
        param_type = splitted[1]

        # Check that type is valid
        if param_type not in PCS_TYPES:
            raise PCSParsingError(line)

        parsed[param_name] = param_type

    return parsed


def cast_parameter(value, pcs_type):
    if pcs_type not in PCS_TYPES:
        raise PCSCastingError(value, pcs_type)

    if pcs_type == "integer":
        try:
            return int(value)
        except ValueError:
            raise PCSCastingError(value, pcs_type)
    elif pcs_type == "real":
        try:
            return float(value)
        except ValueError:
            raise PCSCastingError(value, pcs_type)
    elif pcs_type == 'categorical':
        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        try:
            return _str_to_bool(value)
        except ValueError:
            pass

    return value


def _str_to_bool(s):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        raise ValueError()
