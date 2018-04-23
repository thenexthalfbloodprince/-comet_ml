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

This module contains comet generated Exceptions

'''

class OptimizationMultipleParams(Exception):

    def __str__(self):
        return "You cannot call multiple time set_params or set_params file"


class NotParametrizedException(Exception):

    def __str__(self):
        return "Please call set_params or set_params_file before calling get_suggestion"


class ValidationError(Exception):
    pass


class AuthenticationError(Exception):
    pass


class NoMoreSuggestionsAvailable(Exception):

    def __str__(self):
        return "No more suggestions available!"


class PCSParsingError(Exception):
    """ Exception raised when parsing a PCS file
    """

    def __init__(self, faulty_line):
        self.faulty_line = faulty_line

    
class PCSCastingError(Exception):

    def __init__(self, value, pcs_type):
        self.value = value
        self.pcs_type = pcs_type


class OptimizationMissingExperiment(Exception):
    def __str__(self):
        return "You must create an experiment instance to perform optimization"
