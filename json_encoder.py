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


convert_functions = []


try:
    import numpy

    def convert_numpy_array(value):
        try:
            return numpy.asscalar(value)

        except (ValueError, IndexError, AttributeError):
            return

    convert_functions.append(convert_numpy_array)
except ImportError:
    pass


class NestedEncoder(json.JSONEncoder):
    """
    A JSON Encoder that converts floats/decimals to strings and allows nested objects
    """

    def default(self, obj):

        # First convert the object
        obj = self.convert(obj)

        # Check if the object is convertible
        try:
            json.JSONEncoder().encode(obj)
            return obj

        except TypeError:
            pass

        # Custom conversion
        if obj.__class__.__name__ == "type":
            return str(obj)

        elif hasattr(obj, "repr_json"):
            return obj.repr_json()

        elif isinstance(obj, complex):
            return str(obj)

        else:
            try:
                return json.JSONEncoder.default(self, obj)

            except TypeError as e:
                return "%s not JSON serializable" % obj.__class__.__name__

    def floattostr(self, o, _inf=float("Inf"), _neginf=-float("-Inf"), nan_str="None"):
        if o != o:
            return nan_str

        else:
            return o.__repr__()

    def convert(self, obj):
        """
        Try converting the obj to something json-encodable
        """
        for converter in convert_functions:
            converted = converter(obj)

            if converted is not None:
                obj = converted

        return obj
