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

""" The import hook which monkey patch modules
"""

import imp
import sys

from ._notebook import in_notebook_environment


class CustomFileLoader(object):
    """ A Python 3 loader that use a SourceFileLoader to exec an imported
    module and patch it with CometModuleFinder
    """

    def __init__(self, loader, fullname, finder):
        self.loader = loader
        self.fullname = fullname
        self.finder = finder

    def exec_module(self, module):
        # Execute the module source code to define all the objects
        self.loader.exec_module(module)

        return self.finder._patch(module, self.fullname)

    def create_module(self, spec):
        """ Mandatory in Python 3.6 as we define the exec_module method
        """
        return None


class CometModuleFinder(object):
    def __init__(self):
        self.patcher_functions = {}

        if sys.version_info[0] >= 3:
            from importlib.machinery import PathFinder
            self.pathfinder = PathFinder()

    def register(self, module_name, object_name, patcher_function):
        module_patchers = self.patcher_functions.setdefault(module_name, {})
        module_patchers[object_name] = patcher_function

    def start(self):
        if self not in sys.meta_path and not in_notebook_environment():
            sys.meta_path.insert(0, self)

    def find_module(self, fullname, path=None):
        """ Python 2 import hook
        """
        if fullname not in self.patcher_functions:
            return

        return self

    def load_module(self, fullname):
        """ Python 2 import hook
        """
        module = self._get_module(fullname)
        return self._patch(module, fullname)

    def find_spec(self, fullname, path=None, target=None):
        """ Python 3 import hook
        """
        if fullname not in self.patcher_functions:
            return

        from importlib.machinery import ModuleSpec, SourceFileLoader

        spec = self.pathfinder.find_spec(fullname, path, target)
        loader = SourceFileLoader(fullname, spec.origin)
        return ModuleSpec(fullname, CustomFileLoader(loader, fullname, self))

    def _get_module(self, fullname):
        splitted_name = fullname.split('.')
        parent = '.'.join(splitted_name[:-1])

        if fullname in sys.modules:
            return sys.modules[fullname]
        elif parent in sys.modules:
            parent = sys.modules[parent]
            module_path = imp.find_module(splitted_name[-1], parent.__path__)
            return imp.load_module(fullname, *module_path)
        else:
            module_path = imp.find_module(fullname)
            return imp.load_module(fullname, *module_path)

    def _patch(self, module, fullname):
        objects_to_patch = self.patcher_functions.get(fullname, {})

        for object_name, patcher_function in objects_to_patch.items():
            object_path = object_name.split('.')

            original = self._get_object(module, object_path)

            if original is None:
                # TODO: Send back the error?
                continue

            new_object = patcher_function(original)
            self._set_object(module, object_path, new_object)

        return module

    def _get_object(self, module, object_path):
        current_object = module

        for part in object_path:
            try:
                current_object = getattr(current_object, part)
            except AttributeError:
                return None

        return current_object

    def _set_object(self, module, object_path, new_object):
        object_to_patch = self._get_object(module, object_path[:-1])
        setattr(object_to_patch, object_path[-1], new_object)
