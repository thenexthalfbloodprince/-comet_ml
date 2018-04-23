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
Author: Gideon Mendels

This module contains the components for console interaction like the std
wrapping

'''
import os
import sys

from contextlib import contextmanager


class FakeStd(object):
    """ A fake Std file-like that sends every line to a handler.
    """

    def __init__(self, handler, original):
        self.handler = handler
        self.original = original

    def write(self, line):
        '''
        Overrides the default IO write(). Writes to console + queue.
        :param line: String printed to stdout, probably with print()
        '''
        self.original.write(line)
        self.handler(line)

    def flush(self):
        self.original.flush()

    def isatty(self):
        return False

    def fileno(self):
        return self.original.fileno()


class BaseStdWrapper(object):

    def __init__(self, stdout=False, stdout_handler=None, stderr=False, stderr_handler=None):
        self.stdout = stdout
        self.stdout_handler = stdout_handler
        self.stderr = stderr
        self.stderr_handler = stderr_handler

        self._stdout = None
        self._stderr = None

        self._old_stdout = None
        self._old_stderr = None

    def __enter__(self):
        if self.stdout and self.stdout_handler:
            self._stdout = FakeStd(self.stdout_handler, sys.stdout)
            self._old_stdout = sys.stdout
            sys.stdout = self._stdout

        if self.stderr and self.stderr_handler:
            self._stderr = FakeStd(self.stderr_handler, sys.stderr)
            self._old_stderr = sys.stderr
            sys.stderr = self._stderr

    def __exit__(self, exception_type, exception_value, traceback):
        if self.stdout and self._old_stdout:
            sys.stdout = self._old_stdout
            self._old_stdout = None
            self._stdout = None

        if self.stderr and self._old_stderr:
            sys.stderr = self._old_stderr
            self._old_stderr = None
            self._stderr = None


try:
    # Try using a Wurlitzer based wrapper for Mac and Linux
    from wurlitzer import Wurlitzer

    class WurlitzerStdWrapper(Wurlitzer):
        """ A modified Wurltizer class that forward to the original FD and can
        call callbacks for captured streaming data.
        """

        def __init__(self, *args, **kwargs):
            self.stdout_handler = kwargs.pop("stdout_handler", None)
            self.stderr_handler = kwargs.pop("stderr_handler", None)
            super(StdWrapper, self).__init__(*args, **kwargs)
            self.finished = False

        def _handle_stdout(self, data):
            if self._stdout:
                os.write(self._save_fds["stdout"], data)

                if self.stdout_handler:
                    try:
                        self.stdout_handler(self._decode(data))
                    except Exception:
                        # Avoid raising exceptions
                        pass

        def _handle_stderr(self, data):
            if self._stderr:
                os.write(self._save_fds["stderr"], data)

                if self.stderr_handler:
                    try:
                        self.stderr_handler(self._decode(data))
                    except Exception:
                        # Avoid raising exceptions
                        pass

        def _finish_handle(self):
            self.finished = True

    StdWrapper = WurlitzerStdWrapper
except ImportError:
    StdWrapper = BaseStdWrapper


class StdLogger(object):
    def __init__(self, streamer):
        self.streamer = streamer
        self.experiment = None
        self.wrapper = StdWrapper(
            stdout=True,
            stdout_handler=self.stdout_handler,
            stderr=True,
            stderr_handler=self.stderr_handler)
        self.wrapper.__enter__()
        self.wrapped = True

    def clean(self):
        if self.wrapped is True:
            # Restore sys.std*
            self.wrapper.__exit__(None, None, None)
            self.wrapped = False

    def set_experiment(self, experiment):
        self.experiment = experiment

    def stdout_handler(self, data):
        self.handler(data, "stdout")

    def stderr_handler(self, data):
        self.handler(data, "stderr")

    def handler(self, data, std_name):
        if not self.experiment:
            return

        payload = self.experiment._create_message()

        if std_name == "stdout":
            payload.set_stdout(data)
        elif std_name == "stderr":
            payload.set_stderr(data)
        else:
            raise NotImplementedError()

        self.streamer.put_messge_in_q(payload)
