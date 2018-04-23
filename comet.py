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
Author: Gideon Mendels

This module contains the main components of comet.ml client side

'''
from __future__ import print_function

import json
import os
import re
import sys
import tempfile
import threading
import time
import uuid
import logging
from math import isnan

from comet_ml import config, file_uploader
from comet_ml.json_encoder import NestedEncoder
from six.moves.queue import Queue, Empty

DEBUG = False
ENV_OPTIMIZATION_ID = "COMETML_OPTIMIZATION_ID_{api_key}"
LOGGER = logging.getLogger(__name__)


class CloseMessage(object):
    """ A special messag indicating Streamer to ends and exit
    """
    pass


class Streamer(threading.Thread):
    """
    This class extends threading.Thread and provides a simple concurrent queue
    and an async service that pulls data from the queue and sends it to the server.
    """

    def __init__(self, ws_connection, beat_duration, connection):
        threading.Thread.__init__(self)
        self.daemon = True
        self.messages = Queue()
        self.counter = 0
        self.name = "Streamer(%r)" % (ws_connection)
        self.ws_connection = ws_connection
        self.closed = False
        self.beat_duration = beat_duration / 1000. # We get milliseconds
        self.last_beat = time.time()
        self.connection = connection

    def put_messge_in_q(self, message):
        '''
        Puts a message in the queue
        :param message: Some kind of payload, type agnostic
        '''
        if message is not None:
            self.messages.put(message)

    def close(self):
        '''
        Puts a None in the queue which leads to closing it.
        '''
        if self.closed is True:
            return

        # Send a message to close
        self.put_messge_in_q(CloseMessage())

        self.closed = True

    def run(self):
        '''
        Continuously pulls messages from the queue and sends them to the server.
        '''
        self.ws_connection.wait_for_connection()

        while True:
            out = self._loop()

            # Exit the infinite loop
            if isinstance(out, CloseMessage):
                break

        # Close the websocket connection
        self.ws_connection.close()

        return

    def _loop(self):
        """
        A single loop of running
        """
        try:
            if self.ws_connection is not None and self.ws_connection.is_connected():
                messages = self.getn(1)

                if messages and isinstance(messages[0], CloseMessage):
                    return messages[0]

                if messages is not None:
                    self.ws_connection.send(messages)

                self._check_heartbeat()

        except Exception as e:
            LOGGER.error("Streaming error", exc_info=True)

    def _check_heartbeat(self):
        """
        Check if we should send an heartbeat
        """
        next_beat = self.last_beat + self.beat_duration
        now = time.time()
        if next_beat < now:
            new_beat_duration = self.connection.heartbeat()
            self.beat_duration = new_beat_duration / 1000. # We get milliseconds
            self.last_beat = time.time()

    def getn(self, n):
        """
        Pops n messages from the queue.
        Args:
            n: Number of messages to pull from queue

        Returns: n messages

        """
        try:
            msg = self.messages.get(timeout=self.beat_duration)  # block until at least 1
        except Empty:
            return None

        if isinstance(msg, CloseMessage):
            return [msg]

        self.counter += 1
        msg.set_offset(self.counter)
        result = [msg]
        try:
            while len(result) < n:
                another_msg = self.messages.get(
                    block=False)  # dont block if no more messages
                self.counter += 1
                another_msg.set_offset(self.counter)
                result.append(another_msg)
        except Exception:
            pass
        return result

    def wait_for_finish(self):
        """ Blocks the experiment from exiting until all data was sent to server OR 30 seconds passed."""

        msg = "Uploading stats to Comet before program termination (may take several seconds)"
        LOGGER.info(msg)

        # Wait maximum 2 minutes
        self._wait_for_empty(30)

        if not self.messages.empty():
            msg = "Still uploading stats to Comet before program termination (may take several seconds)"
            LOGGER.info(msg)
            self._wait_for_empty(30)

            if not self.messages.empty():
                self._wait_for_empty(60, verbose=True)

        if not self.messages.empty():
            remaining = self.messages.qsize()
            LOGGER.error(
                "Comet failed to send all the data back (%s messages)", remaining
            )

        if config.experiment is not None:
            if config.experiment.alive:
                LOGGER.info(
                    "Experiment is live on comet.ml: %s",
                    config.experiment._get_experiment_url(),
                )
            else:
                LOGGER.warning("Failed to log run in comet.ml")
        else:
            LOGGER.warning("Failed to log run in comet.ml")

    def _wait_for_empty(self, timeout, verbose=False):
        """ Wait up to TIMEOUT seconds for the messages queue to be empty
        """
        end_time = time.time() + timeout

        while not self.messages.empty() and time.time() < end_time:
            if verbose is True:
                LOGGER.info("%d messages remaining to upload", self.messages.qsize())
            time.sleep(5)

    def _report_upload_not_finished(self):
        """ Called if the queue is not empty during exit
        """
        remaining = self.messages.qsize()

        if remaining == 0:
            return

        err_msg = "Comet failed to send all the data back %d"

        # Display the message
        LOGGER.error(err_msg, remaining)

        self.connection.report(
            event_name="on_exit_didnt_finish_upload_sdk", err_msg=(err_msg  % remaining)
        )

INFINITY = float('inf')


def fix_special_floats(value, _inf=INFINITY, _neginf=-INFINITY):
    """ Fix out of bounds floats (like infinity and -infinity) and Not A
    Number.
    Returns either a fixed value that could be JSON encoded or the original
    value.
    """

    try:
        # Check if the value is Nan, equivalent of math.isnan
        if isnan(value):
            return "NaN"

        elif value == _inf:
            return "Infinity"

        elif value == _neginf:
            return "-Infinity"

    except TypeError:
        # Ndarray cannot be compared
        return value

    return value


class Message(object):
    """
    A bean used to send messages to the server over websockets.
    """

    def __init__(
        self,
        api_key,
        experiment_key,
        run_id,
        project_id,
        context=None,
        optimization_id=None,
        notebook_id=None
    ):
        self.apiKey = api_key
        self.experimentKey = experiment_key
        self.runId = run_id
        self.projectId = project_id
        self.local_timestamp = int(time.time() * 1000)

        # The following attributes are optional
        self.metric = None
        self.param = None
        self.params = None
        self.graph = None
        self.code = None
        self.stdout = None
        self.stderr = None
        self.offset = None
        self.fileName = None
        self.html = None
        self.installed_packages = None
        self.log_other = None

        self.context = context
        self.optimization_id = optimization_id
        self.notebook_id = notebook_id

    def set_log_other(self, key, value):
        self.log_other = {"key": key, "val": value}

    def set_installed_packages(self, val):
        self.installed_packages = val

    def set_offset(self, val):
        self.offset = val

    def set_metric(self, name, value, step=None):
        safe_value = fix_special_floats(value)
        self.metric = {
            "metricName": name,
            "metricValue": safe_value,
            "step": step
        }

    def set_html(self, value):
        self.html = value

    def set_param(self, name, value, step=None):
        safe_value = fix_special_floats(value)
        self.param = {
            "paramName": name,
            "paramValue": safe_value,
            "step": step
        }

    def set_params(self, name, values, step=None):
        safe_values = list(map(fix_special_floats, values))
        self.params = {
            "paramName": name,
            "paramValue": safe_values,
            "step": step
        }

    def set_graph(self, graph):
        self.graph = graph

    def set_code(self, code):
        self.code = code

    def set_stdout(self, line):
        self.stdout = line
        self.stderr = False

    def set_stderr(self, line):
        self.stdout = line
        self.stderr = True

    def set_filename(self, fname):
        self.fileName = fname

    def to_json(self):
        json_re = json.dumps(
            self.repr_json(), sort_keys=True, indent=4, cls=NestedEncoder)
        return json_re

    def repr_json(self):
        return self.__dict__

    def __repr__(self):
        filtered_dict = [(key, value) for key, value in self.__dict__.items()
                         if value]
        string = ", ".join("%r=%r" % item for item in filtered_dict)
        return "Message(%s)" % string

    def __str__(self):
        return self.to_json()

    def __len__(self):
        return len(self.to_json())


def get_cmd_args_dict():
    if len(sys.argv) > 1:
        try:
            return parse_cmd_args(sys.argv[1:])

        except ValueError as e:
            LOGGER.warning(
                "Failed to parse argv values. Falling back to naive parsing."
            )
            return parse_cmd_args_naive(sys.argv[1:])


def parse_cmd_args_naive(to_parse):
    vals = {}
    if len(to_parse) > 1:
        for i, arg in enumerate(to_parse):
            vals["run_arg_%s" % i] = str(arg)

    return vals


def parse_cmd_args(argv_vals):
    """
    Parses the value of argv[1:] to a dictionary of param,value. Expects params name to start with a - or --
    and value to follow. If no value follows that param is considered to be a boolean param set to true.(e.g --test)
    Args:
        argv_vals: The sys.argv[] list without the first index (script name). Basically sys.argv[1:]

    Returns: Dictionary of param_names, param_values

    """

    def guess_type(s):
        import ast
        try:
            value = ast.literal_eval(s)
            return value

        except (ValueError, SyntaxError):
            return str(s)

    results = {}

    current_key = None
    for word in argv_vals:
        word = word.strip()
        prefix = 0

        if word[0] == '-':
            prefix = 1
            if len(word) > 1 and word[1] == '-':
                prefix = 2

            if current_key is not None:
                # if we found a new key but haven't found a value to the previous
                # key it must have been a boolean argument.
                results[current_key] = True

            current_key = word[prefix:]

        else:
            word = word.strip()
            if current_key is None:
                # we failed to parse the string. We think this is a value but we don't know what's the key.
                # fallback to naive parsing.
                raise ValueError("Failed to parse argv arguments")
            else:
                word = guess_type(word)
                results[current_key] = word
                current_key = None

    if current_key is not None:
        # last key was a boolean
        results[current_key] = True

    return results


def save_matplotlib_figure(figure=None):
    """ Try saving either the current global pyplot figure or the given one
    and return None in case of error.
    """
    # Get the right figure to upload
    if figure is None:
        import matplotlib.pyplot

        # Get current global figure
        figure = matplotlib.pyplot.gcf()

    # Check if the figure is empty or not
    figure_lines = figure.gca().lines
    if len(figure_lines) == 0:
        # TODO DISPLAY BETTER ERROR MSG
        msg = "Refuse to upload empty figure, please call log_figure before calling show"
        LOGGER.warning(msg)
        raise TypeError(msg)

    # Save the file to a tempfile but don't delete it, the file uploader
    # process will take care of it
    tmpfile = tempfile.NamedTemporaryFile(suffix='.svg', delete=False)
    figure.savefig(tmpfile, format="svg")

    return tmpfile.name


def generate_guid():
    """ Generate a GUID
    """
    return str(uuid.uuid4()).replace("-", "")


def is_valid_guid(guid):
    """ Validate a GUID
    """
    # The compiled regex is cached
    guid_regex = re.compile(r"[0-9a-f]{32}")
    return guid_regex.match(guid)
