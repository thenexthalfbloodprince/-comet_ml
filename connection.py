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

from __future__ import print_function

import json
import os
import sys
import threading
import time
import traceback
import logging

import requests

import comet_ml
import websocket
from comet_ml import config
from comet_ml.json_encoder import NestedEncoder

server_address = os.environ.get('COMET_URL_OVERRIDE',
                                'https://www.comet.ml/clientlib/')
optimization_address = os.environ.get('COMET_OPTIMIZATION_OVERRIDE',
                                      "https://optimizer.comet.ml/")
TIMEOUT = 10

BACKEND_SESSION = None
OPTIMIZER_SESSION = None
INITIAL_BEAT_DURATION = 10000 # 10 second

LOGGER = logging.getLogger(__name__)


def _comet_version():
    try:
        version_num = comet_ml.__version__
    except NameError:
        version_num = None

    return version_num


def get_backend_session():
    global BACKEND_SESSION
    if BACKEND_SESSION is None:
        BACKEND_SESSION = requests.Session()

    return requests.Session()


def get_optimizer_session():
    global OPTIMIZER_SESSION
    if OPTIMIZER_SESSION is None:
        OPTIMIZER_SESSION = requests.Session()

    return requests.Session()


class RestServerConnection(object):
    """
    A static class that handles the connection with the server.
    """

    def __init__(self, api_key, experiment_id, optimization_id):
        self.api_key = api_key
        self.experiment_id = experiment_id
        self.optimization_id = optimization_id

        # Set once get_run_id is called
        self.run_id = None
        self.project_id = None

        self.session = get_backend_session()

    def heartbeat(self):
        """ Inform the backend that we are still alive
        """
        LOGGER.debug("Doing an heartbeat")
        return self.update_experiment_status(self.run_id, self.project_id, True)

    def update_experiment_status(self, run_id, project_id, is_alive):
        endpoint_url = server_address + "status-report/update"
        headers = {'Content-Type': 'application/json;charset=utf-8'}

        payload = {
            "apiKey": self.api_key,
            "runId": run_id,
            "experimentKey": self.experiment_id,
            "projectId": project_id,
            "optimizationId": self.optimization_id,
            "is_alive": is_alive,
            "local_timestamp": int(time.time() * 1000),
        }

        session = get_backend_session()
        r = session.post(
            url=endpoint_url, data=json.dumps(payload), headers=headers, timeout=TIMEOUT
        )

        if r.status_code != 200:
            raise ValueError(r.content)

        data = r.json()
        beat_duration = data.get("is_alive_beat_duration_millis")

        if beat_duration is None:
            raise ValueError("Missing heart-beat duration")

        return beat_duration

    def get_run_id(self, project_name, team_name):
        """
        Gets a new run id from the server.
        :param api_key: user's API key
        :return: run_id - String
        """
        endpoint_url = server_address + "logger/add/run"
        headers = {'Content-Type': 'application/json;charset=utf-8'}

        payload = {
            "apiKey": self.api_key,
            "local_timestamp": int(time.time() * 1000),
            "projectName": project_name,
            "teamName": team_name,
            "libVersion": _comet_version(),
        }
        session = get_backend_session()

        LOGGER.debug("Get run id URL: %s", endpoint_url)
        r = session.post(
            url=endpoint_url, data=json.dumps(payload), headers=headers, timeout=TIMEOUT
        )

        if r.status_code != 200:
            raise ValueError(r.content)

        res_body = json.loads(r.content.decode('utf-8'))

        return self._parse_run_id_res_body(res_body)

    def get_old_run_id(self, previous_experiment):
        """
        Gets a run id from an existing experiment.
        :param api_key: user's API key
        :return: run_id - String
        """
        endpoint_url = server_address + "logger/get/run"
        headers = {'Content-Type': 'application/json;charset=utf-8'}

        payload = {
            "apiKey": self.api_key,
            "local_timestamp": int(time.time() * 1000),
            "previousExperiment": previous_experiment,
            "libVersion": _comet_version(),
        }
        session = get_backend_session()
        LOGGER.debug("Get old run id URL: %s", endpoint_url)
        r = session.post(
            url=endpoint_url, data=json.dumps(payload), headers=headers, timeout=TIMEOUT
        )

        if r.status_code != 200:
            raise ValueError(r.content)

        res_body = json.loads(r.content.decode('utf-8'))

        return self._parse_run_id_res_body(res_body)

    def _parse_run_id_res_body(self, res_body):
        run_id_server = res_body["runId"]
        ws_full_url = res_body["ws_full_url"]

        project_id = res_body.get("project_id", None)

        is_github = bool(res_body.get("githubEnabled", False))

        focus_link = res_body.get("focusUrl", None)

        if "msg" in res_body:
            LOGGER.info(res_body["msg"])

        # Save run_id and project_id around
        self.run_id = run_id_server
        self.project_id = project_id

        return run_id_server, ws_full_url, project_id, is_github, focus_link

    def report(self, event_name=None,
               err_msg=None):

        try:
            if event_name is not None:
                endpoint_url = notify_url()
                headers = {'Content-Type': 'application/json;charset=utf-8'}

                payload = {
                    "event_name": event_name,
                    "api_key": self.api_key,
                    "run_id": self.run_id,
                    "experiment_key": self.experiment_id,
                    "project_id": self.project_id,
                    "err_msg": err_msg,
                    "timestamp": int(time.time() * 1000),
                }

                LOGGER.debug("Report notify URL: %s", endpoint_url)

                r = requests.post(
                    url=endpoint_url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=TIMEOUT / 2,
                )

        except Exception as e:
            LOGGER.error("Error reporting", exc_info=True)
            pass


class Reporting(object):
    def __init__(self):
        pass

    @staticmethod
    def report(event_name=None,
               api_key=None,
               run_id=None,
               experiment_key=None,
               project_id=None,
               err_msg=None,
               is_alive=None):

        try:
            if event_name is not None:
                endpoint_url = notify_url()
                headers = {'Content-Type': 'application/json;charset=utf-8'}

                payload = {
                    "event_name": event_name,
                    "api_key": api_key,
                    "run_id": run_id,
                    "experiment_key": experiment_key,
                    "project_id": project_id,
                    "err_msg": err_msg,
                    "timestamp": int(time.time() * 1000),
                }

                r = requests.post(
                    url=endpoint_url,
                    data=json.dumps(payload),
                    headers=headers,
                    timeout=TIMEOUT / 2,
                )

        except Exception as e:
            pass


def notebook_source_upload(content_hash, json_model, api_key, notebook_path):
    session = get_backend_session()

    payload = {"apiKey": api_key, "notebookPath": notebook_path, "code": json_model}

    endpoint_url = "%sjupyter-notebook/source/add?notebookId=%s" % (
        server_address, content_hash
    )
    headers = {"Content-Type": "application/json;charset=utf-8"}

    response = session.post(
        url=endpoint_url, data=json.dumps(payload), headers=headers, timeout=TIMEOUT
    )
    response.raise_for_status()

    return response


class WebSocketConnection(threading.Thread):
    """
    Handles the ongoing connection to the server via Web Sockets.
    """

    def __init__(self, ws_server_address):
        threading.Thread.__init__(self)
        self.priority = 0.2
        self.daemon = True
        self.name = "WebSocketConnection(%s)" % (ws_server_address)

        if config.DEBUG:
            websocket.enableTrace(True)

        self.address = ws_server_address
        self.ws = self.connect_ws(self.address)

    def is_connected(self):
        if self.ws.sock is not None:
            return self.ws.sock.connected

        return False

    def connect_ws(self, ws_server_address):
        ws = websocket.WebSocketApp(
            ws_server_address,
            on_message=WebSocketConnection.on_message,
            on_error=WebSocketConnection.on_error,
            on_close=WebSocketConnection.on_close)
        ws.on_open = WebSocketConnection.on_open
        return ws

    def run(self):
        while True:
            try:
                self.ws.run_forever()
                break
            except Exception as e:
                if sys is not None and config.DEBUG:
                    traceback.print_exc(file=sys.stderr)
                # Avoid hammering the backend
                time.sleep(0.5)

    def send(self, messages):
        """ Encode the messages into JSON and send them on the websocket
        connection
        """
        data = self._encode(messages)
        self._send(data)

    def close(self):
        self.ws.close()

    def _encode(self, messages):
        """ Encode a list of messages into JSON
        """
        messages_arr = []
        for message in messages:
            payload = {}
            # make sure connection is actually alive
            if message.stdout is not None:
                payload["stdout"] = message
            else:
                payload["log_data"] = message

            messages_arr.append(payload)

        data = json.dumps(messages_arr, cls=NestedEncoder, allow_nan=False)
        return data

    def _send(self, data):
        if self.ws.sock:
            self.ws.send(data)
            return

        else:
            self.wait_for_connection()

    def wait_for_connection(self, num_of_tries=10):
        """
        waits for the server connection
        Args:
            num_of_tries: number of times to try connecting before giving up

        Returns: True if succeeded to connect.

        """
        if not self.is_connected():
            counter = 0

            while not self.is_connected() and counter < num_of_tries:
                time.sleep(1)
                counter += 1

            if not self.is_connected():
                raise ValueError(
                    "Could not connect to server after multiple tries. ")

        return True

    @staticmethod
    def on_open(ws):
        LOGGER.debug("WS Socket connection open")

    @staticmethod
    def on_message(ws, message):
        LOGGER.debug("WS msg: %s", message)

    @staticmethod
    def on_error(ws, error):
        error_type_str = type(error).__name__
        ignores = [
            'WebSocketBadStatusException', 'error',
            'WebSocketConnectionClosedException', 'ConnectionRefusedError',
            'BrokenPipeError'
        ]

        if error_type_str in ignores:
            return

        LOGGER.debug(error)

    @staticmethod
    def on_close(ws):
        LOGGER.debug("WS connection closed")


def notify_url():
    return server_address + "notify/event"


def visualization_upload_url():
    """ Return the URL to upload visualizations
    """
    return server_address + "visualizations/upload"


class OptimizerConnection(object):

    def __init__(self, headers):
        self.session = get_optimizer_session()
        self.headers = headers

    def authenticate(self, api_key, optimization_id):
        data = {
            "optimizationId": optimization_id,
            "apiKey": api_key,
            "libVersion": _comet_version(),
        }

        response = self.session.post(
            optimization_address + "authenticate", json=data, headers=self.headers
        )
        response.raise_for_status()
        return response

    def create(self, pcs_content):
        files = {"file": pcs_content}
        response = self.session.post(
            optimization_address + "create", files=files, headers=self.headers
        )
        response.raise_for_status()
        return response

    def get_suggestion(self):
        response = self.session.get(
            optimization_address + "suggestion/", headers=self.headers
        )
        response.raise_for_status()
        return response

    def report_score(self, run_id, score):
        response = self.session.post(
            optimization_address + "value/%s" % run_id,
            json={"value": score},
            headers=self.headers,
        )
        response.raise_for_status()
        return response
