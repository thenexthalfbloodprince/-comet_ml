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

"""comet-ml"""
from __future__ import print_function

import atexit
import inspect
import os
import sys
import requests
import time
import os.path
import sys
import tempfile
import traceback
from contextlib import contextmanager
from copy import copy
import logging

import six
from pkg_resources import DistributionNotFound, get_distribution

from .comet import Message, Streamer, config, file_uploader, get_cmd_args_dict, generate_guid, ENV_OPTIMIZATION_ID, is_valid_guid, save_matplotlib_figure
from .config import get_config
from .connection import Reporting, RestServerConnection, server_address, visualization_upload_url, OptimizerConnection, WebSocketConnection, INITIAL_BEAT_DURATION
from .console import StdLogger
from .exceptions import NotParametrizedException, OptimizationMultipleParams, PCSParsingError, ValidationError, NoMoreSuggestionsAvailable, AuthenticationError
from .keras_logger import patch as keras_patch
from .monkey_patching import CometModuleFinder
from .optimization import Suggestion, parse_pcs
from .sklearn_logger import patch as sklearn_patch
from .tensorboard_logger import patch as tb_patch
from .utils import is_list_like
from ._logging import (
    setup,
    INTERNET_CONNECTION_ERROR,
    IPYTHON_NOTEBOOK_WARNING,
    METRIC_ARRAY_WARNING,
    EXPERIMENT_OPTIMIZER_API_KEY_MISMTACH_WARNING,
    PARSING_ERR_MSG,
    NOTEBOOK_MISSING_ID,
)
from ._notebook import _jupyter_server_extension_paths, _jupyter_nbextension_paths, load_jupyter_server_extension, get_notebook_id, in_notebook_environment

try:
    __version__ = get_distribution('comet_ml').version
except DistributionNotFound:
    __version__ = 'Please install comet with `pip install comet_ml`'

__author__ = 'Gideon<Gideon@comet.ml>'
__all__ = ['Experiment']

LOGGER = logging.getLogger(__name__)

# Activate the monkey patching
MODULE_FINDER = CometModuleFinder()
keras_patch(MODULE_FINDER)
sklearn_patch(MODULE_FINDER)
tb_patch(MODULE_FINDER)
MODULE_FINDER.start()

# Configure the logging
setup("INFO")


def start():
    '''
    If you are not using an Experiment in your first loaded Python file, you
    must import `comet_ml` and call `comet_ml.start` before any other imports
    to ensure that comet.ml is initialized correctly.
    '''
    pass


class Experiment(object):
    '''
    Experiment is a unit of measurable research that defines a single run with some data/parameters/code/results.

    Creating an Experiment object in your code will report a new experiment to your Comet.ml project. Your Experiment
    will automatically track and collect many things and will also allow you to manually report anything.

    You can create multiple objects in one script (such as when looping over multiple hyper parameters).

    '''

    def __init__(self,
                 api_key=None,
                 project_name=None,
                 team_name=None,
                 log_code=True,
                 auto_param_logging=True,
                 auto_metric_logging=True,
                 parse_args=True):
        """
        Creates a new experiment on the Comet.ml frontend.
        Args:
            api_key: Your API key obtained from comet.ml
            project_name: Optional. Send your experiment to a specific project. Otherwise will be sent to `Uncategorized Experiments`.
                             If project name does not already exists Comet.ml will create a new project.
            team_name: Optional. Attach an experiment to a project that belongs to this team.
            log_code: Default(True) - allows you to enable/disable code logging
            auto_param_logging: Default(True) - allows you to enable/disable hyper parameters logging
            auto_metric_logging: Default(True) - allows you to enable/disable metrics logging
            parse_args: Default(True) - allows you to enable/disable automatic parsing of CLI arguments
        """
        self.project_name = project_name
        self.team_name = team_name
        if api_key is None:
            self.api_key = os.getenv("COMET_API_KEY", None)
        else:
            self.api_key = api_key

        if self.api_key is None:
            raise ValueError(
                "Comet.ml requires an API key. Please provide as the "
                "first argument to Experiment(api_key) or as an environment"
                " variable named COMET_API_KEY ")

        self.params = {}
        self.metrics = {}
        self.others = {}

        # Base config
        self.config = get_config()

        self.log_code = log_code
        if in_notebook_environment():
            self.log_code = False
            notebook_id = get_notebook_id()

            if notebook_id is None:
                LOGGER.warning(NOTEBOOK_MISSING_ID)

        self.auto_param_logging = auto_param_logging
        self.auto_metric_logging = auto_metric_logging
        self.parse_args = parse_args

        # Generate a unique identifier for this experiment.
        self.id = self._get_experiment_key()
        self.alive = False
        self.is_github = False
        self.focus_link = None

        self.streamer = None
        self.logger = None
        self.run_id = None
        self.project_id = None

        # If set to False, wrappers should only run the original code
        self.disabled_monkey_patching = False

        # Experiment state
        self.context = None
        self.curr_step = None

        self.figure_counter = 0

        self.optimization_id = self._get_optimization_id_for_api_key(api_key)

        self.connection = RestServerConnection(
            self.api_key, self.id, self.optimization_id
        )

        # Cleanup old experiment before replace it
        if config.experiment is not None and config.experiment is not self:
            config.experiment._on_end(wait=False)

        config.experiment = self

        self._start()

        self.connection.report(event_name="experiment_created")

        LOGGER.info("Experiment is live on comet.ml %s\n", self._get_experiment_url())

    def _get_experiment_key(self):
        return generate_guid()

    def _on_start(self):
        """ Called when the Experiment is started
        """
        self._mark_as_started()

    def _mark_as_started(self):
        self.connection.update_experiment_status(
            self.run_id, self.project_id, self.alive
        )

    def _mark_as_ended(self):
        try:
            self.connection.update_experiment_status(
                self.run_id, self.project_id, False
            )
        except Exception as e:
            LOGGER.error("Failed to report experiment status", exc_info=True)

    def _on_end(self, wait=True):
        """ Called when the Experiment is replaced by another one or at the
        end of the script
        """
        self._mark_as_ended()

        if self.logger is not None:
            self.logger.clean()

        if self.streamer is not None:
            self.streamer.close()
            if wait is True:
                self.streamer.wait_for_finish()

    def _start(self):

        try:
            # This init the streamer and logger for the first time.
            # Would only be called once.
            if (self.streamer is None and self.logger is None):
                full_ws_url = self._authenticate()

                # Authentication failed
                if full_ws_url is None:
                    return

                self._initialize_streamer_logger(full_ws_url)

            atexit.register(self._on_end)

            if self.logger:
                self.logger.set_experiment(self)
            self.alive = True

        except Exception as e:
            tb = traceback.format_exc()
            LOGGER.error("Run will not be logged", exc_info=True)
            self.connection.report(event_name="experiment_creation_failed", err_msg=tb)
            return None

        try:
            if in_notebook_environment():
                self.set_notebook_name()
            else:
                self.filename = self._get_filename()
                self.set_filename(self.filename)
        except Exception as e:
            LOGGER.error("Failed to set run file name", exc_info=True)

        try:
            self.set_pip_packages()
        except Exception as e:
            LOGGER.error("Failed to set run pip packages", exc_info=True)

        try:
            if self.parse_args:
                self.set_cmd_args()
        except Exception as e:
            LOGGER.error("Failed to set run cmd args", exc_info=True)

        try:
            if self.log_code:
                self.set_code(self._get_source_code())
        except Exception as e:
            LOGGER.error("Failed to set run source code", exc_info=True)

        try:
            if self.log_code and self.is_github:
                self._upload_repository()
        except Exception as e:
            LOGGER.error("Failed to create git patch", exc_info=True)

        self._on_start()

    def _authenticate(self):
        """
        Do the handshake with the Backend to authenticate the api key and get
        various parameters and settings
        """
        # Get an id for this run
        try:
            self.run_id, full_ws_url, self.project_id, self.is_github, self.focus_link = self.connection.get_run_id(
                self.project_name, self.team_name
            )

            return full_ws_url

        except ValueError as e:
            LOGGER.error(INTERNET_CONNECTION_ERROR, exc_info=True)
            return

    def _initialize_streamer_logger(self, full_ws_url):
        """
        Initialize the streamer and logger with the websocket url received
        during the handshake.
        """
        # Initiate the streamer
        ws_connection = WebSocketConnection(full_ws_url)
        ws_connection.start()
        ws_connection.wait_for_connection()
        self.streamer = Streamer(ws_connection, INITIAL_BEAT_DURATION, self.connection)

        if in_notebook_environment():
            # Don't hijack sys.std* in notebook environment
            self.logger = None
            LOGGER.warning(IPYTHON_NOTEBOOK_WARNING)
        else:
            # Override default sys.stdout and feed to streamer.
            self.logger = StdLogger(self.streamer)
        # Start streamer thread.
        self.streamer.start()

    def _get_experiment_url(self):
        if self.focus_link:
            return self.focus_link + self.id
        return ""

    def _create_message(self):
        """
        Utility wrapper around the Message() constructor
        Returns: Message() object.

        """
        return Message(
            self.api_key,
            self.id,
            self.run_id,
            self.project_id,
            context=self.context,
            optimization_id=self.optimization_id,
            notebook_id=get_notebook_id(),
        )

    def get_metric(self, name):
        return self.metrics[name]

    def get_parameter(self, name):
        return self.params[name]

    def get_other(self, name):
        return self.others[name]

    def log_other(self, key, value):
        """
        Reports key,value to the `Other` tab on Comet.ml. Useful for reporting datasets attributes,
        datasets path, unique identifiers etc.

        See [`log_parameter`](#experimentlog_parameter)

        Args:
            key: Any type of key (str,int,float..)
            value: Any type of value (str,int,float..)

        Returns: None

        """
        if self.alive:
            message = self._create_message()
            message.set_log_other(key, value)
            self.streamer.put_messge_in_q(message)

        self.others[key] = value

    def log_html(self, html):
        """
        Reports any HTML blob to the `HTML` tab on Comet.ml. Useful for creating your own rich reports.
        The HTML will be rendered as an Iframe. Inline CSS/JS supported.
        Args:
            html: Any html string. for example:
            ```
            experiment.log_html('<a href="www.comet.ml"> I love Comet.ml </a>')
            ```

        Returns: None

        """
        if self.alive:
            message = self._create_message()
            message.set_html(html)
            self.streamer.put_messge_in_q(message)

    def set_step(self, step):
        """
        Sets the current step in the training process. In Deep Learning each step is after feeding a single batch
         into the network. This is used to generate correct plots on comet.ml. You can also pass the step directly when reporting [log_metric](#experimentlog_metric), and [log_parameter](#experimentlog_parameter).

        Args:
            step: Integer value

        Returns: None

        """

        if step is not None:
            self.curr_step = step

    def log_epoch_end(self, epoch_cnt, step=None):
        """
        Logs that the  epoch finished. required for progress bars.

        Args:
            epoch_cnt: integer

        Returns: None

        """
        self.set_step(step)

        if self.alive:
            message = self._create_message()
            message.set_param("curr_epoch", epoch_cnt, step=self.curr_step)
            self.streamer.put_messge_in_q(message)

    def log_metric(self, name, value, step=None):
        """
        Logs a general metric (i.e accuracy, f1).

        e.g.
        ```
        y_pred_train = model.predict(X_train)
        acc = compute_accuracy(y_pred_train, y_train)
        experiment.log_metric("accuracy", acc)
        ```

        See also [`log_multiple_metrics`](#experimentlog_multiple_metrics)


        Args:
            name: String - name of your metric
            value: Float/Integer/Boolean/String
            step: Optional. Used as the X axis when plotting on comet.ml

        Returns: None

        """

        self.set_step(step)

        if self.alive:
            message = self._create_message()

            if is_list_like(value):
                # Try to get the first value of the Array
                try:
                    if len(value) != 1:
                        raise TypeError()

                    if not isinstance(value[0], (six.integer_types, float, six.string_types, bool)):
                        raise TypeError()

                    value = value[0]

                except (TypeError):
                    LOGGER.warning(METRIC_ARRAY_WARNING, value)
                    value = str(value)

            message.set_metric(name, value, self.curr_step)
            self.streamer.put_messge_in_q(message)

        # save state.
        self.metrics[name] = value

    def log_parameter(self, name, value, step=None):
        """
        Logs a single hyperparameter. For additional values that are not hyper parameters it's encouraged to use [log_other](#experimentlog_other).

        See also [`log_multiple_params`](#experimentlog_multiple_params).


        Args:
            name: String - name of your parameter
            value: Float/Integer/Boolean/String/List
            step: Optional. Used as the X axis when plotting on comet.ml

        Returns: None

        """
        self.set_step(step)

        if self.alive:
            message = self._create_message()

            # Check if we have a list-like object or a string
            if is_list_like(value):
                message.set_params(name, value, self.curr_step)
            else:
                message.set_param(name, value, self.curr_step)

            self.streamer.put_messge_in_q(message)

        self.params[name] = value

    def log_figure(self, figure_name=None, figure=None):
        """
        Logs the global Pyplot figure or the passed one and upload its svg
        version to the backend.

        Args:
            figure_name: Optional. String - name of the figure
            figure: Optional. The figure you want to log. If not set, the
                    global pyplot figure will be logged and uploaded
        """
        try:
            filename = save_matplotlib_figure(figure)
        except Exception as e:
            # An error occured
            return

        # Pass additional url params
        figure_number = self.figure_counter
        self.figure_counter += 1
        url_params = {
            "step": self.curr_step,
            "figCounter": figure_number,
            "context": self.context,
            "runId": self.run_id
        }

        if figure_name is not None:
            url_params['figName'] = figure_name

        file_uploader.upload_file_process(self.project_id, self.id,
                                          filename,
                                          visualization_upload_url(),
                                          self.api_key,
                                          url_params)

    def log_current_epoch(self, value):
        if self.alive:
            message = self._create_message()
            message.set_metric('curr_epoch', value)
            self.streamer.put_messge_in_q(message)

    def log_multiple_params(self, dic, prefix=None, step=None):
        """
        Logs a dictionary of multiple parameters.
        See also [log_parameter](#experimentlog_parameter).

        e.g:
        ```
        experiment = Experiment(api_key="MY_API_KEY")
        params = {
            "batch_size":64,
            "layer1":"LSTM(128)",
            "layer2":"LSTM(128)",
            "MAX_LEN":200
        }

        experiment.log_multiple_params(params)
        ```
        """
        self.set_step(step)

        if self.alive:
            for k in sorted(dic):
                if prefix is not None:
                    self.log_parameter(prefix + "_" + str(k), dic[k], self.curr_step)
                else:
                    self.log_parameter(k, dic[k], self.curr_step)

    def log_multiple_metrics(self, dic, prefix=None, step=None):
        """
        Logs a key,value dictionary of metrics.
        See also [`log_metric`](#experimentlog_metric)
        """
        self.set_step(step)

        if self.alive:
            for k in sorted(dic):
                if prefix is not None:
                    self.log_metric(prefix + "_" + str(k), dic[k], self.curr_step)
                else:
                    self.log_metric(k, dic[k], step)

    def log_dataset_hash(self, data):
        try:
            import hashlib
            data_hash = hashlib.md5(str(data).encode('utf-8')).hexdigest()
            self.log_parameter("dataset_hash", data_hash[:12])
        except:
            LOGGER.warning("Failed to create dataset hash")

    def set_code(self, code):
        '''
        Sets the current experiment script's code. Should be called once per experiment.
        :param code: String
        '''
        if self.alive:
            message = self._create_message()
            message.set_code(code)
            self.streamer.put_messge_in_q(message)

    def set_model_graph(self, graph):
        '''
        Sets the current experiment computation graph.
        :param graph: JSON
        '''
        if self.alive:

            if type(graph).__name__ == "Graph":  # Tensorflow Graph Definition
                from google.protobuf import json_format
                graph_def = graph.as_graph_def()
                graph = json_format.MessageToJson(graph_def)

            message = self._create_message()
            message.set_graph(graph)
            self.streamer.put_messge_in_q(message)

    def set_filename(self, fname):
        if self.alive:
            message = self._create_message()
            message.set_filename(fname)
            self.streamer.put_messge_in_q(message)

    def set_notebook_name(self):
        self.set_filename("Notebook")

    def set_pip_packages(self):
        """
        Reads the installed pip packages using pip's CLI and reports them to server as a message.
        Returns: None

        """
        if self.alive:
            try:
                import pkg_resources
                installed_packages = [d for d in pkg_resources.working_set]
                installed_packages_list = sorted([
                    "%s==%s" % (i.key, i.version) for i in installed_packages
                ])
                message = self._create_message()
                message.set_installed_packages(installed_packages_list)
                self.streamer.put_messge_in_q(message)
            except:
                LOGGER.warning("Failing to collect the installed pip packages")

    def set_cmd_args(self):
        if self.alive:
            args = get_cmd_args_dict()
            if args is not None:
                for k, v in args.items():
                    self.log_parameter(k, v)

    def set_uploaded_extensions(self, extensions):
        """
        Override the default extensions that will be sent to the server.

        Args:
            extensions: list of extensions strings
        """
        self.config['uploaded_extensions'] = copy(extensions)

    # Context context-managers
    @contextmanager
    def train(self):
        """
        A context manager to mark the beginning and the end of the training
        phase. This allows you to provide a namespace for metrics/params.
        For example:

        ```
        experiment = Experiment(api_key="MY_API_KEY")
        with experiment.train():
            model.fit(x_train, y_train)
            accuracy = compute_accuracy(model.predict(x_train),y_train) # returns the train accuracy
            experiment.log_metric("accuracy",accuracy) # this will be logged as train accuracy based on the context.
        ```

        """
        # Save the old context and set the new one
        old_context = self.context
        self.context = "train"

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def validate(self):
        """
        A context manager to mark the beginning and the end of the validating
        phase. This allows you to provide a namespace for metrics/params.
        For example:

        ```
        with experiment.validate():
            pred = model.predict(x_validation)
            val_acc = compute_accuracy(pred, y_validation)
            experiment.log_metric("accuracy", val_acc) # this will be logged as validation accuracy based on the context.
        ```


        """
        # Save the old context and set the new one
        old_context = self.context
        self.context = "validate"

        yield self

        # Restore the old one
        self.context = old_context

    @contextmanager
    def test(self):
        """
        A context manager to mark the beginning and the end of the testing phase. This allows you to provide a namespace for metrics/params.
        For example:

        ```
        with experiment.test():
            pred = model.predict(x_test)
            test_acc = compute_accuracy(pred, y_test)
            experiment.log_metric("accuracy", test_acc) # this will be logged as test accuracy based on the context.
        ```

        """
        # Save the old context and set the new one
        old_context = self.context
        self.context = "test"

        yield self

        # Restore the old one
        self.context = old_context

    def get_keras_callback(self):
        """
        Returns an instance of Comet.ml's Keras callback. This callback is already added to your Keras `model.fit()` callbacks list automatically, to report model training metrics to Comet.ml.


        e.g:
        ```
        experiment = Experiment(api_key="MY_API_KEY")
        comet_callback = experiment.get_keras_callback()

        Returns: Comet.ml Keras callback.

        """
        if self.alive:
            from comet_ml.frameworks import KerasCallback
            return KerasCallback(
                self,
                log_params=self.auto_param_logging,
                log_metrics=self.auto_metric_logging)

        from comet_ml.frameworks import EmptyKerasCallback
        return EmptyKerasCallback()

    def disable_mp(self):
        ''' Disabling the auto-collection of metrics and monkey-patching of
        the Machine Learning frameworks.
        '''
        self.disabled_monkey_patching = True

    def _get_source_code(self):
        '''
        Inspects the stack to detect calling script. Reads source code from disk and logs it.
        '''

        for frame in inspect.stack(context=1):
            module = inspect.getmodule(frame[0])
            if "comet_ml" != module.__name__:
                filename = module.__file__.rstrip('cd')
                with open(filename) as f:
                    return f.read()

        LOGGER.warning("Failed to find source code module")

    def _get_filename(self):

        if sys.argv:
            pathname = os.path.dirname(sys.argv[0])
            abs_path = os.path.abspath(pathname)
            filename = os.path.basename(sys.argv[0])
            full_path = os.path.join(abs_path, filename)
            return full_path

        return None

    def _upload_repository(self):
        file_uploader.upload_repo_start_process(
            self.project_id,
            self.id,
            self.filename,
            server_address + "logger/repoRoot",
            server_address + "logger/uploadFiles",
            self.api_key,
            config=self.config)

    def _get_optimization_id_for_api_key(self, api_key):
        if config.optimizer is not None:
            if config.optimizer.api_key == api_key:
                return config.optimizer.id
            else:
                LOGGER.warning(EXPERIMENT_OPTIMIZER_API_KEY_MISMTACH_WARNING)
        return None


class ExistingExperiment(Experiment):
    """
    Existing Experiment allows you to report information to an experiment that already exists on comet.ml
    and is not currently running. This is usfeul in cases where your training and testing happens on different scripts.

    For example:

    train.py:
    ```
    exp = Experiment(api_key="my-key")
    score = train_model()
    exp.log_metric("train accuarcy", score)
    ```

    Now obtain the experiment key from comet.ml. If it's not visible on your experiment table
    you can click `Customize` and add it as a column.


    test.py:
    ```
    exp = ExistingExperiment(api_key="my-key", previous_experiment = "your experiment key from comet.ml")
    score = test_model()
    exp.log_metric("test accuarcy", score)
    ```

    """

    def __init__(self, api_key, previous_experiment):
        # Validate the previous experiment id
        if not is_valid_guid(previous_experiment):
            raise ValueError("Invalid experiment key: %s" % previous_experiment)

        self.previous_experiment = previous_experiment
        super(ExistingExperiment, self).__init__(api_key)

    def _get_experiment_key(self):
        return self.previous_experiment

    def _authenticate(self):
        """
        Do the handshake with the Backend to authenticate the api key and get
        various parameters and settings
        """
        # Get an id for this run
        try:
            self.run_id, full_ws_url, self.project_id, self.is_github, self.focus_link = self.connection.get_old_run_id(
                self.previous_experiment)

            return full_ws_url

        except ValueError as e:
            LOGGER.error(INTERNET_CONNECTION_ERROR, exc_info=True)
            return

    def _create_message(self):
        # Create a default context to distinguish between script runs
        if self.context is None:
            new_context = generate_guid()
            self.context = new_context

        return super(ExistingExperiment, self)._create_message()


class Optimizer(object):
    '''
    An Optimizer is the object that you can use to dynamically optimize your hyper-parameters on the cloud with hassle.

    You can use it this way:

    ```
    optimizer = Optimizer("API_KEY")

    # Declare your hyper-parameters in the PCS format
    params = """
    x integer [1, 10] [10]
    y real [1, 10] [1.0]
    """

    optimizer.set_params(params)

    # get_suggestion will raise when no new suggestion is available
    while True:
        # Get a suggestion
        suggestion = optimizer.get_suggestion()

        # Create a new experiment associated with the Optimizer
        experiment = Experiment("API_KEY")

        # Test the model
        score = fit(suggestion["x"])

        # Report the score back
        suggestion.report_score("accuracy",score)
    ```

    '''

    def __init__(self, api_key):
        """
        Creates an Optimizer that you can use to get hyper-parameter
        suggestions.
        Args:
            api_key: Your API key obtained from comet.ml
        """
        self.api_key = api_key
        self.id = generate_guid()
        self.headers = {"OPTIMIZATION-ID": self.id}
        self.types = {}
        self.params_set = False
        self.connection = OptimizerConnection(self.headers)

        self._authenticate()

        # Save the optimization id in global config so Experiment can use it.
        config.optimizer = self

    def _authenticate(self):
        # Authenticate to the hyper-parameter service
        try:
            self.connection.authenticate(self.api_key, self.id)
        except requests.exceptions.HTTPError as e:
            response = e.response
            # If their was a validation error, print it
            if response.status_code == 400:
                try:
                    data = response.json()
                    error = data.get("msg")
                    if error:
                        six.raise_from(AuthenticationError(error), None)
                except ValueError:
                    pass

            raise

    def set_params_file(self, file_path):
        """
        Declare your hyper-parameter using a file in the PCS format.
        Args:
            file_path: The PCS file path
        """
        with open(file_path, 'r') as pcs_file:
            pcs_content = pcs_file.read()
        self._send_params(pcs_content)

    def set_params(self, pcs_content):
        """
        Declare your hyper-parameter using a string in the PCS format.
        Args:
            pcs_content: A string in the PCS format. Leading spaces are not
                         significant.
        """
        self._send_params(pcs_content)

    def _send_params(self, pcs_content):
        """ Send the pcs content to the hyper-parameter optimization API and
        try to parse it to extract type information
        """
        # Check that we can send params only once
        if self.params_set is True:
            raise OptimizationMultipleParams()

        # Parse the pcs_content
        try:
            self.types = parse_pcs(pcs_content)
        except PCSParsingError:
            # Fallback on no types and print warning
            LOGGER.warning(PARSING_ERR_MSG)
            self.types = {}

        # Send the requests
        try:
            self.connection.create(pcs_content)

            self.params_set = True
        except requests.exceptions.HTTPError as e:
            response = e.response
            # If their was a validation error, print it
            if response.status_code == 400:
                try:
                    data = response.json()
                    error = data.get("error")
                    if error:
                        six.raise_from(ValidationError(error), None) 
                except ValueError:
                    pass

            raise

    def get_suggestion(self):
        """ Return a new [Suggestion](Suggestion/#Suggestion) object
        containing values for each (TODO: Check with conditions) declare
        hyper-parameter.
        """

        if self.params_set is False:
            raise NotParametrizedException()

        try:
            while True:
                response = self.connection.get_suggestion()

                data = response.json()
                suggestion = data.get("suggestion")
                if suggestion:
                    return Suggestion(suggestion, self, self.types)

                else:
                    if data.get("terminated"):
                        raise NoMoreSuggestionsAvailable()

                    time.sleep(1)
        except requests.exceptions.HTTPError as err:
            exc_info = sys.exc_info()
            if err.response.status_code != 404:
                raise

            try:
                data = err.response.json()
                if data.get("error") == "Missing instance":
                    err_msg = "No optimization session has been found, did you called set_params or set_params_file before calling get_suggestion?"
                    six.raise_from(Exception(err_msg), None)

            except ValueError:
                raise err

    def _report_score(self, run_id, score):
        # Maximize the score
        score = -1 * score

        self.connection.report_score(run_id, score)
