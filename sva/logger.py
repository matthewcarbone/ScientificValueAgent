"""BSD 3-Clause License

Copyright (c) 2023, Matthew R. Carbone, Stepan Fomichev & John Sous
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import sys
from contextlib import contextmanager
from copy import copy
from functools import wraps
from pathlib import Path
from warnings import catch_warnings, warn

from loguru import logger

NO_DEBUG_LEVELS = ["INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
LOGGER_STATE = {"levels": NO_DEBUG_LEVELS, "python_standard_warnings": False}


def generic_filter(names):
    if names == "all":
        return None

    def f(record):
        return record["level"].name in names

    return f


format_mapping = {
    "DEBUG": "[<lvl>D</>] [{name}:{function}:{line}] <lvl>{message}</>",
    "INFO": "[<lvl>I</>] <lvl>{message}</>",
    "SUCCESS": "[<lvl>S</>] <lvl>{message}</>",
    "WARNING": "[<lvl>W</>] <lvl>{message}</>",
    "ERROR": "[<lvl>E</>] <lvl>{message}</>",
    "CRITICAL": "[<lvl>C</>] <lvl>{message}</>",
}


def configure_loggers(
    levels=LOGGER_STATE["levels"],
    log_directory=None,
    python_standard_warnings=LOGGER_STATE["python_standard_warnings"],
):
    """Configures the ``loguru`` loggers. Note that the loggers are initialized
    using the default values by default.

    Parameters
    ----------
    levels : list, optional
        Description
    python_standard_warnings : bool, optional
        Raises dummy warnings on ``logger.warning`` and ``logger.error``.
    """

    logger.remove(None)  # Remove ALL handlers

    for level in levels:
        logger.add(
            sys.stdout if level in ["DEBUG", "INFO", "SUCCESS"] else sys.stderr,
            colorize=True,
            filter=generic_filter([level]),
            format=format_mapping[level],
        )
        if log_directory is None:
            continue
        p = Path(log_directory)
        logger.add(
            p / "log.out"
            if level in ["DEBUG", "INFO", "SUCCESS"]
            else p / "log.err",
            level=level,
            format=format_mapping[level],
            colorize=False,
            backtrace=True,
            diagnose=True,
        )

    if python_standard_warnings:
        logger.add(lambda _: warn("DUMMY WARNING"), level="WARNING")
        logger.add(lambda _: warn("DUMMY ERROR"), level="ERROR")


def add_logger(sink, levels):
    """Quickly configure the loggers up to the level provided."""

    for level in levels:
        if isinstance(sink, (str, os.PathLike)):
            logger.add(
                sink,
                filter=generic_filter([level]),
                format=format_mapping[level],
                colorize=False,
                backtrace=True,
                enqueue=True,
            )
        else:
            logger.add(
                sink,
                filter=generic_filter([level]),
                format=format_mapping[level],
                colorize=True,
                backtrace=True,
                enqueue=True,
            )


def logger_setup(state, d):
    """Sets up the logger for SVA. There are a few modes of operation:
    state==debug does the following:
    - Logs the debug stream to d/log.debug
    - Logs the stdout stream to d/log.out (info, success)
    - Logs the stderr stream to d/log.err
    - Logs everything to the console as well
    state==normal has different behavior
    - Does not log the debug stream
    - Logs the stdout stream to d/log.out (info, success)
    - Logs the stderr stream to d/log.err
    - Logs success, error and critical to the console; omits warnings
    state==no_console completely disables the console logger but leaves the
    rest in normal mode. This is useful for when using multiprocessing or
    the hydra joblib launcher with the verbosity>0
    """

    logger.remove(None)
    d = Path(d)

    if state == "debug":
        add_logger(sys.stdout, ["DEBUG", "INFO", "SUCCESS"])
        add_logger(sys.stderr, ["WARNING", "ERROR", "CRITICAL"])
        add_logger(d / "log.debug", ["DEBUG"])
    elif state == "no_console":
        pass
    elif state == "normal":
        add_logger(sys.stdout, ["SUCCESS"])
        add_logger(sys.stderr, ["ERROR", "CRITICAL"])
    else:
        raise ValueError(f"unknown logging state {state}")

    add_logger(d / "log.out", ["INFO", "SUCCESS"])
    add_logger(d / "log.err", ["WARNING", "ERROR", "CRITICAL"])


def logger_configure_debug_mode():
    """Quick helper to enable DEBUG mode."""

    levels = ["DEBUG"] + copy(NO_DEBUG_LEVELS)
    LOGGER_STATE["levels"] = levels
    configure_loggers(levels=levels)


def logger_configure_testing_mode():
    """Enables a testing mode where loggers are configured as usual but where
    the logger.warning and logger.error calls actually also raise a dummy
    warning with the text "DUMMY WARNING" and "DUMMY ERROR", respectively.
    Used for unit tests."""

    LOGGER_STATE["python_standard_warnings"] = True
    levels = LOGGER_STATE["levels"]
    configure_loggers(levels=levels, python_standard_warnings=True)


def logger_disable_debug_mode():
    levels = copy(NO_DEBUG_LEVELS)
    LOGGER_STATE["levels"] = levels
    configure_loggers(levels=levels)


@contextmanager
def disable_logger():
    """Context manager for disabling the logger."""

    logger.disable("")
    try:
        yield None
    finally:
        logger.enable("")


@contextmanager
def logger_testing_mode():
    state = copy(LOGGER_STATE)
    logger_configure_testing_mode()
    try:
        yield None
    finally:
        configure_loggers(**state)


@contextmanager
def logger_debug():
    state = copy(LOGGER_STATE)
    logger_configure_debug_mode()
    try:
        yield None
    finally:
        configure_loggers(**state)


@contextmanager
def logger_level(*args, **kwargs):
    state = copy(LOGGER_STATE)
    configure_loggers(*args, **kwargs)
    try:
        yield None
    finally:
        configure_loggers(**state)


class CustomWarning:
    def __init__(self, w):
        self.w = w

    @property
    def name(self):
        return self.w.category.__name__

    @property
    def message(self):
        return str(self.w.message)

    @property
    def all_vars(self):
        return vars(self.w)

    def __str__(self):
        return f"{self.name}: {self.message} | {self.all_vars}"

    def __hash__(self):
        return hash(self.__str__())


def log_warnings(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        with catch_warnings(record=True, action="once") as w:
            output = f(*args, **kwargs)
        w = [CustomWarning(ww) for ww in w]
        for warning in set(w):
            logger.warning(str(warning))
        return output

    return wrapper


configure_loggers()
