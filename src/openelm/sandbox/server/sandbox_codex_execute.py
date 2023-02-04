# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is adapted from OpenAI's release
# https://github.com/openai/human-eval/blob/master/human_eval/execution.py

import contextlib
import faulthandler
import io
import os
import platform
import re
import signal
import tempfile
from enum import Enum
from typing import Any, Optional


class ExecResult(Enum):
    """An Enum to represent the result for the execution of generated code."""

    VALID = 0
    TEST_FAILED = 1
    TIMEOUT_EXCEPTION = 2
    SYNTAX_ERROR = 3
    TYPE_ERROR = 4
    EXCEPTION = 5


def unsafe_execute(
    code_str: str,
    func_name: Optional[str] = None,
    args: Optional[dict[str, Any]] = None,
    ground_truth: Optional[dict[tuple, Any]] = None,
    timeout: float = 5.0,
    debug: bool = False,
):
    if len(code_str) == 0 or "def " not in code_str:
        # No code found or no function found.
        if debug:
            print("No code found or no function found.")
            print(code_str)
        return ExecResult(5)
    func_match = re.search(r"def (\w+)\s*\((.*?)\):", code_str)
    if not func_match:
        # No proper function found in code.
        if debug:
            print("No proper function found in code.")
            print(code_str)
        return ExecResult(5)
    elif func_match and func_name is None:
        func_name = func_match.groups()[0]
    with create_tempdir():

        # Disable functionalities that can make destructive changes.
        func_dct: dict[str, Any] = reliability_guard()

        try:
            # TODO: Check https://arxiv.org/abs/2209.07753 code.
            code_dct: dict = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(code_str, code_dct)
                    if ground_truth is None:
                        if args is None:
                            result = code_dct[func_name]()
                        elif args is not None:
                            result = code_dct[func_name](**args)
                    elif ground_truth is not None:
                        if all(
                            [
                                code_dct[func_name](*arguments) == res
                                for arguments, res in ground_truth.items()
                            ]
                        ):
                            result = 0
                        else:
                            result = ExecResult(1)
        except TimeoutException as e:
            if debug:
                print(type(e), e.args)
                print(code_str)
            result = ExecResult(2)
        except SyntaxError as e:
            if debug:
                print(type(e), e.args)
                print(code_str)
            result = ExecResult(3)
        except TypeError as e:
            if debug:
                print(type(e), e.args)
                print(code_str)
            result = ExecResult(4)
        except Exception as e:
            if debug:
                print(type(e), e.args)
                print(code_str)
            result = ExecResult(5)

        # Restore system functionalities.
        reverse_reliability_guard(func_dct)

        return result


@contextlib.contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    if seconds <= 0.0:
        yield
    else:
        signal.setitimer(signal.ITIMER_REAL, seconds)
        signal.signal(signal.SIGALRM, signal_handler)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from."""

    def read(self, *args, **kwargs):
        raise OSError

    def readline(self, *args, **kwargs):
        raise OSError

    def readlines(self, *args, **kwargs):
        raise OSError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None) -> dict[str, Any]:
    """
    Safety guard for model-generated code.

    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    Warning:
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """
    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":  # MacOS doesn't have RLIMIT_STACK
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins
    import os
    import shutil
    import subprocess
    import sys

    import urllib3

    func_dct = {
        "builtins.exit": builtins.exit,
        "builtins.quit": builtins.quit,
        "builtins.input": builtins.input,
        "builtins.eval": builtins.eval,
        "builtins.help": builtins.help,
        "OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS"),
        "os.kill": os.kill,
        "os.system": os.system,
        "os.putenv": os.putenv,
        "os.remove": os.remove,
        "os.removedirs": os.removedirs,
        "os.rmdir": os.rmdir,
        "os.fchdir": os.fchdir,
        "os.setuid": os.setuid,
        "os.fork": os.fork,
        "os.forkpty": os.forkpty,
        "os.killpg": os.killpg,
        "os.rename": os.rename,
        "os.renames": os.renames,
        "os.truncate": os.truncate,
        "os.replace": os.replace,
        "os.unlink": os.unlink,
        "os.fchmod": os.fchmod,
        "os.fchown": os.fchown,
        "os.chmod": os.chmod,
        "os.chown": os.chown,
        "os.chroot": os.chroot,
        "os.lchown": os.lchown,
        "os.getcwd": os.getcwd,
        "os.chdir": os.chdir,
        "shutil.rmtree": shutil.rmtree,
        "shutil.move": shutil.move,
        "shutil.chown": shutil.chown,
        "urllib3.PoolManager": urllib3.PoolManager,
        "urllib3.HTTPConnectionPool": urllib3.HTTPConnectionPool,
        "urllib3.HTTPSConnectionPool": urllib3.HTTPSConnectionPool,
        "urllib3.HTTPResponse": urllib3.HTTPResponse,
        "subprocess.Popen": subprocess.Popen,
        "ipdb": sys.modules.get("ipdb"),
        "joblib": sys.modules.get("joblib"),
        "resource": sys.modules.get("resource"),
        "psutil": sys.modules.get("psutil"),
        "tkinter": sys.modules.get("tkinter"),
    }

    if hasattr(os, "lchmod"):
        func_dct["os.lchmod"] = os.lchmod
    if hasattr(os, "lchflags"):
        func_dct["os.lchflags"] = os.lchflags

    for key, value in func_dct.items():
        if (
            key.startswith("builtins")
            or key.startswith("os")
            or key.startswith("shutil")
            or key.startswith("urllib3")
            or key.startswith("subprocess")
        ):
            exec(key + " = None")
        elif key == "OMP_NUM_THREADS":
            exec(key + " = '1'")
        elif value is not None:
            exec("sys.modules['" + key + "'] = None")

    return func_dct


# flake8:  noqa: F401
def reverse_reliability_guard(func_dct: dict[str, Any]):
    faulthandler.enable()

    import builtins
    import os
    import shutil
    import subprocess
    import sys

    import urllib3

    for key, value in func_dct.items():
        if value is not None:
            if (
                key.startswith("builtins")
                or key.startswith("os")
                or key.startswith("shutil")
                or key.startswith("urllib3")
                or key.startswith("subprocess")
            ):
                exec(key + " = value")
            elif key == "OMP_NUM_THREADS":
                exec(key + " = '" + value + "'")
            else:
                exec("sys.modules['" + key + "'] = value")
