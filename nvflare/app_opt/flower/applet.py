# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import time

from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.cli_applet import CLIApplet
from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.app_common.tie.process_mgr import CommandDescriptor, ProcessManager, start_process
from nvflare.app_opt.flower.defs import Constant
from nvflare.app_opt.flower.utils import create_channel
from nvflare.fuel.f3.drivers.net_utils import get_open_tcp_port
from nvflare.security.logging import secure_format_exception


class FlowerClientApplet(CLIApplet):
    def __init__(
        self,
        client_app: str,
    ):
        """Constructor of FlowerClientApplet, which extends CLIApplet.

        Args:
            client_app: the client app specification of the Flower app
        """
        CLIApplet.__init__(self)
        self.client_app = client_app

    def get_command(self, ctx: dict) -> CommandDescriptor:
        """Implementation of the get_command method required by the super class CLIApplet.
        It returns the CLI command for starting Flower's client app, as well as the full path of the log file
        for the client app.

        Args:
            ctx: the applet run context

        Returns: CLI command for starting client app and name of log file.

        """
        addr = ctx.get(Constant.APP_CTX_SERVER_ADDR)
        fl_ctx = ctx.get(Constant.APP_CTX_FL_CONTEXT)
        if not isinstance(fl_ctx, FLContext):
            self.logger.error(f"expect APP_CTX_FL_CONTEXT to be FLContext but got {type(fl_ctx)}")
            raise RuntimeError("invalid FLContext")

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        if not isinstance(ws, Workspace):
            self.logger.error(f"expect workspace to be Workspace but got {type(ws)}")
            raise RuntimeError("invalid workspace")

        job_id = fl_ctx.get_job_id()
        custom_dir = ws.get_app_custom_dir(job_id)
        app_dir = ws.get_app_dir(job_id)
        cmd = f"flower-client-app --insecure --grpc-adapter --superlink {addr} --dir {custom_dir} {self.client_app}"

        # use app_dir as the cwd for flower's client app.
        # this is necessary for client_api to be used with the flower client app for metrics logging
        # client_api expects config info from the "config" folder in the cwd!
        self.logger.info(f"starting flower client app: {cmd}")
        return CommandDescriptor(cmd=cmd, cwd=app_dir, log_file_name="client_app_log.txt", stdout_msg_prefix="FLWR-CA")


class FlowerServerApplet(Applet):
    def __init__(self, server_app: str, database: str, superlink_ready_timeout: float):
        """Constructor of FlowerServerApplet.

        Args:
            server_app: Flower's server app specification
            database: database spec to be used by the server app
            superlink_ready_timeout: how long to wait for the superlink process to become ready
        """
        Applet.__init__(self)
        self._app_process_mgr = None
        self._superlink_process_mgr = None
        self.server_app = server_app
        self.database = database
        self.superlink_ready_timeout = superlink_ready_timeout
        self._start_error = False

    def _start_process(self, name: str, cmd_desc: CommandDescriptor, fl_ctx: FLContext) -> ProcessManager:
        self.logger.info(f"starting {name}: {cmd_desc.cmd}")
        try:
            return start_process(cmd_desc, fl_ctx)
        except Exception as ex:
            self.logger.error(f"exception starting applet: {secure_format_exception(ex)}")
            self._start_error = True

    def start(self, ctx: dict):
        """Start the applet.

        Flower requires two processes for server application:
            superlink: this process is responsible for client communication
            server_app: this process performs server side of training.

        We start the superlink first, and wait for it to become ready, then start the server app.
        Each process will have its own log file in the job's run dir. The superlink's log file is named
        "superlink_log.txt". The server app's log file is named "server_app_log.txt".

        Args:
            ctx: the run context of the applet.

        Returns:

        """
        # try to start superlink first
        driver_port = get_open_tcp_port(resources={})
        if not driver_port:
            raise RuntimeError("failed to get a port for Flower driver")
        driver_addr = f"127.0.0.1:{driver_port}"

        server_addr = ctx.get(Constant.APP_CTX_SERVER_ADDR)
        fl_ctx = ctx.get(Constant.APP_CTX_FL_CONTEXT)
        if not isinstance(fl_ctx, FLContext):
            self.logger.error(f"expect APP_CTX_FL_CONTEXT to be FLContext but got {type(fl_ctx)}")
            raise RuntimeError("invalid FLContext")

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        if not isinstance(ws, Workspace):
            self.logger.error(f"expect workspace to be Workspace but got {type(ws)}")
            raise RuntimeError("invalid workspace")

        custom_dir = ws.get_app_custom_dir(fl_ctx.get_job_id())

        db_arg = ""
        if self.database:
            db_arg = f"--database {self.database}"

        superlink_cmd = (
            f"flower-superlink --insecure {db_arg} "
            f"--fleet-api-address {server_addr} --fleet-api-type grpc-adapter "
            f"--driver-api-address {driver_addr}"
        )

        cmd_desc = CommandDescriptor(cmd=superlink_cmd, log_file_name="superlink_log.txt", stdout_msg_prefix="FLWR-SL")

        self._superlink_process_mgr = self._start_process(name="superlink", cmd_desc=cmd_desc, fl_ctx=fl_ctx)
        if not self._superlink_process_mgr:
            raise RuntimeError("cannot start superlink process")

        # wait until superlink's port is ready before starting server app
        # note: the server app will connect to driver_addr, not server_addr
        start_time = time.time()
        create_channel(
            server_addr=driver_addr,
            grpc_options=None,
            ready_timeout=self.superlink_ready_timeout,
            test_only=True,
        )
        self.logger.info(f"superlink is ready for server app in {time.time()-start_time} seconds")

        # start the server app
        app_cmd = f"flower-server-app --insecure --superlink {driver_addr} --dir {custom_dir} {self.server_app}"
        cmd_desc = CommandDescriptor(
            cmd=app_cmd,
            log_file_name="server_app_log.txt",
            stdout_msg_prefix="FLWR-SA",
        )

        self._app_process_mgr = self._start_process(name="server_app", cmd_desc=cmd_desc, fl_ctx=fl_ctx)
        if not self._app_process_mgr:
            # stop the superlink
            self._superlink_process_mgr.stop()
            self._superlink_process_mgr = None
            raise RuntimeError("cannot start server_app process")

    @staticmethod
    def _stop_process(p: ProcessManager):
        if p:
            try:
                p.stop()
            except:
                pass

    def stop(self, timeout=0.0):
        """Stop the server applet's superlink and server app processes.

        Args:
            timeout: how long to wait before forcefully stopping (kill) the process.

        Note: we always stop the process immediately - do not wait for the process to stop itself.

        Returns:

        """
        self._stop_process(self._app_process_mgr)
        self._app_process_mgr = None

        self._stop_process(self._superlink_process_mgr)
        self._superlink_process_mgr = None

    @staticmethod
    def _is_process_stopped(p: ProcessManager):
        if p:
            return_code = p.poll()
            if return_code is None:
                return False, 0
            else:
                return True, return_code
        else:
            return True, 0

    def is_stopped(self) -> (bool, int):
        """Check whether the server applet is already stopped

        Returns: a tuple of: whether the applet is stopped, exit code if stopped.

        Note: if either superlink or server app is stopped, we treat the applet as stopped.

        """
        if self._start_error:
            return True, TieConstant.EXIT_CODE_CANT_START

        # check server app
        app_stopped, app_rc = self._is_process_stopped(self._app_process_mgr)
        if app_stopped:
            self._app_process_mgr = None

        superlink_stopped, superlink_rc = self._is_process_stopped(self._superlink_process_mgr)
        if superlink_stopped:
            self._superlink_process_mgr = None

        if app_stopped or superlink_stopped:
            self.stop()

        if app_stopped:
            return True, app_rc
        elif superlink_stopped:
            return True, superlink_rc
        else:
            return False, 0
