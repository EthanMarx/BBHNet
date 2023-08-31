import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from textwrap import dedent
from typing import Sequence

import luigi
from pycondor import Job

logger = logging.getLogger("luigi-interface")


class ImageNotFound(Exception):
    pass


class ApptainerTask(luigi.Task):
    @property
    def image(self) -> str:
        raise NotImplementedError

    @property
    def command(self) -> str:
        return "echo hello world"

    @property
    def environment(self) -> dict:
        return {}

    @property
    def binds(self) -> dict:
        return {}

    @property
    def gpus(self) -> Sequence[int]:
        return []

    @property
    def log_output(self) -> bool:
        return True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not os.path.exists(self.image):
            raise ImageNotFound(
                f"Couldn't find container image {self.image} locally"
            )
        self._binds = self.binds
        self.__logger = logger

    @property
    def base_command(self):
        return ["apptainer", "run"]

    def build_command(self):
        cmd = self.base_command
        for source, dest in self._binds.items():
            cmd.extend(["--bind", f"{source}:{dest}"])

        if self.gpus:
            cmd.append("--nv")

            gpus = ",".join(map(str, self.gpus))
            cmd.extend(["--env", f"APPTAINERENV_CUDA_VISIBLE_DEVICES={gpus}"])

        cmd.append(self.image)

        command = dedent(self.command).replace("\n", " ")
        command = shlex.split(command)
        cmd.extend(command)

        return cmd

    def build_env(self):
        env = {}
        for key, value in self.environment.items():
            env[f"APPTAINERENV_{key}"] = value
        return env

    def run(self):
        env = self.build_env()
        cmd = self.build_command()

        try:
            proc = subprocess.run(
                cmd, capture_output=True, check=True, env=env, text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                "Command '{}' failed with return code {} "
                "and stderr:\n{}".format(
                    shlex.join(e.cmd), e.returncode, e.stderr
                )
            ) from None

        if self.log_output:
            self.__logger.info(proc.stdout)


class AframeApptainerTask(ApptainerTask):
    dev = luigi.BoolParameter(default=False)

    def __init__(self, *args, **kwargs):
        root = Path(__file__).resolve()
        while root.name != "aframe":
            root = root.parent
        self.root = root.parent
        super().__init__(*args, **kwargs)

        if self.dev:
            self._binds[self.root] = "/opt/aframe"


class CondorApptainerTask(AframeApptainerTask):
    submit_dir = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return "aframe"

    @property
    def base_command(self):
        return ["exec"]

    @property
    def queue(self):
        # to allow e.g. "queue start,stop from segments.txt" syntax.
        # this will require a pycondor change
        # to allow `queue` values that are strings
        return "queue"

    def build_env(self):
        env = ""
        for key, value in self.environment.items():
            env += f"APPTAINERENV_{key} = {value} "
        return env

    def run(self):
        env = self.build_env()
        cmd = self.build_command()

        job = Job(
            name=self.name,
            executable=shutil.which("apptainer"),
            error=self.submit_dir,
            output=self.submit_dir,
            log=self.submit_dir,
            arguments=" ".join(cmd),
            extra_lines=[f"environment = {env}"],
            queue=self.queue,
        )
        job.build_submit(fancyname=False)
