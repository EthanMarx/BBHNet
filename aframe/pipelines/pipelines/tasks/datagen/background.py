import os

import luigi
from pipelines.configs import aframe
from pipelines.tasks.apptainer import CondorApptainerTask
from pipelines.tasks.datagen.segments import Segments


class GenerateBackground(CondorApptainerTask):
    data_dir = luigi.Parameter(default=os.getenv("DATA_DIR", ""))
    start = luigi.FloatParameter()
    stop = luigi.FloatParameter()
    state_flag = luigi.Parameter()
    minimum_length = luigi.FloatParameter()
    ifos = luigi.ListParameter(default=aframe().ifos)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return "generate_background"

    @property
    def queue(self):
        return (
            f"queue start,stop,writepath from {self.data_dir}/parameters.txt"
        )

    def requires(self):
        Segments(
            data_dir=self.data_dir,
            start=self.start,
            stop=self.stop,
            state_flag=self.state_flag,
            minimum_length=self.minimum_length,
            ifos=self.ifos,
        )

    @property
    def command(self):
        command = f"""
            python /opt/aframe/aframe/datagen/datagen/background.py
                --start $(start)
                --stop $(stop)
                --channel {self.channel}
                --ifos {' '.join(self.ifos)}
                --sample-rate {self.sample_rate}
                --write-path $(writepath)
        """
        return command

    @property
    def image(self) -> str:
        default = os.path.expanduser("~/aframe/images")
        root = os.environ.get("AFRAME_CONTAINER_ROOT", default)
        return os.path.join(root, "datagen.sif")

    def validate(self):
        # here we will validate cached files and write
        # a new file containint start,stop for data we need to query
        pass

    def run(self):
        self.validate()
        super.run()