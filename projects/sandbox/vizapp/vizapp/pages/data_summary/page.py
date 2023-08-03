from vizapp.pages.data_summary.training_taxonomy import TrainingTaxonomy
from vizapp.pages.page import Page


class DataSummaryPage(Page):
    def __init__(self, app):
        self.app = app

        self.training_taxonomy = TrainingTaxonomy(
            self,
            self.app.waveform_prob,
            self.app.glitch_prob,
            self.app.downweight,
            self.app.swap_frac,
            self.app.mute_frac,
        )

        self.initialize_sources()

    def initialize_sources(self) -> None:
        self.training_taxonomy.initialize_sources()

    def get_layout(self):
        training_taxonomy = self.training_taxonomy.get_layout(
            height=400, width=600
        )
        return training_taxonomy

    def update(self):
        pass
