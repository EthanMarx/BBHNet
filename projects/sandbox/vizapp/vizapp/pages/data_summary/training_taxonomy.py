import numpy as np
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, CustomJS, Slider
from bokeh.palettes import TolRainbow
from bokeh.plotting import figure


class TrainingTaxonomy:
    def __init__(
        self,
        page,
        waveform_prob: float,
        glitch_prob: float,
        downweight: float,
        swap_frac: float,
        mute_frac: float,
    ):
        self.page = page
        self.waveform_prob = waveform_prob
        self.glitch_prob = glitch_prob
        self.downweight = downweight
        self.swap_frac = swap_frac
        self.mute_frac = mute_frac

    def initialize_sources(self):
        self.source = dict()
        self.source["category"] = [
            "Background",
            "Background, Muted",
            "Background, Swapped",
            "Background, Swapped, Muted",
            "One Glitch",
            "One Glitch, Muted",
            "One Glitch, Swapped",
            "One Glitch, Swapped, Muted",
            "Two Glitches",
            "Two Glitches, Muted",
            "Two Glitches, Swapped",
            "Two Glitches, Swapped, Muted",
            "Waveform",
            "Waveform, One Glitch",
            "Waveform, Two Glitches",
        ]

        new_waveform_prob = self.waveform_prob / (
            (1 - self.glitch_prob * (1 - self.downweight)) ** 2
            * (
                1
                - (
                    self.swap_frac
                    + self.mute_frac
                    - self.mute_frac * self.swap_frac
                )
            )
        )
        background = (1 - self.glitch_prob) ** 2 * (1 - new_waveform_prob)
        background_swapped = (
            (1 - self.glitch_prob) ** 2
            * new_waveform_prob
            * self.swap_frac
            * (1 - self.mute_frac)
        )
        background_muted = (
            (1 - self.glitch_prob) ** 2
            * new_waveform_prob
            * self.mute_frac
            * (1 - self.swap_frac)
        )
        background_swap_mute = (
            (1 - self.glitch_prob) ** 2
            * new_waveform_prob
            * self.mute_frac
            * self.swap_frac
        )
        one_glitch = (
            2
            * self.glitch_prob
            * (1 - self.glitch_prob)
            * (1 - new_waveform_prob * self.downweight)
        )
        one_glitch_swapped = (
            2
            * self.glitch_prob
            * (1 - self.glitch_prob)
            * new_waveform_prob
            * self.downweight
            * self.swap_frac
            * (1 - self.mute_frac)
        )
        one_glitch_muted = (
            2
            * self.glitch_prob
            * (1 - self.glitch_prob)
            * new_waveform_prob
            * self.downweight
            * self.mute_frac
            * (1 - self.swap_frac)
        )
        one_glitch_swap_mute = (
            2
            * self.glitch_prob
            * (1 - self.glitch_prob)
            * new_waveform_prob
            * self.downweight
            * self.swap_frac
            * self.mute_frac
        )
        two_glitches = self.glitch_prob**2 * (
            1 - new_waveform_prob * self.downweight**2
        )
        two_glitches_swapped = (
            self.glitch_prob**2
            * new_waveform_prob
            * self.downweight**2
            * self.swap_frac
            * (1 - self.mute_frac)
        )
        two_glitches_muted = (
            self.glitch_prob**2
            * new_waveform_prob
            * self.downweight**2
            * self.mute_frac
            * (1 - self.swap_frac)
        )
        two_glitches_swap_mute = (
            self.glitch_prob**2
            * new_waveform_prob
            * self.downweight**2
            * self.swap_frac
            * self.mute_frac
        )
        waveform = (
            (1 - self.glitch_prob) ** 2
            * new_waveform_prob
            * (1 - self.swap_frac)
            * (1 - self.mute_frac)
        )
        waveform_one_glitch = (
            2
            * self.glitch_prob
            * (1 - self.glitch_prob)
            * new_waveform_prob
            * self.downweight
            * (1 - self.swap_frac)
            * (1 - self.mute_frac)
        )
        waveform_two_glitches = (
            self.glitch_prob**2
            * new_waveform_prob
            * self.downweight**2
            * (1 - self.swap_frac)
            * (1 - self.mute_frac)
        )

        self.source["fraction"] = [
            background,
            background_muted,
            background_swapped,
            background_swap_mute,
            one_glitch,
            one_glitch_muted,
            one_glitch_swapped,
            one_glitch_swap_mute,
            two_glitches,
            two_glitches_muted,
            two_glitches_swapped,
            two_glitches_swap_mute,
            waveform,
            waveform_one_glitch,
            waveform_two_glitches,
        ]

        end_angle = [
            val * 2 * np.pi for val in np.cumsum(self.source["fraction"])
        ]
        start_angle = np.roll(end_angle, 1)
        start_angle[0] = 0

        self.source["color"] = TolRainbow[15]
        self.source["start_angle"] = start_angle
        self.source["end_angle"] = end_angle
        self.source = ColumnDataSource(data=self.source)

    def get_layout(self, height: int, width: int):
        self.p = figure(
            height=height,
            width=width,
            title="Kernel Categories",
            toolbar_location=None,
            tools="hover",
            tooltips="@category: @fraction",
            x_range=(-0.5, 1.0),
        )

        self.p.wedge(
            x=0,
            y=1,
            radius=0.4,
            start_angle="start_angle",
            end_angle="end_angle",
            line_color="white",
            fill_color="color",
            legend_group="category",
            source=self.source,
        )

        self.waveform_slider = Slider(
            title="Waveform Prob",
            start=0,
            end=1,
            step=0.01,
            value=self.waveform_prob,
        )
        self.glitch_slider = Slider(
            title="Glitch Prob",
            start=0,
            end=1,
            step=0.01,
            value=self.glitch_prob,
        )
        self.downweight_slider = Slider(
            title="Downweight",
            start=0,
            end=1,
            step=0.01,
            value=self.downweight,
        )
        self.swap_slider = Slider(
            title="Swap Frac", start=0, end=1, step=0.01, value=self.swap_frac
        )
        self.mute_slider = Slider(
            title="Mute Frac", start=0, end=1, step=0.01, value=self.mute_frac
        )

        self.p.axis.axis_label = None
        self.p.axis.visible = False
        self.p.grid.grid_line_color = None

        self.create_callback()

        return column(
            row(self.waveform_slider, self.glitch_slider, self.mute_slider),
            row(self.downweight_slider, self.swap_slider),
            self.p,
        )

    def create_callback(self):
        callback = CustomJS(
            args=dict(source=self.source),
            code="""
            const data = source.data
            const wf_prob = wf_slider.value
            const gl_prob = gl_slider.value
            const dw = dw_slider.value
            const sw_frac = sw_slider.value
            const mu_frac = mu_slider.value
            const new_wf_prob = wf_prob / (Math.pow((1 - gl_prob * (1 - dw)),2)
                                * (1 - (sw_frac + mu_frac - sw_frac*mu_frac)))
            const bg = Math.pow((1 - gl_prob), 2) * (1 - new_wf_prob)
            const bg_swapped = Math.pow((1 - gl_prob), 2)
                               * new_wf_prob * sw_frac * (1 - mu_frac)
            const bg_muted = Math.pow((1 - gl_prob), 2) * new_wf_prob
                             * mu_frac * (1 - sw_frac)
            const bg_swap_mute = Math.pow((1 - gl_prob), 2) * new_wf_prob
                                 * sw_frac * mu_frac
            const one_gl = 2 * gl_prob * (1 - gl_prob) * (1 - new_wf_prob * dw)
            const one_gl_swapped = 2 * gl_prob * (1 - gl_prob) * new_wf_prob
                                   * dw * sw_frac * (1 - mu_frac)
            const one_gl_muted = 2 * gl_prob * (1 - gl_prob) * new_wf_prob
                                 * dw * mu_frac * (1 - sw_frac)
            const one_gl_swap_mute = 2 * gl_prob * (1 - gl_prob) * new_wf_prob
                                     * dw * sw_frac * mu_frac
            const two_gl = Math.pow(gl_prob, 2) * (1 - new_wf_prob * dw ** 2)
            const two_gl_swapped = Math.pow(gl_prob, 2) * new_wf_prob * dw ** 2
                                   * sw_frac * (1 - mu_frac)
            const two_gl_muted = Math.pow(gl_prob, 2) * new_wf_prob * dw ** 2
                                 * mu_frac * (1 - sw_frac)
            const two_gl_swap_mute = Math.pow(gl_prob, 2) * new_wf_prob
                                     * dw ** 2 * sw_frac * mu_frac
            const wf = Math.pow((1 - gl_prob), 2) * new_wf_prob * (1 - sw_frac)
                       * (1 - mu_frac)
            const wf_one_gl = 2 * gl_prob * (1 - gl_prob) * new_wf_prob * dw
                              * (1 - sw_frac) * (1 - mu_frac)
            const wf_coinc_gl = Math.pow(gl_prob, 2) * new_wf_prob
                                * Math.pow(dw, 2) * (1 - sw_frac)
                                * (1 - mu_frac)
            const start_angle = data["start_angle"]
            const end_angle = data["end_angle"]
            const values = [bg,
                            bg_muted,
                            bg_swapped,
                            bg_swap_mute,
                            one_gl,
                            one_gl_muted,
                            one_gl_swapped,
                            one_gl_swap_mute,
                            two_gl,
                            two_gl_muted,
                            two_gl_swapped,
                            two_gl_swap_mute,
                            wf,
                            wf_one_gl,
                            wf_coinc_gl]
            let sum = 0
            for (let i = 0; i < start_angle.length; i++){
                start_angle[i] = sum * 2 * Math.PI
                sum = sum + values[i]
                end_angle[i] = sum * 2 * Math.PI
                data["fraction"][i] = values[i]
            }
            if (new_wf_prob > 1){
                for (let i = 0; i < start_angle.length; i++){
                    start_angle[i] = 0
                    end_angle[i] = 0
                }
            }
            source.change.emit()
        """,
        )

        self.waveform_slider.js_on_change("value", callback)
        callback.args["wf_slider"] = self.waveform_slider

        self.glitch_slider.js_on_change("value", callback)
        callback.args["gl_slider"] = self.glitch_slider

        self.downweight_slider.js_on_change("value", callback)
        callback.args["dw_slider"] = self.downweight_slider

        self.swap_slider.js_on_change("value", callback)
        callback.args["sw_slider"] = self.swap_slider

        self.mute_slider.js_on_change("value", callback)
        callback.args["mu_slider"] = self.mute_slider
