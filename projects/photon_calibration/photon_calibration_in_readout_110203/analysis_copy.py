""" Basic analysis module

This file contains the basic analysis. A common analysis for all sequences. Plots and fits the data.

Made by: Malthe A. M. Nielsen (vpq602, malthe.asmus.nielsen@nbi.ku.dk)
Last updated: 2023-05-21
"""

from typing import List, Tuple

from dataanalyzer import Valueclass
import dataanalyzer as da
import matplotlib.pyplot as plt
import numpy as np

from dataanalyser.methods.legacy._analysis_abc import AnalysisABC


class Analysis(AnalysisABC):
    def __init__(self, fit_model: classmethod = None, **kwargs):
        super().__init__(fit_model=fit_model, **kwargs)
        self.plot_2d = kwargs.get("plot_2d", True)
        self.bare_axis = kwargs.get("bare_axis", False)
        self.trace_operation = kwargs.get("trace_operation", None)
        if not type(self.trace_operation) == list:
            self.trace_operation = [self.trace_operation]
        self.trace_operation_axis = kwargs.get("trace_operation_axis", 1)
        self.logy = kwargs.get("logy", False)

    def perform_analysis(self, data: dict = None, fig: plt.figure = None, **kwargs):
        super().perform_analysis(data=data, fig=fig, **kwargs)
        result = self._perform_analysis()
        self.finish_plot()

    def _perform_analysis(self) -> dict:
        """Determines the type of analysis to perform, and performs it.

        Returns:
            dict: The result of the analysis.
        """
        data_type = "averaged"
        data_type = "single_shot" if self.is_single_shot else data_type
        data_type = "tomography" if self.is_tomography else data_type
        data_type = "adc" if self.is_adc else data_type

        function_name = f"_{self.sweep_dimensions}d_{data_type}_analysis"

        if not self.plot_2d:
            function_name = function_name.replace("_analysis", "_flat_analysis")

        if not hasattr(self, function_name):
            if self.plot_2d:
                raise NotImplementedError(
                    f"Analysis for {self.sweep_dimensions}D {data_type} data not implemented."
                )
            raise NotImplementedError(
                f"Analysis for {self.sweep_dimensions}D {data_type} data when plot_2d=False not implemented."
            )

        return getattr(self, function_name)()

    def _0d_adc_analysis(self):
        y_I = self.readout_data["adc_I"]
        y_Q = self.readout_data["adc_Q"]
        # y_abs = self.readout_data["adc_abs"]

        x = Valueclass(name="Time", unit="s", value=np.arange(len(y_I)) * 1e-9)
        # self.plot = da.Plotter((1, 1), fig=self.fig, **self.plt_settings)
        # self.plot.errorbar(x, y_abs, ax=(0, 0), title="abs")
        self.plot = da.Plotter((2, 1), fig=self.fig, **self.plt_settings)
        self.plot.plot(x, y_I, ax=(0, 0), title="I")
        self.plot.plot(x, y_Q, ax=(1, 0), title="Q")

    def _1d_averaged_analysis(self):
        x = self.sweep_axes

        if self.readout_type == "logical":
            y = self.readout_data["logical_0"]
            y_states = [
                self.readout_data[f"logical_{i}"]
                for i in range(self.readout_data["logical"].shape[0])
            ]

            for y_state in y_states:
                y_state.name = f"|{y_state.name.split('_')[-1].replace(')', '')}>"

                if y_state.name == "|-1>":
                    y_state.name = "out."

        else:
            y = self.readout_data[self.readout_type]
            y_states = None

        self.plot = da.Plotter((1, 1), fig=self.fig, **self.plt_settings)

        if self.fit_model is not None:
            self.fit = da.Fitter(model=self.fit_model, x=x, y=y, **self.fit_settings)
            self.plot.plot_fit(self.fit, plot_data=True)

            for name, fit_parameter in self.fit.get_parameters().items():
                self.experiment.dataset = self.experiment.dataset.assign(
                    {
                        name: fit_parameter["value"],
                        f"{name}__error": fit_parameter["error"],
                    }
                )

        else:
            self.plot.errorbar(x, y)

        if y_states is not None:
            for y_state in y_states:
                self.plot.errorbar(x, y_state, ax=0, label=y_state.name)
                self.plot.ax.set_ylabel("Expectation value")

    def _2d_averaged_analysis(self):
        x, y = self.sweep_axes
        z = self.readout_data[self.readout_type]
        if any(self.trace_operation):
            for operation in self.trace_operation:
                z = z.traces(operation, axis=self.trace_operation_axis)
        self.plot = da.Plotter((1, 1), fig=self.fig, **self.plt_settings)
        self.plot.pcolormesh(x, y, z)
        if self.logy:
            self.plot.ax.set_yscale("log")

    def _3d_averaged_analysis(self):
        x1, x2, x3 = self.sweep_axes
        y = self.readout_data[self.readout_type]

        self.plot = da.Plotter((len(x3), 1), fig=self.fig, **self.plt_settings)

        for i, x3i in enumerate(x3):
            self.plot.pcolormesh(
                x1,
                x2,
                y[i, :, :],
                ax=(i, 0),
                title=x3i.tostr(string_type="name_value"),
            )
            if self.bare_axis:
                if i < len(x3) - 1:
                    self.plot.clear_ax()
                else:
                    self.plot.ax.set_title("")
                    self.plot.ax.colorbar[0].remove()

                self.plot.ax.text(
                    0.05,
                    0.95,
                    x3i.tostr(string_type="name_value"),
                    transform=self.plot.ax.transAxes,
                    va="top",
                    ha="left",
                    bbox=dict(facecolor="white", alpha=0.5),
                )

    def _4d_averaged_analysis(self):
        x1, x2, x3, x4 = self.sweep_axes
        y = self.readout_data[self.readout_type]

        self.plot = da.Plotter((len(x4), len(x3)), fig=self.fig, **self.plt_settings)
        for i, x4i in enumerate(x4):
            for j, x3j in enumerate(x3):
                self.plot.pcolormesh(
                    x1,
                    x2,
                    y[i, j, :, :],
                    ax=(len(x4) - 1 - i, j),
                    title=f"{x4i.tostr(string_type='name_value')} \n {x3j.tostr(string_type='name_value')}",
                )
                if self.bare_axis:
                    if i > 0 or j > 0:
                        self.plot.clear_ax()
                    else:
                        self.plot.ax.set_title("")
                        self.plot.ax.colorbar[0].remove()

    def _0d_single_shot_analysis(self):
        I = self.readout_data["I"]
        Q = self.readout_data["Q"]

        self.plot = da.Plotter((1, 1), fig=self.fig, **self.plt_settings)
        self.plot.plot(I, Q, marker="o", linestyle="none", alpha=0.75)

    def _1d_single_shot_analysis(
        self,
    ):  # TODO: What is going on here, is this single_shot data???
        x_vec = self.sweep_axes
        I_vec = self.readout_data["I"]
        Q_vec = self.readout_data["Q"]
        self.plot = da.Plotter((1, 1), fig=self.fig, **self.plt_settings)
        for I, Q, x in zip(I_vec, Q_vec, x_vec):
            self.plot.plot(
                I,
                Q,
                marker="o",
                linestyle="none",
                alpha=0.75,
                label=x.tostr(string_type="value_name"),
            )

    # Flat analysis ------------------------------------------------------------------ #
    def _2d_averaged_flat_analysis(self):
        x, y = self.sweep_axes
        z = self.readout_data[self.readout_type]

        self.plot = da.Plotter((1, 1), fig=self.fig, **self.plt_settings)

        # Add colorbar
        self.plot._add_colorbar(None, y, keep_colorbar=False)

        for i, yi in enumerate(y):
            # Determine color from max and min of z
            zcolor = self.get_color(yi.value, y.value, cmap="viridis")

            self.plot.errorbar(x, z[i, :], color=zcolor)
