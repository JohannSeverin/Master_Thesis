""" Readout frequency analysis

This module contains the analysis class for the readout frequency analysis.

Made by: Malthe A. M. Nielsen (vpq602, malthe.asmus.nielsen@nbi.ku.dk)
Last updated: 2023-06-06
"""

from typing import List, Tuple

from dataanalyzer import Valueclass, Fitter
import dataanalyzer as da
import matplotlib.pyplot as plt
import numpy as np

from opx_control._clean_version.analysis._analysis_abc import AnalysisABC


class Analysis(AnalysisABC):
    def __init__(self, fit_model: classmethod = None, **kwargs):
        super().__init__(fit_model=fit_model, **kwargs)

        self.fit_model = fit_model
        self.fit = None

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

        if not hasattr(self, function_name):
            raise NotImplementedError(f"Analysis for {self.sweep_dimensions}D {data_type} data not implemented.")

        return getattr(self, function_name)()

    def _2d_averaged_analysis(self):
        x, y = self.sweep_axes
        z = self.readout_data[self.readout_type]

        # Ig = self.readout_data["I"][0]
        # Ie = self.readout_data["I"][1]

        # Qg = self.readout_data["Q"][0]
        # Qe = self.readout_data["Q"][1]

        # Z = (Ie.value - Ig.value) + 1j * (Qe.value - Qg.value)
        # var = (Ig.error + Qg.error + Ie.error + Qe.error) / 4
        # SNR = ((np.abs(Z)) ** 2) / (2 * var)

        SNR, var = self.calculate_SNR(self.readout_data["I"][-2:], self.readout_data["Q"][-2:])
        SNR = Valueclass(tag="SNR", name="SNR", value=SNR, error=var)

        self.plot = da.Plotter((2, 1), fig=self.fig, **self.plt_settings)

        for i, yi in enumerate(y):
            self.plot.errorbar(x, z[i], label=f"{yi.value[0]:.2f}", markersize=2, ls="-", ax=0, color=f"C{i}")

        # self.plot.errorbar(x, SNR, markersize=2, ax=1, color="C2", ls="-")

        if self.fit_model is not None:
            self.fit = Fitter(self.fit_model, x, SNR)
            self.plot.plot_fit(self.fit, ax=1)

    def _3d_averaged_analysis(self):
        x, y, v = self.sweep_axes

        z = self.readout_data[self.readout_type]

        SNR, var = self.calculate_SNR(self.readout_data["I"][..., -2:, :], self.readout_data["Q"][..., -2:, :])
        self.SNR = Valueclass(tag="SNR", name="SNR", value=SNR, error=var)

        self.plot = da.Plotter((3, 1), fig=self.fig, **self.plt_settings)

        for i, yi in enumerate(y):
            self.plot.errorbar(x, z[0, i, :], label=f"{yi.value[0]:.2f}", markersize=2, ls="-", ax=0, color=f"C{i}")

        center_frequency = np.zeros(len(v))
        center_frequency_error = np.zeros(len(v))

        for i in range(len(v)):
            if self.fit_model is not None:
                self.fit = Fitter(self.fit_model, x, self.SNR[i])
                self.fit.do_fit()
                center_frequency[i] = self.fit.parameters["center"]["value"]
                center_frequency_error[i] = self.fit.parameters["center"]["error"]

        self.center_frequency = Valueclass(
            tag="center_frequency",
            name="Center frequency",
            value=center_frequency,
            error=center_frequency_error,
            unit="Hz",
        )

        self.plot.heatmap(x, v, self.SNR, ax=1)

        self.plot.errorbar(v, self.center_frequency, ax=2)

    @staticmethod
    def calculate_SNR(I, Q):
        """Calculates the SNR for n states."""

        Z = np.std(I.value, axis=-2) + 1j * np.std(Q.value, axis=-2)
        var = np.mean(np.concatenate((I.error, Q.error), axis=-2), axis=-2)

        return ((np.abs(Z)) ** 2) / (2 * var), var
