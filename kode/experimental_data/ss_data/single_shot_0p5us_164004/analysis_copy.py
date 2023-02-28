from dataanalyzer import Valueclass, Plotter, Fitter
from dataanalyzer.utilities.utilities import group_by_attr, split_by_attr


class Analysis:
    def __init__(self, fit_model=None, **kwargs):
        self.fit_model = fit_model
        self.readout_type = kwargs.get("readout_type", "I")

    def perform_analysis(self, fig, params, data, **kwargs):
        fixed_params, sweep_params = split_by_attr(params, "sweep_idx")

        sorted_sweep_params = group_by_attr(sweep_params, "sweep_idx")

        dimension = len(sorted_sweep_params)

        is_ss_data = len(data["final_I"].shape) > dimension

        if is_ss_data:
            if dimension == 1:
                plot = Plotter(fig=fig)
                x_axis = sorted_sweep_params[0][0]
                # data_complex = data["final_I"] + 1j * data["final_Q"]
                # data_complex.name = "final_IQ"
                size = min(data["final_I"].shape[0], data["final_Q"].shape[0])
                for i in range(data["final_I"].shape[1]):

                    label = f"{x_axis.name} = {x_axis.value[i]}"
                    plot.scatter(data["final_I"][:size, i], data["final_Q"][:size, i], label=label, alpha=0.5)

            # data["final_"] = data["final_complex"].real

            # data[f"final_"]

        # x = sorted_sweep_params[0][0]
        # y = data[f"final_{self.readout_type}"]

        # plot = Plotter(fig=fig)
        # if self.fit_model is not None:
        #     fit = Fitter(func=self.fit_model, x=x, y=y)
        #     fit.do_fit()
        #     plot.plot_fit(fit)
        # else:
        #     label = (
        #         None
        #         if len(sorted_sweep_params) == 1
        #         else [f"{x.name} = {x.value[0]}" for x in sorted_sweep_params[1][0]]
        #     )

        #     plot.errorbar(x, y, marker=".", label=label)
        plot.add_metadata(sweep_params)
        plot.add_metadata(fixed_params)
        plot.add_metadata(data["samples"])
        plot.add_metadata(data["elapsed_time"])
        fig = plot.show(return_fig=True)
