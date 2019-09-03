"""`tensile_test.hardening.py`"""

import numpy as np
from plotly import graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from ipywidgets import widgets

from tensile_test.tensile_test import TensileTest
from tensile_test.utils import find_nearest_index


class HardeningLawFitter(object):

    DEFAULT_THETA_0 = 250
    DEFAULT_TAU_SAT = 80
    FIG_WIDTH = 380
    FIG_HEIGHT = 280
    FIG_MARG = {
        't': 50,
        'l': 60,
        'r': 50,
        'b': 80,
    }
    FIG_PAD = [0.01, 5]
    STEP_SIZE = 1e-5

    def __init__(self, exp_stress, exp_strain, youngs_modulus=None,
                 trial_taylor_factor=None, hardening_law_params=None):

        exp_test = TensileTest(exp_stress, exp_strain,
                               youngs_modulus=youngs_modulus,
                               taylor_factor=trial_taylor_factor)

        if not hardening_law_params:
            hardening_law_params = {
                'theta_0': HardeningLawFitter.DEFAULT_THETA_0,
                'tau_sat': HardeningLawFitter.DEFAULT_TAU_SAT,
            }

        initial_values = [exp_test.shear_strain[0], exp_test.shear_stress[0]]
        hlaw = HardeningLaw(**hardening_law_params, initial_values=initial_values)
        hlaw.solve(exp_test.shear_strain[-1], HardeningLawFitter.STEP_SIZE)

        self.tensile_tests = [exp_test]
        self.hardening_laws = [hlaw]
        self.taylor_factors = [exp_test.taylor_factor]

        self._widgets = self._generate_widgets()
        self._visual = None

    def add_simulated_tensile_test(self, true_stress, true_strain):

        sim_test = TensileTest(
            true_stress=true_stress/1e6,
            true_strain=true_strain,
            youngs_modulus=self.exp_tensile_test.youngs_modulus,
            plastic_range=self.exp_tensile_test.plastic_range,
        )
        self.tensile_tests.append(sim_test)
        sim_idx = len(self.tensile_tests) - 2

        ss_type = self._widgets['macro_stress_strain']['controls']['stress_strain_type'].value
        if ss_type == 'Engineering':
            macro_stress = sim_test.eng_stress
            macro_strain = sim_test.eng_strain
        else:
            macro_stress = sim_test.true_stress
            macro_strain = sim_test.true_strain

        # Add macroscopic curve:
        fig = self._widgets['macro_stress_strain']['fig']
        fig.add_trace(
            go.Scatter(
                x=macro_strain,
                y=macro_stress,
                name='Simulated #{}'.format(sim_idx + 1),
                line=go.scatter.Line(color=DEFAULT_PLOTLY_COLORS[sim_idx + 1]),
            )
        )
        self._widgets['macro_stress_strain']['fig_trace_idx']['macro_stress_strain'].append(
            len(fig.data) - 1)

        # Add plastic curve:
        fig = self._widgets['plastic_stress_strain']['fig']
        fig.add_trace(
            go.Scatter(
                x=sim_test.plastic_strain,
                y=sim_test.plastic_stress,
                name='Simulated #{}'.format(sim_idx + 1),
                line=go.scatter.Line(color=DEFAULT_PLOTLY_COLORS[sim_idx + 1]),
            )
        )
        self._widgets['plastic_stress_strain']['fig_trace_idx']['plastic_stress_strain'].append(
            len(fig.data) - 1)

        # Make youngs modulus, trial taylor factor and plastic range "read-only" if not already.
        if len(self.tensile_tests) == 2:
            self._widgets['macro_stress_strain']['controls']['plastic_range'].disabled = True
            self._widgets['plastic_stress_strain']['controls']['youngs_modulus'].disabled = True
            self._widgets['shear_stress_strain']['controls']['taylor_factor'].disabled = True

        # Estimate new Taylor factor:
        taylor_factor_est = []

        fractions = np.arange(0.1, 1.1, 0.1)
        min_shear_strain = self.exp_tensile_test.min_shear_strain
        range_shear_strain = self.exp_tensile_test.range_shear_strain
        min_plastic_strain = sim_test.min_plastic_strain
        range_plastic_strain = sim_test.range_plastic_strain

        for i in fractions:

            shear_strain = min_shear_strain + i * range_shear_strain
            shear_idx = find_nearest_index(
                self.exp_tensile_test.shear_strain, shear_strain)
            shear_stress = self.exp_tensile_test.shear_stress[shear_idx]

            plastic_strain = min_plastic_strain + i * range_plastic_strain
            plastic_idx = find_nearest_index(sim_test.plastic_strain, plastic_strain)
            plastic_stress = sim_test.plastic_stress[plastic_idx]

            taylor_factor_est.append(plastic_stress / shear_stress)

        new_taylor_factor = np.mean(taylor_factor_est)
        self.taylor_factors.append(new_taylor_factor)

        # Add shear stress/strain curve using new taylor factor:
        new_sstress, new_sstrain = self.exp_tensile_test.get_shear_stress_strain(
            new_taylor_factor)
        fig = self._widgets['shear_stress_strain']['fig']
        fig.add_trace(
            go.Scatter(
                x=new_sstrain,
                y=new_sstress,
                name='M<sub>{}</sub> = {:.2f}'.format(sim_idx + 1, new_taylor_factor),
                line=go.scatter.Line(color=DEFAULT_PLOTLY_COLORS[sim_idx + 1]),
            )
        )
        self._widgets['shear_stress_strain']['fig_trace_idx']['shear_stress_strain'].append(
            len(fig.data) - 1)

        # Add a new HardeningLaw, with starting parameters the same as previous:
        hardening_law_params = {
            'theta_0': self.hardening_laws[-1].theta_0,
            'tau_sat': self.hardening_laws[-1].tau_sat,
        }

        initial_values = [new_sstrain[0], new_sstress[0]]
        hlaw = HardeningLaw(**hardening_law_params, initial_values=initial_values)
        hlaw.solve(new_sstrain[-1], HardeningLawFitter.STEP_SIZE)
        self.hardening_laws.append(hlaw)

        # Add new hardening law to widgets:
        fig.add_trace(
            go.Scatter(
                x=hlaw.gamma,
                y=hlaw.tau,
                name='Hard. law #{}'.format(sim_idx + 1),
                line=go.scatter.Line(
                    color=DEFAULT_PLOTLY_COLORS[sim_idx + 1], dash='dash', width=1),
            )
        )
        self._widgets['shear_stress_strain']['fig_trace_idx']['hardening_law'].append(
            len(fig.data) - 1)

        fig = self._widgets['hardening_rate']['fig']
        fig.add_trace(
            go.Scatter(
                x=hlaw.gamma,
                y=hlaw.theta,
                name='Hard. law #{}'.format(sim_idx + 1),
                line=go.scatter.Line(
                    color=DEFAULT_PLOTLY_COLORS[sim_idx + 1], dash='dash', width=1),
            )
        )
        self._widgets['hardening_rate']['fig_trace_idx']['hardening_rate'].append(
            len(fig.data) - 1)

    @property
    def trial_taylor_factor(self):
        return self.taylor_factors[0]

    @property
    def exp_tensile_test(self):
        return self.tensile_tests[0]

    def _update_widgets_stress_strain_type(self, change):

        ss_type = self._widgets['macro_stress_strain']['controls']['stress_strain_type']
        fig = self._widgets['macro_stress_strain']['fig']
        ss_trace_idx = self._widgets['macro_stress_strain']['fig_trace_idx']['macro_stress_strain']

        with fig.batch_update():
            # Update all macroscopic stress strain curves:
            for idx, i in enumerate(ss_trace_idx):
                if ss_type.value == 'Engineering':
                    stress = self.tensile_tests[idx].eng_stress
                    strain = self.tensile_tests[idx].eng_strain
                else:
                    stress = self.tensile_tests[idx].true_stress
                    strain = self.tensile_tests[idx].true_strain
                fig.data[i].x = strain
                fig.data[i].y = stress
            fig.layout.xaxis.title.text = ss_type.value + ' strain, ε'
            fig.layout.yaxis.title.text = ss_type.value + ' stress, σ / MPa'

    def _update_widgets_plastic_range(self, change):

        # Only update first one, because if once multiple stress-strain curves, plastic range is fixed.

        plastic_range = self._widgets['macro_stress_strain']['controls']['plastic_range'].value
        self.exp_tensile_test.plastic_range = plastic_range

        fig_macro = self._widgets['macro_stress_strain']['fig']
        plastic_range_trace_idx = self._widgets['macro_stress_strain']['fig_trace_idx']['plastic_range_boundaries'][0]
        with fig_macro.batch_update():
            fig_macro.data[plastic_range_trace_idx].x = [
                self.exp_tensile_test.plastic_range[0]] * 2 + [None] + [
                self.exp_tensile_test.plastic_range[1]] * 2

            fig_macro.data[plastic_range_trace_idx].y = [
                -HardeningLawFitter.FIG_PAD[1],
                HardeningLawFitter.FIG_PAD[1] + self.exp_tensile_test.max_stress,
                None,
                -HardeningLawFitter.FIG_PAD[1],
                HardeningLawFitter.FIG_PAD[1] + self.exp_tensile_test.max_stress,
            ]

        fig_plastic = self._widgets['plastic_stress_strain']['fig']
        stress_strain_trace_idx = self._widgets['plastic_stress_strain']['fig_trace_idx']['plastic_stress_strain'][0]
        with fig_plastic.batch_update():
            fig_plastic.data[stress_strain_trace_idx].x = self.exp_tensile_test.plastic_strain
            fig_plastic.data[stress_strain_trace_idx].y = self.exp_tensile_test.plastic_stress
            fig_plastic.layout['xaxis']['range'] = [self.exp_tensile_test.min_plastic_strain,
                                                    self.exp_tensile_test.max_plastic_strain]
            fig_plastic.layout['yaxis']['range'] = [self.exp_tensile_test.min_plastic_stress,
                                                    self.exp_tensile_test.max_plastic_stress]

        fig_shear = self._widgets['shear_stress_strain']['fig']
        shear_trace_idx = self._widgets['shear_stress_strain']['fig_trace_idx']['shear_stress_strain'][0]
        self.exp_tensile_test._set_shear_stress_strain()
        with fig_shear.batch_update():
            fig_shear.data[shear_trace_idx].x = self.exp_tensile_test.shear_strain
            fig_shear.data[shear_trace_idx].y = self.exp_tensile_test.shear_stress
            fig_shear.layout['xaxis']['range'] = [self.exp_tensile_test.min_shear_strain,
                                                  self.exp_tensile_test.max_shear_strain]
            fig_shear.layout['yaxis']['range'] = [self.exp_tensile_test.min_shear_stress,
                                                  self.exp_tensile_test.max_shear_stress]

    def _update_trial_taylor_factor(self, change):

        # Trial taylor factor can only be changed when there is one tensile test so
        # only need to update first trace.

        tay_fac = self._widgets['shear_stress_strain']['controls']['taylor_factor'].value
        self.taylor_factors[0] = tay_fac
        self.exp_tensile_test.taylor_factor = tay_fac
        self.exp_tensile_test._set_shear_stress_strain()

        fig = self._widgets['shear_stress_strain']['fig']
        stress_strain_trace_idx = self._widgets['shear_stress_strain']['fig_trace_idx']['shear_stress_strain'][0]
        with fig.batch_update():
            fig.data[stress_strain_trace_idx].x = self.exp_tensile_test.shear_strain
            fig.data[stress_strain_trace_idx].y = self.exp_tensile_test.shear_stress
            fig.layout['xaxis']['range'] = [self.exp_tensile_test.min_shear_strain,
                                            self.exp_tensile_test.max_shear_strain]
            fig.layout['yaxis']['range'] = [self.exp_tensile_test.min_shear_stress,
                                            self.exp_tensile_test.max_shear_stress]

        hlaw = self.hardening_laws[-1]
        hlaw.solve(self.tensile_tests[-1].shear_strain[-1], HardeningLawFitter.STEP_SIZE)

        fig_shear = self._widgets['shear_stress_strain']['fig']
        shear_trace_idx = self._widgets['shear_stress_strain']['fig_trace_idx']['hardening_law'][0]
        with fig_shear.batch_update():
            fig_shear.data[shear_trace_idx].x = hlaw.gamma
            fig_shear.data[shear_trace_idx].y = hlaw.tau

        fig_hard = self._widgets['hardening_rate']['fig']
        hard_trace_idx = self._widgets['hardening_rate']['fig_trace_idx']['hardening_rate'][0]
        with fig_hard.batch_update():
            fig_hard.data[hard_trace_idx].x = hlaw.gamma
            fig_hard.data[hard_trace_idx].y = hlaw.theta

    def _update_youngs_modulus(self, change):

        # Young's modulus can only be changed when there is one tensile test so
        # only need to update first trace.

        youngs_mod = self._widgets['plastic_stress_strain']['controls']['youngs_modulus'].value
        self.exp_tensile_test.youngs_modulus = youngs_mod
        self.exp_tensile_test._set_plastic_stress_strain()
        self.exp_tensile_test._set_shear_stress_strain()

        fig_plastic = self._widgets['plastic_stress_strain']['fig']
        stress_strain_trace_idx = self._widgets['plastic_stress_strain']['fig_trace_idx']['plastic_stress_strain'][0]
        with fig_plastic.batch_update():
            fig_plastic.data[stress_strain_trace_idx].x = self.exp_tensile_test.plastic_strain
            fig_plastic.data[stress_strain_trace_idx].y = self.exp_tensile_test.plastic_stress
            fig_plastic.layout['xaxis']['range'] = [self.exp_tensile_test.min_plastic_strain,
                                                    self.exp_tensile_test.max_plastic_strain]
            fig_plastic.layout['yaxis']['range'] = [self.exp_tensile_test.min_plastic_stress,
                                                    self.exp_tensile_test.max_plastic_stress]

        fig_shear = self._widgets['shear_stress_strain']['fig']
        shear_trace_idx = self._widgets['shear_stress_strain']['fig_trace_idx']['shear_stress_strain'][0]
        self.exp_tensile_test._set_shear_stress_strain()
        with fig_shear.batch_update():
            fig_shear.data[shear_trace_idx].x = self.exp_tensile_test.shear_strain
            fig_shear.data[shear_trace_idx].y = self.exp_tensile_test.shear_stress
            fig_shear.layout['xaxis']['range'] = [self.exp_tensile_test.min_shear_strain,
                                                  self.exp_tensile_test.max_shear_strain]
            fig_shear.layout['yaxis']['range'] = [self.exp_tensile_test.min_shear_stress,
                                                  self.exp_tensile_test.max_shear_stress]

    def _update_hardening_rules(self, change):

        # Customisable hardening parameters should always refer to the final tensile test.

        theta_0 = self._widgets['hardening_rate']['controls']['hardening_law_theta_0'].value
        tau_sat = self._widgets['hardening_rate']['controls']['hardening_law_tau_sat'].value

        hlaw = self.hardening_laws[-1]
        hlaw.theta_0 = theta_0
        hlaw.tau_sat = tau_sat
        hlaw.solve(self.tensile_tests[-1].shear_strain[-1], HardeningLawFitter.STEP_SIZE)

        fig_shear = self._widgets['shear_stress_strain']['fig']
        shear_trace_idx = self._widgets['shear_stress_strain']['fig_trace_idx']['hardening_law'][-1]
        with fig_shear.batch_update():
            fig_shear.data[shear_trace_idx].x = hlaw.gamma
            fig_shear.data[shear_trace_idx].y = hlaw.tau

        fig_hard = self._widgets['hardening_rate']['fig']
        hard_trace_idx = self._widgets['hardening_rate']['fig_trace_idx']['hardening_rate'][-1]
        with fig_hard.batch_update():
            fig_hard.data[hard_trace_idx].x = hlaw.gamma
            fig_hard.data[hard_trace_idx].y = hlaw.theta

    def _generate_widgets_macro_stress_strain(self):

        data = [
            {
                'x': self.exp_tensile_test.eng_strain,
                'y': self.exp_tensile_test.eng_stress,
                'name': 'Experimental',
                'line': {
                    'color': DEFAULT_PLOTLY_COLORS[0],
                },
            },
            {
                'x': [self.exp_tensile_test.plastic_range[0]] * 2 + [None] + [
                    self.exp_tensile_test.plastic_range[1]] * 2,
                'y': [
                    -HardeningLawFitter.FIG_PAD[1],
                    HardeningLawFitter.FIG_PAD[1] + self.exp_tensile_test.max_stress,
                    None,
                    -HardeningLawFitter.FIG_PAD[1],
                    HardeningLawFitter.FIG_PAD[1] + self.exp_tensile_test.max_stress,
                ],
                'mode': 'lines',
                'line': {
                    'color': '#888',
                    'width': 2,
                },
                'showlegend': False,
            }
        ]
        layout = {
            'title': 'Experimental Data',
            'width': HardeningLawFitter.FIG_WIDTH,
            'height': HardeningLawFitter.FIG_HEIGHT,
            'margin': HardeningLawFitter.FIG_MARG,
            'xaxis': {
                'title': 'Engineering strain, ε',
                'range': [-HardeningLawFitter.FIG_PAD[0],
                          HardeningLawFitter.FIG_PAD[0] + self.exp_tensile_test.max_strain],
            },
            'yaxis': {
                'title': 'Engineering stress, σ / MPa',
                'range': [-HardeningLawFitter.FIG_PAD[1],
                          HardeningLawFitter.FIG_PAD[1] + self.exp_tensile_test.max_stress],
            },
        }

        widget_ss_type = widgets.RadioButtons(
            options=['Engineering', 'True'],
            description='Stress/strain:',
            value='Engineering',
        )
        plastic_range_widget = widgets.FloatRangeSlider(
            value=self.exp_tensile_test.plastic_range,
            step=0.005,
            min=self.exp_tensile_test.min_true_strain,
            max=self.exp_tensile_test.max_true_strain,
            description='Plastic range:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout_format='.4f',
            layout=widgets.Layout(width='90%'),
        )
        widget_ss_type.observe(self._update_widgets_stress_strain_type, names='value')
        plastic_range_widget.observe(self._update_widgets_plastic_range, names='value')
        out = {
            'fig': go.FigureWidget(data=data, layout=layout),
            'fig_trace_idx': {
                'macro_stress_strain': [0],
                'plastic_range_boundaries': [1],
            },
            'controls': {
                'stress_strain_type': widget_ss_type,
                'plastic_range': plastic_range_widget,
            },
        }

        return out

    def _generate_widgets_plastic_stress_strain(self):

        data = [
            {
                'x': self.exp_tensile_test.plastic_strain,
                'y': self.exp_tensile_test.plastic_stress,
                'name': 'Experimental',
                'line': {
                    'color': DEFAULT_PLOTLY_COLORS[0],
                },
            },
        ]
        layout = {
            'title': 'Plastic strain',
            'width': HardeningLawFitter.FIG_WIDTH,
            'height': HardeningLawFitter.FIG_HEIGHT,
            'margin': HardeningLawFitter.FIG_MARG,
            'xaxis': {
                'title': 'Plastic strain, εₚ',
                'range': [self.exp_tensile_test.min_plastic_strain,
                          self.exp_tensile_test.max_plastic_strain],
            },
            'yaxis': {
                'title': 'True stress, σ / MPa',
                'range': [self.exp_tensile_test.min_plastic_stress,
                          self.exp_tensile_test.max_plastic_stress],
            },
            'showlegend': True,
        }

        youngs_mod_widget = widgets.BoundedFloatText(
            value=self.exp_tensile_test.youngs_modulus,
            description='Young\'s modulus (GPa):',
            min=1,
            max=2000,
            step=0.1,
            style={'description_width': 'initial'},
        )
        youngs_mod_widget.observe(self._update_youngs_modulus, names='value')

        out = {
            'fig': go.FigureWidget(data=data, layout=layout),
            'fig_trace_idx': {
                'plastic_stress_strain': [0],
            },
            'controls': {
                'youngs_modulus': youngs_mod_widget,
            },
        }

        return out

    def _generate_widgets_shear_stress_strain(self):

        hlaw = self.hardening_laws[0]
        data = [
            {
                'x': self.exp_tensile_test.shear_strain,
                'y': self.exp_tensile_test.shear_stress,
                'name': 'M<sub>0</sub> = {:.2f}'.format(self.trial_taylor_factor),
                'line': {
                    'color': DEFAULT_PLOTLY_COLORS[0],
                },
            },
            {
                'x': hlaw.gamma,
                'y': hlaw.tau,
                'name': 'Hard. law #0',
                'line': {
                    'color': DEFAULT_PLOTLY_COLORS[0],
                    'dash': 'dash',
                    'width': 1,
                },
            }
        ]
        layout = {
            'title': 'Single crystal',
            'width': HardeningLawFitter.FIG_WIDTH,
            'height': HardeningLawFitter.FIG_HEIGHT,
            'margin': HardeningLawFitter.FIG_MARG,
            'xaxis': {
                'title': 'Shear strain, γ',
                'range': [self.exp_tensile_test.min_shear_strain,
                          self.exp_tensile_test.max_shear_strain],
            },
            'yaxis': {
                'title': 'Shear stress, τ / MPa',
                'range': [self.exp_tensile_test.min_shear_stress,
                          self.exp_tensile_test.max_shear_stress],
            },
        }

        taylor_factor_widget = widgets.BoundedFloatText(
            value=self.exp_tensile_test.taylor_factor,
            description=r'Trial taylor factor, $M_{0}$:',
            min=1,
            max=5,
            step=0.1,
            style={'description_width': 'initial'},
        )
        taylor_factor_widget.observe(self._update_trial_taylor_factor, names='value')

        out = {
            'fig': go.FigureWidget(data=data, layout=layout),
            'fig_trace_idx': {
                'shear_stress_strain': [0],
                'hardening_law': [1],
            },
            'controls': {
                'taylor_factor': taylor_factor_widget,
            },
        }

        return out

    def _generate_widgets_hardening_rate(self):

        hlaw = self.hardening_laws[0]
        data = [
            {
                'x': hlaw.gamma,
                'y': hlaw.theta,
                'name': 'Hard. law #0',
                'line': {
                    'color': DEFAULT_PLOTLY_COLORS[0],
                    'dash': 'dash',
                    'width': 1,
                },
            },
        ]
        layout = {
            'title': 'Hardening rate',
            'width': HardeningLawFitter.FIG_WIDTH,
            'height': HardeningLawFitter.FIG_HEIGHT,
            'margin': HardeningLawFitter.FIG_MARG,
            'xaxis': {
                'title': 'Shear strain, γ',
                'range': [self.exp_tensile_test.min_shear_strain,
                          self.exp_tensile_test.max_shear_strain],
            },
            'yaxis': {
                'title': 'Hardening rate, θ',
            },
            'showlegend': True,
        }

        hardening_label_widget = widgets.HBox([
            widgets.Label(value='Hardening law: '),
            widgets.Label(
                value=r'$\theta = \theta_{0}\left(1 - \tau/\tau_{\textrm{sat}}\right)$')
        ])

        hardening_theta_0_widget = widgets.FloatText(
            value=hlaw.theta_0,
            description=r'$\theta_{0}$ (MPa)',
        )
        hardening_tau_sat_widget = widgets.FloatText(
            value=hlaw.tau_sat,
            description=r'$\tau_{\textrm{sat}}$ (MPa)',
        )
        hardening_theta_0_widget.observe(self._update_hardening_rules, names='value')
        hardening_tau_sat_widget.observe(self._update_hardening_rules, names='value')

        out = {
            'fig': go.FigureWidget(data=data, layout=layout),
            'fig_trace_idx': {
                'hardening_rate': [0],
            },
            'controls': {
                'hardening_law_label': hardening_label_widget,

                'hardening_law_theta_0': hardening_theta_0_widget,
                'hardening_law_tau_sat': hardening_tau_sat_widget,
            },
        }

        return out

    def _generate_widgets(self):

        out = {
            'macro_stress_strain': self._generate_widgets_macro_stress_strain(),
            'plastic_stress_strain': self._generate_widgets_plastic_stress_strain(),
            'shear_stress_strain': self._generate_widgets_shear_stress_strain(),
            'hardening_rate': self._generate_widgets_hardening_rate(),
        }

        return out

    def _generate_visual(self):
        'Layout widgets.'

        sorted_widgets = []
        for i in [
            'macro_stress_strain',
            'plastic_stress_strain',
            'shear_stress_strain',
            'hardening_rate',
        ]:
            if i in self._widgets:
                sorted_widgets.append(self._widgets[i])

        vertical_children = []
        for w in sorted_widgets:
            horizontal_children = [w['fig']]
            vertical_sub_children = [v for k, v in w['controls'].items()]
            horizontal_children.append(
                widgets.VBox(
                    vertical_sub_children,
                    layout=widgets.Layout(
                        margin='5rem 0 0 5rem',
                        width='40%',
                        #border='1px solid red',
                    )
                )
            )

            vertical_children.append(
                widgets.HBox(
                    horizontal_children,
                    layout=widgets.Layout(
                        margin='0',
                        #border='1px solid blue',
                    )
                )
            )

        return widgets.VBox(vertical_children)

    @property
    def visual(self):
        if not self._visual:
            self._visual = self._generate_visual()
        return self._visual

    def show(self, stress_strain_type='engineering'):
        'Plot stress/strain data'
        ss_type = {
            'engineering': 'Engineering',
            'true': 'True',
        }
        self._widgets['macro_stress_strain']['controls']['stress_strain_type'].value = ss_type[stress_strain_type]
        return self.visual


class HardeningLaw(object):

    def __init__(self, theta_0, tau_sat, initial_values, name='Untitled'):

        self.theta_0 = theta_0
        self.tau_sat = tau_sat
        self.initial_values = initial_values
        self.name = name

        self._gamma = None
        self._tau = None
        self._theta = None

    def hardening_rate(self, tau):
        theta = self.theta_0 * (1 - (tau / self.tau_sat))
        return theta

    def solve(self, final_strain, step_size):

        num_steps = int((final_strain - self.initial_values[0]) / step_size) + 1
        gamma_i = np.linspace(self.initial_values[0], final_strain, num_steps)
        tau_i = np.zeros_like(gamma_i) * np.nan
        theta_i = np.zeros_like(gamma_i) * np.nan

        tau_i[0] = self.initial_values[1]

        for j in range(num_steps - 1):
            theta_i[j] = self.hardening_rate(tau_i[j])
            tau_i[j + 1] = tau_i[j] + (step_size * theta_i[j])

        self._gamma = gamma_i
        self._tau = tau_i
        self._theta = theta_i

    @property
    def gamma(self):
        return self._gamma

    @property
    def tau(self):
        return self._tau

    @property
    def theta(self):
        return self._theta
