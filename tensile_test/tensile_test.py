"""`tensile_test.tensile_test.py`"""

from pathlib import Path
import json

import numpy as np
from plotly import graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from ipywidgets import widgets

from tensile_test.utils import find_nearest_index, read_non_uniform_csv, nan_array_to_list


def get_true_stress_strain(eng_stress, eng_strain):
    'Convert engineering stress/strain to true stress/strain'

    common = 1 + eng_strain
    true_stress = eng_stress * (common)
    true_strain = np.log(common)

    return true_stress, true_strain


def get_eng_stress_strain(true_stress, true_strain):
    'Convert true stress/strain to engineering stress/strain.'
    eng_strain = np.exp(true_strain) - 1
    eng_stress = true_stress / (1 + eng_strain)

    return eng_stress, eng_strain


class TensileTest(object):

    DEFAULT_TAYLOR_FACTOR = 2.5
    DEFAULT_YOUNGS_MOD = 79  # GPa
    DEFAULT_PLASTIC_RANGE_START = 0.02
    FIG_WIDTH = 380
    FIG_HEIGHT = 280
    FIG_MARG = {
        't': 50,
        'l': 60,
        'r': 50,
        'b': 80,
    }
    FIG_PAD = [0.01, 5]

    def __init__(self, eng_stress=None, eng_strain=None, true_stress=None,
                 true_strain=None, youngs_modulus=None, taylor_factor=None,
                 plastic_range=None, meta=None):

        msg = ('Specify (`eng_stress` and `eng_strain`) or (`true_stress` and '
               '`true_strain`)')
        if eng_stress is None and eng_strain is None:
            if true_strain is None or true_stress is None:
                raise ValueError(msg)
            if len(true_strain) != len(true_strain):
                raise ValueError('Stress and strain do not have the same length.')

        if true_stress is None and true_strain is None:
            if eng_stress is None or eng_strain is None:
                raise ValueError(msg)
            if len(eng_stress) != len(eng_strain):
                raise ValueError('Stress and strain do not have the same length.')

        self._eng_stress = self._prepare_array(eng_stress)
        self._eng_strain = self._prepare_array(eng_strain)

        self._true_stress = self._prepare_array(true_stress)
        self._true_strain = self._prepare_array(true_strain)

        self.youngs_modulus = youngs_modulus or TensileTest.DEFAULT_YOUNGS_MOD
        self.taylor_factor = taylor_factor or TensileTest.DEFAULT_TAYLOR_FACTOR

        self.plastic_range = plastic_range
        self._plastic_range_idx = None
        self._plastic_stress = None
        self._plastic_strain = None

        self._shear_stress = None
        self._shear_strain = None

        self.meta = meta or {}
        self._visual = None
        self._widgets = self._generate_widgets()

    def _prepare_array(self, arr):
        'Cast a list (with potentially `None` values) into a float array.'
        if arr is not None:
            arr = np.array(arr, dtype=np.float)
        return arr

    def to_dict(self):
        'Represent as a JSON-compatible dict.'

        out = {}
        # Only save one of eng/true stress/strain:
        if self._true_stress is not None:
            out.update({
                'true_stress': nan_array_to_list(self._true_stress),
                'true_strain': nan_array_to_list(self._true_strain),
            })
        elif self._eng_stress is not None:
            out.update({
                'eng_stress': nan_array_to_list(self._eng_stress),
                'eng_strain': nan_array_to_list(self._eng_strain),
            })

        out.update({
            'youngs_modulus': self.youngs_modulus,
            'taylor_factor': self.taylor_factor,
            'plastic_range': self.plastic_range,
            'meta': self.meta,
        })

        return out

    def to_json_file(self, json_path):
        'Save the TensileTest object to a JSON file'

        json_path = Path(json_path)
        dct = self.to_dict()
        with json_path.open('w') as handle:
            json.dump(dct, handle, sort_keys=True, indent=4)

        return json_path

    @classmethod
    def from_json_file(cls, json_path):
        'Load a TensileTest from a JSON file.'

        with Path(json_path).open() as handle:
            contents = json.load(handle)

        return cls(**contents)

    def __repr__(self):
        out = '{}()'.format(self.__class__.__name__)
        return out

    def __len__(self):
        return len(self.eng_stress)

    def _set_true_stress_strain(self):
        'Convert engineering stress/strain to true stress/strain.'
        tstress, tstrain = get_true_stress_strain(self.eng_stress, self.eng_strain)
        self._true_stress = tstress
        self._true_strain = tstrain

    def _set_eng_stress_strain(self):
        'Convert true stress/strain to engineering stress/strain.'
        estress, estrain = get_eng_stress_strain(self.true_stress, self.true_strain)
        self._eng_stress = estress
        self._eng_strain = estrain

    def _set_plastic_stress_strain(self):
        'Use `plastic_range` to set plastic stress/strain.'

        idx = [find_nearest_index(self.true_strain, self.plastic_range[i])
               for i in [0, 1]]

        stress_sub = self.true_stress[slice(*idx)]
        strain_sub = self.true_strain[slice(*idx)]

        elastic_strain = stress_sub / self.youngs_modulus_MPa
        self._plastic_strain = strain_sub - elastic_strain
        self._plastic_stress = stress_sub
        self._plastic_range_idx = idx

    def _set_shear_stress_strain(self):
        sstress, sstrain = self.get_shear_stress_strain(self.taylor_factor)
        self._shear_strain = sstrain
        self._shear_stress = sstress

    @property
    def youngs_modulus_MPa(self):
        return self.youngs_modulus * 1e3

    @property
    def youngs_modulus_SI(self):
        return self.youngs_modulus * 1e9

    @property
    def plastic_range(self):
        return self._plastic_range

    @plastic_range.setter
    def plastic_range(self, plastic_range):
        # Validate within min-max and start<stop, set default otherwise.
        if not plastic_range:
            plastic_range = [TensileTest.DEFAULT_PLASTIC_RANGE_START,
                             self.true_strain[int(0.8 * len(self.true_strain))]]

        msg = ('Plastic range must be a range between the minimum and maximum true '
               'strain values: {} {}'.format(self.min_true_strain, self.max_true_strain))
        if plastic_range[0] < self.min_true_strain:
            raise ValueError(msg)
        if plastic_range[1] > self.max_true_strain:
            raise ValueError(msg)

        self._plastic_range = plastic_range
        self._set_plastic_stress_strain()
        self._set_shear_stress_strain()

    @property
    def eng_stress(self):
        if self._eng_stress is None:
            self._set_eng_stress_strain()
        return self._eng_stress

    @property
    def eng_strain(self):
        if self._eng_strain is None:
            self._set_eng_stress_strain()
        return self._eng_strain

    @property
    def true_stress(self):
        if self._true_stress is None:
            self._set_true_stress_strain()
        return self._true_stress

    @property
    def true_strain(self):
        if self._true_strain is None:
            self._set_true_stress_strain()
        return self._true_strain

    @property
    def max_stress(self):
        return np.nanmax([self.max_eng_stress, self.max_true_stress])

    @property
    def min_stress(self):
        return np.nanmin([self.min_eng_stress, self.min_true_stress])

    @property
    def max_true_stress(self):
        return np.nanmax(self.true_stress)

    @property
    def min_true_stress(self):
        return np.nanmin(self.true_stress)

    @property
    def max_eng_stress(self):
        return np.nanmax(self.eng_stress)

    @property
    def min_eng_stress(self):
        return np.nanmin(self.eng_stress)

    @property
    def max_strain(self):
        return np.nanmax([self.max_eng_strain, self.max_true_strain])

    @property
    def min_strain(self):
        return np.nanmin([self.min_eng_strain, self.min_true_strain])

    @property
    def max_true_strain(self):
        return np.nanmax(self.true_strain)

    @property
    def min_true_strain(self):
        return np.nanmin(self.true_strain)

    @property
    def max_eng_strain(self):
        return np.nanmax(self.eng_strain)

    @property
    def min_eng_strain(self):
        return np.nanmin(self.eng_strain)

    @property
    def plastic_strain(self):
        if self._plastic_strain is None:
            self._set_plastic_stress_strain()
        return self._plastic_strain

    @property
    def plastic_stress(self):
        if self._plastic_stress is None:
            self._set_plastic_stress_strain()
        return self._plastic_stress

    @property
    def max_plastic_strain(self):
        return np.nanmax(self.plastic_strain)

    @property
    def min_plastic_strain(self):
        return np.nanmin(self.plastic_strain)

    @property
    def range_plastic_strain(self):
        return self.max_plastic_strain - self.min_plastic_strain

    @property
    def max_plastic_stress(self):
        return np.nanmax(self.plastic_stress)

    @property
    def min_plastic_stress(self):
        return np.nanmin(self.plastic_stress)

    @property
    def shear_stress(self):
        if self._shear_stress is None:
            self._set_shear_stress_strain()
        return self._shear_stress

    @property
    def shear_strain(self):
        if self._shear_strain is None:
            self._set_shear_stress_strain()
        return self._shear_strain

    @property
    def max_shear_strain(self):
        return np.nanmax(self.shear_strain)

    @property
    def min_shear_strain(self):
        return np.nanmin(self.shear_strain)

    @property
    def range_shear_strain(self):
        return self.max_shear_strain - self.min_shear_strain

    @property
    def max_shear_stress(self):
        return np.nanmax(self.shear_stress)

    @property
    def min_shear_stress(self):
        return np.nanmin(self.shear_stress)

    def get_shear_stress_strain(self, taylor_factor):
        'Use a Taylor factor to estimate single crystal shear strain/stress.'
        shear_strain = self.plastic_strain * taylor_factor
        shear_stress = self.plastic_stress / taylor_factor

        return shear_stress, shear_strain

    def _update_widgets_stress_strain_type(self, change):

        ss_type = self._widgets['stress_strain_type']
        fig = self._widgets['figure']

        with fig.batch_update():
            if ss_type.value == 'Engineering':
                stress = self.eng_stress
                strain = self.eng_strain
            else:
                stress = self.true_stress
                strain = self.true_strain

            fig.data[0].x = strain
            fig.data[0].y = stress
            fig.layout.xaxis.title.text = ss_type.value + ' strain, ε'
            fig.layout.yaxis.title.text = ss_type.value + ' stress, σ / MPa'

    def _generate_widgets(self):
        data = [
            {
                'x': self.eng_strain,
                'y': self.eng_stress,
                'name': self.meta.get('name') or '',
                'line': {
                    'color': DEFAULT_PLOTLY_COLORS[0],
                },
            },
        ]
        layout = {
            'title': 'Tensile test',
            'width': TensileTest.FIG_WIDTH,
            'height': TensileTest.FIG_HEIGHT,
            'margin': TensileTest.FIG_MARG,
            'xaxis': {
                'title': 'Engineering strain, ε',
                'range': [-TensileTest.FIG_PAD[0],
                          TensileTest.FIG_PAD[0] + self.max_strain],
            },
            'yaxis': {
                'title': 'Engineering stress, σ / MPa',
                'range': [-TensileTest.FIG_PAD[1] * 1e6,
                          TensileTest.FIG_PAD[1] * 1e6 + self.max_stress],
            },
        }
        fig_wig = go.FigureWidget(data=data, layout=layout)
        widget_ss_type = widgets.RadioButtons(
            options=['Engineering', 'True'],
            description='Stress/strain:',
            value='Engineering',
        )
        widget_ss_type.observe(self._update_widgets_stress_strain_type, names='value')
        widget_dict = {
            'figure': fig_wig,
            'stress_strain_type': widget_ss_type,
        }
        return widget_dict

    def _generate_visual(self):

        out = widgets.HBox(
            children=[
                self._widgets['figure'],
                self._widgets['stress_strain_type']
            ]
        )
        return out

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
        self._widgets['stress_strain_type'].value = ss_type[stress_strain_type]
        return self.visual
