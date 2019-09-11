'Fitting crystal plasticity parameters using a Levenberg-Marquardt optimisation process.'

import copy
import json
from pathlib import Path
from importlib import import_module

import numpy as np
from plotly import graph_objects

from tensile_test.tensile_test import TensileTest
from tensile_test.utils import find_nearest_index


class FittingParameter(object):
    'Represents a parameter to be fit and its fitted values.'

    def __init__(self, name, values, address, perturbation):

        self.name = name
        self.address = address
        self.perturbation = perturbation

        self.values = np.array(values)

    def to_dict(self):
        'Represent as a JSON-compatible dict.'
        out = {
            'name': self.name,
            'address': self.address,
            'perturbation': self.perturbation,
            'values': self.values.tolist(),
        }
        return out

    def to_json_file(self, json_path):
        'Save as a JSON file'

        json_path = Path(json_path)
        dct = self.to_dict()
        with json_path.open('w') as handle:
            json.dump(dct, handle, sort_keys=True, indent=4)

        return json_path

    @classmethod
    def from_json_file(cls, json_path):
        'Load from a JSON file.'

        with Path(json_path).open() as handle:
            contents = json.load(handle)

        return cls(**contents)

    @property
    def initial_value(self):
        return self.values[0]

    def get_perturbation(self, idx=-1):
        return self.values[idx] * self.perturbation

    def get_perturbed_value(self, idx=-1):
        return self.values[idx] + self.get_perturbation(idx)

    def __repr__(self):
        out = (
            '{}('
            'name={!r}, '
            'values={!r}'
            ')').format(
            self.__class__.__name__,
            self.name,
            self.values,
        )
        return out


class LMFitterOptimisation(object):
    'Represents a single optimisation step in the LMFitter object.'

    def __init__(self, lm_fitter, sim_tensile_tests, _damping_factor=None, _error=None,
                 _delay_validation=False):

        # TODO: probably better to use __new__ somehow instead of this nonsense:
        if not _delay_validation:
            self.lm_fitter = lm_fitter
            self.sim_tensile_tests = self._validate_tensile_tests(sim_tensile_tests)

            exp_strain_idx, sim_strain_idx = self._get_strain_idx()

            self.exp_strain_idx = exp_strain_idx
            self.sim_strain_idx = sim_strain_idx
            self._jacobian = self._approximate_jacobian()
        else:
            self.sim_tensile_tests = sim_tensile_tests

        # These are assigned by `find_new_parameters`:
        self._damping_factor = _damping_factor
        self._error = _error

    def _validate(self, lm_fitter):

        self.lm_fitter = lm_fitter
        self.sim_tensile_tests = self._validate_tensile_tests(self.sim_tensile_tests)

        exp_strain_idx, sim_strain_idx = self._get_strain_idx()

        self.exp_strain_idx = exp_strain_idx
        self.sim_strain_idx = sim_strain_idx
        self._jacobian = self._approximate_jacobian()

    def to_dict(self):
        'Represent as a JSON-compatible dict.'
        out = {
            'sim_tensile_tests': [i.to_dict() for i in self.sim_tensile_tests],
            '_damping_factor': self.damping_factor,
            '_error': self.error,
        }
        return out

    def __repr__(self):
        out = (
            '{}('
            'error={}, '
            'damping_factor={}'
            ')'
        ).format(
            self.__class__.__name__,
            self.error,
            self.damping_factor,
        )
        return out

    @property
    def error(self):
        return self._error

    @property
    def damping_factor(self):
        return self._damping_factor

    @property
    def trial_damping_factors(self):
        if self.index > 0:
            prev_opt = self.lm_fitter.optimisations[self.index - 1]
            prev_damp = prev_opt.damping_factor
            out = [2 * prev_damp, prev_damp, 0.5 * prev_damp]
        else:
            out = self.lm_fitter.initial_damping

        return out

    @property
    def index(self):
        'Get the index of this optimisation step.'
        if self in self.lm_fitter.optimisations:
            # Already added to opt list:
            return self.lm_fitter.optimisations.index(self)
        else:
            # Not yet added to opt list:
            return len(self.lm_fitter.optimisations)

    def _validate_tensile_tests(self, sim_tensile_tests):
        'Check the correct number of tensile tests.'

        if len(sim_tensile_tests) != self.lm_fitter.sims_per_iteration:
            msg = ('There must be {} new tensile tests to add an optimisation '
                   'step, since there are {} fitting parameters.')
            raise ValueError(msg.format(self.lm_fitter.sims_per_iteration,
                                        self.lm_fitter.num_params))

        return sim_tensile_tests

    def get_exp_stress(self):
        return self.lm_fitter.exp_tensile_test.true_stress[self.exp_strain_idx]

    def get_exp_strain(self):
        return self.lm_fitter.exp_tensile_test.true_strain[self.exp_strain_idx]

    def get_sim_stress(self, sim_idx):
        return self.sim_tensile_tests[sim_idx].true_stress[self.sim_strain_idx[sim_idx]]

    def get_sim_strain(self, sim_idx):
        return self.sim_tensile_tests[sim_idx].true_strain[self.sim_strain_idx[sim_idx]]

    def _get_strain_idx(self):
        """Use the first simulated tensile test (unperturbed fitting parameters) to define
        the strain increments used in the Jacobian approximation."""

        first_tt = self.sim_tensile_tests[0]
        exp_tt = self.lm_fitter.exp_tensile_test

        exp_strain_idx = []
        sim_strain_idx = [[] for _ in range(len(self.sim_tensile_tests))]
        for i_idx, strain_val in enumerate(first_tt.true_strain):

            exp_strain_idx.append(find_nearest_index(exp_tt.true_strain, strain_val))
            sim_strain_idx[0] = np.arange(len(first_tt.true_strain))

            for j_idx, sim_tt in enumerate(self.sim_tensile_tests[1:], 1):
                sim_strain_idx[j_idx].append(
                    find_nearest_index(sim_tt.true_strain, strain_val))

        exp_strain_idx = np.array(exp_strain_idx)
        sim_strain_idx = np.array(sim_strain_idx)

        return exp_strain_idx, sim_strain_idx

    def _approximate_jacobian(self):
        """Use the base and perturbed simulated tensile tests to approximate the Jacobian matrix.

        Notes
        -----
        Each row in the Jacobian matrix contains stress values from different strain
        increments. Each column in the Jacobian matrix represents the stress results
        from a simulation with a particular fitting parameter perturbed, minus the
        stress from the unperturbed simulation.

        """

        # Get stresses from perturbed sims:
        cols = [self.get_sim_stress(i) for i in range(1, self.lm_fitter.num_params + 1)]
        cols = np.vstack(cols).T

        # Subtract off stress from unperturbed sim:
        cols -= self.get_sim_stress(0)[:, None]

        perts = np.array([i.get_perturbation(self.index)
                          for i in self.lm_fitter.fitting_params])
        jacobian = cols / perts

        return jacobian

    @property
    def fitting_params(self):
        out = {}
        for param in self.lm_fitter.fitting_params:
            out.update({param.name: param.values[self.index]})
        return out

    @property
    def jacobian(self):
        return self._jacobian

    def find_new_parameters(self):

        deltas = []
        errors = []

        jac_prod = self.jacobian.T @ self.jacobian
        right = self.jacobian.T @ (self.get_exp_stress() -
                                   self.get_sim_stress(0))[:, None]

        for damping in self.trial_damping_factors:

            left = jac_prod + damping * np.diag(jac_prod)
            delta = np.linalg.solve(left, right)
            stress_diff = (self.get_exp_stress() - self.get_sim_stress(0))[:, None]
            error = np.sum(np.abs(stress_diff - self.jacobian @ delta))
            deltas.append(delta)
            errors.append(error)

        best_idx = np.argmin(errors)
        best_delta = deltas[best_idx]

        self._damping_factor = self.trial_damping_factors[best_idx]
        self._error = errors[best_idx]

        cur_params = [i.values[-1] for i in self.lm_fitter.fitting_params]
        new_params = [i + j for i, j in zip(cur_params, best_delta)]

        return new_params


class LMFitter(object):

    FIG_WIDTH = 480
    FIG_HEIGHT = 380
    FIG_MARG = {
        't': 80,
        'l': 60,
        'r': 50,
        'b': 80,
    }
    FIG_PAD = [0.01, 5]

    def __init__(self, exp_tensile_test, material_params, fitting_params, inputs_writer,
                 base_sim_dir=None, sim_dir=None, initial_damping=None, optimisations=None):
        """Use a Levenberg-Marquardt optimisation process to find crystal plasticity
        hardening parameters.

        Parameters
        ----------
        exp_tensile_test : list of TensileTest
            The experimental data to which the hardening parameters should be fit.            
        material_params : dict
            Dict of parameters that define the material properties in whatever simulation
            software is chosen. This dict is passed to `input_file_func` to generate the
            necessary input files.
        fitting_params : dict of (str: list)
            A dict whose keys are the names of parameters that are to be fit, and whose
            values are lists that determine the "address" of each parameter in the
            `material_params` dict, since, in general, the `material_params` dict may be
            arbitrarily nested.
        base_sim_dir : str or Path
            Directory path in which simulation inputs files reside that are the same for
            all optimisation simulations. When generating input files, any files found in
            this directory are copied to the new simulation directory, along with the
            input files associated with the `material_params` dict.
        sim_dir : str or Path
            Directory path in which simulation inputs files will be written.
        inputs_writer : dict
            Dict that encodes which function in which module should be invoked to generate
            the simulation input files. It has the following keys and values:
                function: str
                    The name of the function that is defined in the `module` module that
                    writes the simulation input file.            
                dir_path_arg : str
                    The argument name of `function` that determines the directory path
                    into which the simulation inputs files are written.                    
                module : str, optional
                    The name of the module in which the input file writer function is
                    found. This module is imported using the standard library
                    `importlib.import_module`. If not specified, the function is expected
                    to be directly invokable.

        """

        self.exp_tensile_test = exp_tensile_test
        self.optimisations = optimisations or []

        self.base_material_params = material_params
        self.fitting_params = self._validate_fitting_params(fitting_params)
        self.inputs_writer = self._validate_inputs_writer(inputs_writer)
        self.initial_damping = initial_damping or [2, 1, 0.5]

        base_sim_dir, sim_dir = self._validate_dirs(base_sim_dir, sim_dir)
        self._base_sim_dir = str(base_sim_dir)
        self._sim_dir = str(sim_dir)

        self._visual = None

    def _validate_fitting_params(self, fitting_params):
        'Check the correct numpy of fitting parameter values'
        for i in fitting_params:
            if len(i.values) != len(self.optimisations) + 1:
                msg = ('If there are N optimisations, each fitting parameter '
                       'must have N-1 values,')
                raise ValueError(msg)

        return fitting_params

    def to_dict(self):
        'Represent as a JSON-compatible dict.'
        out = {
            'exp_tensile_test': self.exp_tensile_test.to_dict(),
            'fitting_params': [i.to_dict() for i in self.fitting_params],
            'optimisations': [i.to_dict() for i in self.optimisations],
            'material_params': self.base_material_params,
            'inputs_writer': self.inputs_writer,
            'initial_damping': self.initial_damping,
            'base_sim_dir': self._base_sim_dir,
            'sim_dir': self._sim_dir,
        }
        return out

    def to_json_file(self, json_path):
        """Save the state of the LMFitter object to a JSON file, to allow continuation of
        the fitting process at a later date."""

        json_path = Path(json_path)
        dct = self.to_dict()
        with json_path.open('w') as handle:
            json.dump(dct, handle, sort_keys=True, indent=4)

        return json_path

    @classmethod
    def from_json_file(cls, json_path):
        'Load an LMFitter from a JSON file.'

        with Path(json_path).open() as handle:
            contents = json.load(handle)

        contents['exp_tensile_test'] = TensileTest(**contents['exp_tensile_test'])

        contents['fitting_params'] = [
            FittingParameter(**i)
            for i in contents['fitting_params']
        ]

        for idx, i in enumerate(contents['optimisations']):
            i.update({
                'lm_fitter': None,
                '_delay_validation': True,
                'sim_tensile_tests': [TensileTest(**j) for j in i['sim_tensile_tests']]
            })
            contents['optimisations'][idx] = LMFitterOptimisation(**i)

        lm_fitter = cls(**contents)
        for opt in lm_fitter.optimisations:
            opt._validate(lm_fitter)

        return lm_fitter

    @property
    def opt_index(self):
        return len(self.optimisations)

    @property
    def base_sim_dir(self):
        return Path(self._base_sim_dir)

    @property
    def sim_dir(self):
        return Path(self._sim_dir)

    @property
    def num_params(self):
        return len(self.fitting_params)

    @property
    def sims_per_iteration(self):
        return self.num_params + 1

    @property
    def inputs_writer_func(self):
        'Get the function (object) to write the input files.'
        func = getattr(
            import_module(self.inputs_writer['module']),
            self.inputs_writer['function'],
        )
        return func

    def get_parameter(self, name, address):
        'Get a parameter value from the `material_params` dict.'
        out = self.base_material_params
        for i in address:
            out = out[i]
        return out[name]

    def set_parameter(self, dct, name, address, value):
        for i in address:
            dct = dct[i]
        dct[name] = value

    def get_material_params(self, fitting_idx):
        'Get a list of material params dict for a given optimisation.'

        base_params = copy.deepcopy(self.base_material_params)
        for fit_param in self.fitting_params:
            self.set_parameter(
                base_params,
                fit_param.name,
                fit_param.address,
                fit_param.values[fitting_idx]
            )
        all_params = [base_params]

        for fit_param in self.fitting_params:
            perturbed_params = copy.deepcopy(base_params)
            self.set_parameter(
                perturbed_params,
                fit_param.name,
                fit_param.address,
                fit_param.get_perturbed_value(fitting_idx),
            )
            all_params.append(perturbed_params)

        return all_params

    def _validate_dirs(self, *dirs):
        paths = []
        for i in dirs:
            i = Path(i)
            if not i.is_dir():
                msg = '"{}" is not a directory.'.format(i.resolve())
                raise ValueError(msg)
            paths.append(i)
        return paths

    def _validate_inputs_writer(self, inputs_writer):

        msg = ('`inputs_writer` must be a dict with keys: `function`, `dir_path_arg`'
               ' and `module` (optional).')

        if not isinstance(inputs_writer, dict):
            raise ValueError(msg)

        req_keys = ['function']
        allowed_keys = set(req_keys + ['function_args', 'module'])
        req_keys = set(req_keys)

        if (req_keys & set(inputs_writer.keys())) != req_keys:
            raise ValueError(msg)
        if (set(inputs_writer.keys()) & allowed_keys) != set(inputs_writer.keys()):
            raise ValueError(msg)

        if inputs_writer.get('module') is None:
            inputs_writer['module'] = __name__

        # Check function is importable:
        try:
            _ = getattr(import_module(inputs_writer['module']), inputs_writer['function'])
        except ModuleNotFoundError:
            msg = 'Could not find the input file writer function "{}" in module "{}".'
            raise ValueError(msg.format(
                inputs_writer['function'], inputs_writer['module']))

        return inputs_writer

    def __repr__(self):
        out = ('{}('
               'fitting_params={!r}'
               ')').format(
            self.__class__.__name__,
            self.fitting_params,
        )
        return out

    def _prepare_func_args(self, unparsed_args, params, sim_dir):

        func_args = {}
        for arg, val in unparsed_args.items():

            if isinstance(val, str):
                if '<<SIM_DIR>>' in val:
                    val = val.replace('<<SIM_DIR>>', str(sim_dir))
                if val == '<<PARAMETERS>>':
                    val = params
            elif isinstance(val, dict):
                val = self._prepare_func_args(val, params, sim_dir)

            func_args.update({arg: val})

        return func_args

    def generate_simulation_inputs(self):
        'Generate the next set of simulation input files.'

        all_params = self.get_material_params(self.opt_index)

        # Generate new simulation directories:
        sim_dirs = []
        com = self.opt_index * self.sims_per_iteration
        sim_dir_range = range(com, com + self.sims_per_iteration)
        for idx, i in enumerate(sim_dir_range):
            sim_dir = self.sim_dir.joinpath('{}'.format(i))
            sim_dir.mkdir(exist_ok=False, parents=False)
            sim_dirs.append(sim_dir)

            # Copy base input files:
            for j in self.base_sim_dir.glob('*'):
                if j.is_file():
                    dest = sim_dir.joinpath(j.name)
                    dest.write_bytes(j.read_bytes())

            # Prepare function args:
            func_args = self._prepare_func_args(
                self.inputs_writer['function_args'],
                all_params[idx],
                sim_dir,
            )
            # Invoke function
            self.inputs_writer_func(**func_args)

    def add_simulated_tensile_tests(self, tensile_tests):
        """Add simulation results to progress the optimisation process.

        Parameters
        ----------
        tensile_tests : list of TensileTest
            For M fitting parameters, this list must be of length M + 1.
            The order of TensileTests should match the order of simulation
            input files written by `generate_simulation_inputs`.

        """

        opt = LMFitterOptimisation(self, tensile_tests)
        self.optimisations.append(opt)
        new_params = opt.find_new_parameters()
        for i, j in zip(self.fitting_params, new_params):
            i.values = np.append(i.values, j)

    def _generate_visual(self):

        data = [{
            'x': self.optimisations[0].get_exp_strain(),
            'y': self.optimisations[0].get_exp_stress() / 1e6,
            'mode': 'lines',
            'name': 'Exp.',
            'line': {
                'dash': 'dash',
            }
        }]
        data.extend([
            {
                'x': opt.get_sim_strain(0),
                'y': opt.get_sim_stress(0) / 1e6,
                'mode': 'lines',
                'name': 'Sim. {}'.format(idx)
            } for idx, opt in enumerate(self.optimisations, 1)
        ])

        layout = {
            'title': 'Levenberg-Marquardt optimisation',
            'width': LMFitter.FIG_WIDTH,
            'height': LMFitter.FIG_HEIGHT,
            'margin': LMFitter.FIG_MARG,
            'xaxis': {
                'title': 'True strain, ε',
                'range': [
                    -LMFitter.FIG_PAD[0],
                    self.optimisations[0].get_exp_strain()[-1] + LMFitter.FIG_PAD[0]
                ],
            },
            'yaxis': {
                'title': 'True stress σ / MPa',
            }
        }

        fig = graph_objects.FigureWidget(data=data, layout=layout)

        return fig

    def show(self):
        if not self._visual:
            self._visual = self._generate_visual()
        return self._visual
