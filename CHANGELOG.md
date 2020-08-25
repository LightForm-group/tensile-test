# Change Log

## [0.1.3] - 2020.08.25

### Changed

- Perform LM fit up to upper plastic range of experimental data only.

## [0.1.2] - 2019.09.11

### Added

- Added property `fitting_params` to `LMFitterOptimisation` to show the values of the fitting parameters at a given iteration.

### Changed

- Sims are hidden by default in `LMFitter` visualisation.
- Allow skipping directory validation with `ignore_missing_dirs` argument to `LMFitter.from_json_file`, which allows loading the JSON file on another computer.

## [0.1.1] - 2019.09.08

### Added

- Support saving and loading a `TensileTest` from a JSON file.
- Added a class `LMFitter` for fitting experimental tensile tests using a Levenberg-Marquardt optimisation process.

## [0.1.0] - 2019.09.03

- Initial release.
