# PINNs-Plug-n-Play-Integration

This library provides a Differential-Algebraic system of Equations (DAEs) solvers implemented in PyTorch based on the Simultaneous-Implicit method. The developed algorithms, based on Runge-Kutta integration schemes, allow the integration of Physics-Informed Neural Networks, boosting their computation performance and unlocking a wide range of modelling and privacy opportunities.

The motivation, methodology and applications are discussed in the following paper:

Ignasi Ventura and Jochen Stiasny and Spyros Chatzivasileiadis. "Physics-Informed Neural Networks: a Plug and Play Integration into Power System Dynamic Simulations". 2024.

## Installation

To install latest on GitHub:

```
pip install git+https://github.com/ignvenad/PINNs-Plug-n-Play-Integration
```

## Examples
We encourage those who are interested in using this library to run the [main.py](https://github.com/ignvenad/PINNs-Plug-n-Play-Integration/blob/main/main.py) with the default parameters and the ones reproduced in the paper and stated below.

The simulation parameters invoked with [main.py](https://github.com/ignvenad/PINNs-Plug-n-Play-Integration/blob/main/main.py):

- `--system` is the studied system in the simulation. For now, only the IEEE 9-bus test system is supported.
- `--machine` denotes the generator that can be modelled with a PINN.
- `--event_type` specifies which type of contingency is to be considered.
- `--event_location` fixes the location of the contingency.
- `--event_magnitude` specifies the contingency's magnitude.
- `--sim_time` is the simulation time.
- `--time_step_size` is the time step used in the Simulataneous-Implicit algorithm.
- `--rk_scheme` represents the integration scheme for the differential equations.
- `--compare_pure_RKscheme` flag only if we want to compare the hybrid and pure Runge-Kutta solvers.
- `--compare_ground_truth` flag only if we want to see the ground truth simulated with commercial software. 
- `--study_selection` to state which type of developed study is preferred.
- `--gpu` requires access to the gpu for inference.

#### Supported Runge-Kutta schemes:
- `trapezoidal` Trapezoidal rule.
- `backward_euler` Backward Euler or Implicit Euler method.

#### Supported network contingencies:
- `w_setpoint` Relative rotor-angle speed &Delta; &omega; steps.
- `p_setpoint` Mechanical power output &#80; steps.

## Basic Usage
This library provides one main interface `main` which contains all the information required to run the desired DAE simulations.

The `main` file leverages two main scripts:
- [PINN_architecture.py](https://github.com/ignvenad/PINNs-Plug-n-Play-Integration/blob/main/src/pinn_architecture.py) to define the fully-connected neural network.
- [tds_dae_rk_schemes.py](https://github.com/ignvenad/PINNs-Plug-n-Play-Integration/blob/main/src/tds_dae_rk_schemes.py) which contains the Simultaneous-Implicit algorithm and the PINN integration.

`post_processing` folder contains the plotting capabilities of this repository, `gt_simulations` the used ground truth simulations obtained with conventional software, `final_models` a trained PINN for each dynamical component of the test system, and `config_files` the used network and dynamical component parameters.

To boost the Simultaneous-Implicit algorithm with PINNs, the `tds_dae_rk_schemes.TDS_simulation` needs to activate the `pinn_boost` flag specifying the component modelled with a PINN, as long as the weights and operational limits of such PINN, `pinn_weights` and `pinn_ops_limits` respectively. These inputs are to be changed in the `main` file with either the released PINNs or imported PINNs.

## Reproducibility

The results published in the pre-print can be attained with the following parameters.

#### Overview figure `--study_selection` = 1
Depicts all simulation variables for the specific case and failure studied.

#### Figure 2 with `--time_step_size` = 8e-3, `--sim_time` = 10, and `--study_selection` = 2
Depicts Figure 4 from the manuscript.

#### Figure 3 with `--time_step_size` = 4e-2, `--sim_time` = 2, and `--study_selection` = 2
Depicts Figure 5 from the manuscript.

#### Figure 4 with `--time_step_size` = [1e-3:4e-2] and `--study_selection` = 3
Depicts Figure 6 from the manuscript.

#### Figure 5 with `--time_step_size` = 1e-2 and `--study_selection` = 4
Depicts Figure 7 from the manuscript.

#### Figure 6 with `--time_step_size` = [1e-3:4e-2] and `--study_selection` = 5
Depicts Figure 8 from the manuscript.


## References

The motivation, methodology and applications are discussed in the following paper. If you find this work helpful, please cite this work:

TBD

## License

This project is available under the MIT License.
