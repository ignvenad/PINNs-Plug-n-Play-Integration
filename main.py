import os
import argparse
import time
import yaml
import numpy as np

import torch

from src.tds_dae_rk_schemes import TDS_simulation

import src.PINN_architecture

from post_processing.trajectories_overview_plot import trajectories_overview
from post_processing.custom_overview_plots import custom_overview1, custom_overview2

parser = argparse.ArgumentParser('TDS-Simulation')
parser.add_argument('--system', type=str, choices=['ieee9bus'], default='ieee9bus')
parser.add_argument('--machine', type=int, choices=[1, 2, 3], default=3)
parser.add_argument('--event_type', choices=['p_setpoint', 'w_setpoint'], default='w_setpoint')
parser.add_argument('--event_location', type=int, choices=[1, 2, 3], default=3)
parser.add_argument('--event_magnitude', type=float, default=1e-2)
parser.add_argument('--sim_time', type=float, default=2.)
parser.add_argument('--time_step_size', type=float, default=4e-2)
parser.add_argument('--rk_scheme', choices=['trapezoidal', 'backward_euler'], default='trapezoidal')
parser.add_argument('--compare_pure_RKscheme', action='store_true')
parser.add_argument('--compare_ground_truth', action='store_true', default=True)
parser.add_argument('--study_selection', choices=[1, 2, 3, 4, 5], default=2)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


if args.compare_ground_truth:
    gt_dir = './gt_simulations/'
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)

    if not os.path.isfile(file_npy):
        args.compare_ground_truth = False

def config_file(yaml_file) -> dict:
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def extract_values_dynamic_components(config_file) -> tuple:
    freq     = config_file['freq']
    H        = list(config_file['inertia_H'].values())
    Rs       = list(config_file['Rs'].values())
    Xd_p     = list(config_file['Xd_prime'].values())
    pg_pf    = list(config_file['Pg_setpoints'].values())
    dampings = list(config_file['Damping_D'].values())
    return freq, H, Rs, Xd_p, pg_pf, dampings

def extract_values_static_components(config_file) -> tuple:
    voltages_magnitude = list(config_file['Voltage_magnitude'].values())
    voltages_angles    = list(config_file['Voltage_angle'].values())
    Xd                 = list(config_file['Xd'].values())
    Xq                 = list(config_file['Xq'].values())
    Xq_prime           = list(config_file['Xq_prime'].values())
    voltages_complex   = np.array(voltages_magnitude)*np.exp(1j*np.array(voltages_angles)*np.pi/180)

    return voltages_complex, Xd, Xq, Xq_prime

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

def load_pinn_machine(pinn_model):
    loaded_pinn = torch.load(pinn_model, map_location=device)
    norm_range, lb_range                     = loaded_pinn['range_norm']
    _, num_neurons, num_layers, inputs, outputs = loaded_pinn['architecture']
    pinn_integrated = src.PINN_architecture.FCN_RESNET(inputs,outputs,num_neurons,num_layers, norm_range, lb_range)
    pinn_integrated.load_state_dict(loaded_pinn['state_dict'])
    pinn_integrated.eval()
    return pinn_integrated

def load_pinn_parameters(pinn_model) -> tuple:
    loaded_pinn = torch.load(pinn_model, map_location=device)
    damping_pinn, pg_pinn, H_pinn, Xd_pinn, _ = loaded_pinn['machine_parameters']
    return H_pinn, Xd_pinn, pg_pinn, damping_pinn

def load_ini_conditions_true(start_value):
    gt_dir = './gt_simulations/'
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)
    states_assimulo = np.load(file_npy)
    return torch.tensor(states_assimulo[start_value, :-1], dtype=torch.float64)

def load_ini_conditions_true_option4(start_value):
    gt_dir = './gt_simulations/'
    file_name = f'sim10s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)
    states_assimulo = np.load(file_npy)
    return torch.tensor(states_assimulo[start_value, :-1], dtype=torch.float64)

def return_true_solution() -> tuple:
    gt_dir = './gt_simulations/'
    file_name = f'sim{int(args.sim_time)}s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)
    states_assimulo = np.load(file_npy)
    return states_assimulo[:, 30], states_assimulo[:, :-1]

def return_true_solution_option4(start_point=0, end_point=-1) -> tuple:
    gt_dir = './gt_simulations/'
    file_name = f'sim10s_{args.event_type}_{args.event_location}.npy'
    file_npy = os.path.join(gt_dir, file_name)
    states_assimulo = np.load(file_npy)
    return states_assimulo[start_point:end_point, 30], states_assimulo[start_point:end_point, :-1]

def compute_pinn_ops_limits(pinn_model) -> tuple:
    loaded_pinn = torch.load(pinn_model, map_location=device)
    voltage_limits = loaded_pinn['voltage_stats'][0]
    theta_limits   = loaded_pinn['theta_stats'][0]
    delta_limits   = loaded_pinn['init_state'][2]
    omega_limits   = loaded_pinn['init_state'][3]
    return voltage_limits, theta_limits, delta_limits, omega_limits

def compute_time_step_assimulo(time_array_assimulo):
    time_step_assimulo = round(time_array_assimulo[1] - time_array_assimulo[0], 8)
    for i in np.random.randint(0, len(time_array_assimulo), size=10).tolist():
        assert round(time_array_assimulo[i] - time_array_assimulo[i-1], 8) == time_step_assimulo
    return time_step_assimulo

def compute_time_step_ratio(time_step, time_step_assimulo):
    assert time_step >= time_step_assimulo
    ratio_trapz_assimulo = time_step/time_step_assimulo
    assert ratio_trapz_assimulo % 1 == 0
    ratio_trapz_assimulo = round(ratio_trapz_assimulo)
    return ratio_trapz_assimulo

def double_check_ratio_errors(t_array_trapz, t_array_assimulo, ratio_trapz_assimulo):
    check_time_steps_correct = np.random.randint(0, len(t_array_trapz)-1, size=15).tolist()
    for i in check_time_steps_correct:
        if not round(t_array_trapz[i], 8) == round(t_array_assimulo[i*ratio_trapz_assimulo], 8):
            return False
    return True

def errors_analysis(t_sim_array, studied_states, t_sim_assimulo, states_array_assimulo, ratio_trapz_assimulo, error_type= 'abs'):
    assert double_check_ratio_errors(t_sim_array, t_sim_assimulo, ratio_trapz_assimulo)
    assert np.array_equal(states_array_assimulo[0, :], studied_states[0, :])
    number_of_checks = studied_states.shape[0] - 1
    states_to_check_errors = [2, 3, 8, 9, 12, 13, 18, 19, 22, 23, 28, 29] # state variables and terminal voltages of every machine
    errors_sim_array = np.ones((number_of_checks, len(states_to_check_errors)))
    for i in range(number_of_checks):
        for ind, state in enumerate(states_to_check_errors):
            if state in [2, 12, 22]:
                value_for_error_study      = studied_states[i+1, state] - studied_states[i+1, state+7] 
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, state] - states_array_assimulo[(i+1)*ratio_trapz_assimulo, state+7]
                current_error  = value_for_error_study - true_value_for_error_study
            elif state == 9:
                value_for_error_study      = studied_states[i+1, state] - studied_states[i+1, 19] 
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, state] - states_array_assimulo[(i+1)*ratio_trapz_assimulo, 19]
                current_error  = value_for_error_study - true_value_for_error_study
            elif state == 19:
                value_for_error_study      = studied_states[i+1, 9] - studied_states[i+1, 29] 
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, 9] - states_array_assimulo[(i+1)*ratio_trapz_assimulo, 29]
                current_error  = value_for_error_study - true_value_for_error_study
            elif state == 29:
                value_for_error_study      = studied_states[i+1, 19] - studied_states[i+1, 29] 
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, 19] - states_array_assimulo[(i+1)*ratio_trapz_assimulo, 29]
                current_error  = value_for_error_study - true_value_for_error_study
            else:
                value_for_error_study      = studied_states[i+1, state]
                true_value_for_error_study = states_array_assimulo[(i+1)*ratio_trapz_assimulo, state]
                current_error  = value_for_error_study - true_value_for_error_study
            if error_type == 'plot_dif':
                errors_sim_array[i, ind] = current_error
            elif error_type == 'abs':
                errors_sim_array[i, ind] = abs(current_error)
            elif error_type == 'percentage':
                errors_sim_array[i, ind] = abs(current_error / true_value_for_error_study) * 100
    return errors_sim_array

if __name__ == "__main__":

    pinn_directory = './final_models/'
    pinn_name  = f'model_DAE_machine_{args.machine}.pth'

    pinn_location = os.path.join(pinn_directory, pinn_name)

    simulation_pinn = load_pinn_machine(pinn_location)

    parameters_dc_raw = config_file('./config_files/config_machines_dynamic.yaml')
    freq, H, Rs, Xd_p, pg_pf, dampings = extract_values_dynamic_components(parameters_dc_raw)
    assert len(H)     == 3; assert len(Rs)       == 3; assert len(Xd_p) == 3
    assert len(pg_pf) == 3; assert len(dampings) == 3

    parameters_pinn = load_pinn_parameters(pinn_location)

    assert H[args.machine-1] == parameters_pinn[0]
    assert Xd_p[args.machine-1] == parameters_pinn[1]
    assert pg_pf[args.machine-1] == parameters_pinn[2]
    assert dampings[args.machine-1] == parameters_pinn[3]

    assert all(damp > 0 for damp in dampings)
    assert all(inertia > 0 for inertia in H)

    pinn_ops_limits = compute_pinn_ops_limits(pinn_location)

    Yadmittance = torch.load('./config_files/network_admittance.pt')

    parameters_initialization = config_file('./config_files/config_machines_static.yaml')
    volt, Xd, Xq, Xq_p = extract_values_static_components(parameters_initialization)
    ini_cond_sim = load_ini_conditions_true(start_value=0)

    assert args.sim_time > 0.
    assert args.time_step_size > 0.
    t_final_simulations = args.sim_time
    step_size_pure_rk = args.time_step_size
    step_size_hybrid_rk = args.time_step_size

    if args.study_selection == 1:
        solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_pure_rk)
        t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
        solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_hybrid_rk, 
                                            pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
        t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
        t_true, states_true = return_true_solution()
        plotting = trajectories_overview(args.sim_time, t_evo_pinn, states_evo_pinn, t_test_pure_rk, states_evo, t_true, states_true)
        plotting.compute_results(pure_rk_scheme=True, assimulo_states=True)
        plotting.show_results(save_fig=False)
    
    elif args.study_selection == 2:
        solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_pure_rk)
        t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
        solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_hybrid_rk, 
                                            pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
        t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
        t_true, states_true = return_true_solution()
        time_step_assimulo = compute_time_step_assimulo(t_true)
        ratio_trapz_assimulo = compute_time_step_ratio(args.time_step_size, time_step_assimulo)
        errors_pure_array = errors_analysis(t_test_pure_rk, states_evo, t_true, states_true, ratio_trapz_assimulo)
        errors_pinn_array = errors_analysis(t_evo_pinn, states_evo_pinn, t_true, states_true, ratio_trapz_assimulo)
        plotting = custom_overview1(args.sim_time, t_test_pure_rk, errors_pure_array, t_evo_pinn, errors_pinn_array)
        plotting.trajectory_and_errors_plot(8, 10, t_true, states_true, states_evo, states_evo_pinn)
        plotting.show_results()

    elif args.study_selection == 3:
        timesteps_to_study = [5e-3, 8e-3, 0.01, 0.02, 0.025, 0.04]
        maximums_pure = np.ones((len(timesteps_to_study), 2))
        maximums_hybrid = np.ones((len(timesteps_to_study), 2))
        for i, c_timestep in enumerate(timesteps_to_study):
            solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=c_timestep)
            t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
            solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=c_timestep, 
                                                pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
            t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
            t_true, states_true = return_true_solution()
            time_step_assimulo = compute_time_step_assimulo(t_true)
            ratio_trapz_assimulo = compute_time_step_ratio(c_timestep, time_step_assimulo)
            errors_pure_array = errors_analysis(t_test_pure_rk, states_evo, t_true, states_true, ratio_trapz_assimulo)
            errors_pinn_array = errors_analysis(t_evo_pinn, states_evo_pinn, t_true, states_true, ratio_trapz_assimulo)
            maximums_pure[i, :] = [np.max(errors_pure_array[:, 8]), np.max(errors_pure_array[:, 10])]
            maximums_hybrid[i, :] = [np.max(errors_pinn_array[:, 8]), np.max(errors_pinn_array[:, 10])]
            print(i+1, len(timesteps_to_study))
        plotting = custom_overview2(timesteps_to_study)
        plotting.show_results(maximums_pure, maximums_hybrid)

    elif args.study_selection == 4:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0, 0])
        ax0.set_title('Delta_Theta Gen. 3')
        ax0.grid()
        ax1 = plt.subplot(gs[0, 1])
        ax1.set_title('Voltage magnitude Gen. 3')
        ax1.grid()
        t_true_complete, states_true_complete = return_true_solution_option4()
        time_step_assimulo = compute_time_step_assimulo(t_true_complete)
        ratio_trapz_assimulo = compute_time_step_ratio(step_size_hybrid_rk, time_step_assimulo)
        start_ini_cond = np.random.randint(0, len(t_true_complete)-1000, size=30).tolist()
        for ind, i in enumerate(start_ini_cond):
            ini_cond_sim = load_ini_conditions_true_option4(i)
            solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_pure_rk)
            t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
            solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=t_final_simulations, step_size=step_size_hybrid_rk, 
                                                pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
            t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
            t_true_ini, states_true = return_true_solution_option4(i, int(i+ratio_trapz_assimulo*t_final_simulations/step_size_hybrid_rk+1))
            t_true = t_true_ini-t_true_complete[i]
            errors_pure_array = errors_analysis(t_test_pure_rk, states_evo, t_true, states_true, ratio_trapz_assimulo, error_type='plot_dif')
            errors_pinn_array = errors_analysis(t_evo_pinn, states_evo_pinn, t_true, states_true, ratio_trapz_assimulo, error_type='plot_dif')
            zero_initial_errors = np.zeros((1, errors_pure_array.shape[1]))
            errors_simulator_p = np.vstack([zero_initial_errors, errors_pure_array])
            errors_simulator_h = np.vstack([zero_initial_errors, errors_pinn_array])
            ax0.plot(t_test_pure_rk, errors_simulator_p[:, 8], color = 'orange', label='Pure solver')
            ax0.plot(t_evo_pinn, errors_simulator_h[:, 8], color = 'blue', label='Hybrid solver')
            ax1.plot(t_test_pure_rk, errors_simulator_p[:, 10], color = 'orange', label='Pure solver')
            ax1.plot(t_evo_pinn, errors_simulator_h[:, 10], color = 'blue', label='Hybrid solver')
            print(ind+1, len(start_ini_cond))
        plt.tight_layout()
        plt.show()
    
    elif args.study_selection == 5:
        timesteps_to_study =  [0.006, 0.008, 0.01, 0.014, 0.02, 0.024, 0.034, 0.04]
        plotting_states_machine3 = [2, 3, 8, 9, 12, 13, 18, 19, 22, 23, 28, 29]
        np.random.seed(25)
        error_dist_per_timestep_pure = np.ones((len(timesteps_to_study), 2*len(plotting_states_machine3)))
        error_dist_per_timestep_hybrid = np.ones((len(timesteps_to_study), 2*len(plotting_states_machine3)))
        t_true_complete, states_true_complete = return_true_solution_option4()
        time_step_assimulo = compute_time_step_assimulo(t_true_complete)
        nums_random_true_sim = np.random.randint(0, len(t_true_complete)-200, size=100).tolist()
        for ind_timestep, current_timestep in enumerate(timesteps_to_study):
            print(current_timestep, time_step_assimulo)
            ratio_trapz_assimulo = compute_time_step_ratio(current_timestep, time_step_assimulo)
            errors_dist_trapz  = np.ones((len(nums_random_true_sim), len(plotting_states_machine3)))
            errors_dist_hybrid = np.ones((len(nums_random_true_sim), len(plotting_states_machine3)))
            for ind_array, num_array_multi in enumerate(nums_random_true_sim):
                ini_cond_sim = load_ini_conditions_true_option4(num_array_multi)
                solver_pure_rk_method = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=current_timestep, step_size=current_timestep)
                t_test_pure_rk, states_evo = solver_pure_rk_method.simulation_main_loop(integration_scheme=args.rk_scheme)
                solver_hybrid_pinn      = TDS_simulation(dampings, freq, H, Xd_p, Yadmittance, pg_pf, ini_cond_sim, t_final=current_timestep, step_size=current_timestep, 
                                                    pinn_boost=args.machine, pinn_weights=simulation_pinn, pinn_limits=pinn_ops_limits)
                t_evo_pinn, states_evo_pinn = solver_hybrid_pinn.simulation_main_loop(integration_scheme=args.rk_scheme)
                t_true_sim, states_true_sim = return_true_solution_option4(num_array_multi, int(num_array_multi+ratio_trapz_assimulo+1))
                t_true_sim += -t_true_complete[num_array_multi]
                errors_pure_array = errors_analysis(t_test_pure_rk, states_evo, t_true_sim, states_true_sim, ratio_trapz_assimulo)
                errors_pinn_array = errors_analysis(t_evo_pinn, states_evo_pinn, t_true_sim, states_true_sim, ratio_trapz_assimulo)
                errors_dist_trapz[ind_array, :]  = errors_pure_array
                errors_dist_hybrid[ind_array, :] = errors_pinn_array
            medians_dist_trapz = np.median(errors_dist_trapz, axis=0)
            q1_trapz = np.percentile(errors_dist_trapz, 25, axis=0)
            q3_trapz = np.percentile(errors_dist_trapz, 75, axis=0)
            iqr_trapz = q3_trapz - q1_trapz
            upper_whisker_trapz = q3_trapz + 1.5 * iqr_trapz
            boxplot_data_trapz = np.hstack([medians_dist_trapz, upper_whisker_trapz])
            medians_dist_hybrid = np.median(errors_dist_hybrid, axis=0)
            q1_hybrid = np.percentile(errors_dist_hybrid, 25, axis=0)
            q3_hybrid = np.percentile(errors_dist_hybrid, 75, axis=0)
            iqr_hybrid = q3_hybrid - q1_hybrid
            upper_whisker_hybrid = q3_hybrid + 1.5 * iqr_hybrid
            boxplot_data_hybrid = np.hstack([medians_dist_hybrid, upper_whisker_hybrid])
            error_dist_per_timestep_pure[ind_timestep, :] = boxplot_data_trapz
            error_dist_per_timestep_hybrid[ind_timestep, :] = boxplot_data_hybrid
            print(ind_timestep+1, len(timesteps_to_study))
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        plotting_states_final = [8,10,4, 6] # [8,9,10,7]
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 4)
        for i, state_plot in enumerate(plotting_states_final):
            ax = plt.subplot(gs[0, i])
            if i ==0:
                ax.set_title('Delta3-Omega3')
            else:
                ax.set_title('Voltage magnitude 3')
            ax.plot(timesteps_to_study, error_dist_per_timestep_pure[:,state_plot], color='r', linestyle = '-', label='trapz')
            ax.plot(timesteps_to_study, error_dist_per_timestep_hybrid[:,state_plot],  color='g', linestyle= '-', label='PINN-trapz')
            ax.plot(timesteps_to_study, error_dist_per_timestep_pure[:,state_plot+12], color='r', linestyle = '--')
            ax.plot(timesteps_to_study, error_dist_per_timestep_hybrid[:,state_plot+12],  color='g', linestyle= '--')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
        plt.tight_layout()
        plt.show()