import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class custom_overview1:
    def __init__(self, sim_time, t_test_solver1, errors_solver1, t_test_solver2, errors_solver2) -> None:
        assert sim_time > 0
        assert t_test_solver1[-1] == sim_time
        assert t_test_solver2[-1] == sim_time
        self.simulation_range = sim_time
        self.t_test_solver1 = t_test_solver1
        self.t_test_solver2 = t_test_solver2
        self.errors_solver1 = errors_solver1
        self.errors_solver2 = errors_solver2
        self.num_plots = errors_solver1.shape[1]
        if len(t_test_solver1) > errors_solver1.shape[0]:
            self.errors_solver1 = self.add_zeros_initial_value(errors_solver1)
        if len(t_test_solver2) > errors_solver2.shape[0]:
            self.errors_solver2 = self.add_zeros_initial_value(errors_solver2)
        assert len(t_test_solver1) == self.errors_solver1.shape[0]
        assert len(t_test_solver2) == self.errors_solver2.shape[0]

    def trajectory_and_errors_plot(self, state_1, state_2, time_array_true, states_array_true, states_array_pure, states_array_hybrid):
        assert self.errors_solver1.shape[1] == self.errors_solver2.shape[1]
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0, 0])
        ax0.set_title('Delta_Theta Gen. 3')
        ax0.grid()
        ax1 = plt.subplot(gs[0, 1])
        ax1.set_title('Voltage magnitude Gen. 3')
        ax1.grid()

        ax0.plot(time_array_true, states_array_true[:, 22]-states_array_true[:, 29], color='k', linestyle='--')
        ax0.plot(self.t_test_solver1, states_array_pure[:, 22]- states_array_pure[:, 29], color='orange', linestyle='-') # , marker='x', markersize=6
        ax0.plot(self.t_test_solver2, states_array_hybrid[:, 22]- states_array_hybrid[:, 29], color='blue', linestyle='-') # , marker='o', markersize=6
        ax0.set_ylabel('Delta-Theta [rad]')
        ax0.set_xlabel('Time [s]')

        ax1.plot(time_array_true, states_array_true[:, 28], color='k', linestyle='--')
        ax1.plot(self.t_test_solver1, states_array_pure[:, 28], color='orange', linestyle='-') # , marker='x', markersize=6
        ax1.plot(self.t_test_solver2, states_array_hybrid[:, 28], color='blue', linestyle='-') # , marker='o', markersize=6
        ax1.set_ylabel('Voltage magnitude [p.u.]')
        ax1.set_xlabel('Time [s]')

        ax0_twin = ax0.twinx()
        ax0_twin.set_ylabel('Delta-Theta l1 errors')
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylabel('V magn l1 errors')

        ax0_twin.plot(self.t_test_solver1, self.errors_solver1[:, state_1], color='orange', linestyle='--', alpha=0.6)
        ax0_twin.fill_between(self.t_test_solver1, self.errors_solver1[:, state_1], color='orange', alpha=0.3)
        ax0_twin.plot(self.t_test_solver2, self.errors_solver2[:, state_1], color='blue', linestyle='--', alpha=0.6)
        ax0_twin.fill_between(self.t_test_solver2, self.errors_solver2[:, state_1], color='blue', alpha=0.3)
        ax0_twin.set_ylim(0, max(self.errors_solver1[:, state_1])*5)
        ticks = ax0_twin.get_yticks()
        ax0_twin.set_yticks([tick for tick in ticks if tick <= max(self.errors_solver1[:, state_1])])

        ax1_twin.plot(self.t_test_solver1, self.errors_solver1[:, state_2], color='orange', linestyle='--', alpha=0.6)
        ax1_twin.fill_between(self.t_test_solver1, self.errors_solver1[:, state_2], color='orange', alpha=0.3)
        ax1_twin.plot(self.t_test_solver2, self.errors_solver2[:, state_2], color='blue', linestyle='--', alpha=0.6)
        ax1_twin.fill_between(self.t_test_solver2, self.errors_solver2[:, state_2], color='blue', alpha=0.3)
        ax1_twin.set_ylim(0, max(self.errors_solver1[:, state_2])*5)
        ticks = ax1_twin.get_yticks()
        ax1_twin.set_yticks([tick for tick in ticks if tick <= max(self.errors_solver1[:, state_2])])

    def add_zeros_initial_value(self, errors_simulator):
        zero_initial_errors = np.zeros((1, errors_simulator.shape[1]))
        errors_simulator = np.vstack([zero_initial_errors, errors_simulator])
        return errors_simulator
    
    def show_results(self, save_fig=False):
        plt.tight_layout()

        if save_fig:
            plt.savefig('overviewfinal')

        plt.show()

class custom_overview2:
    def __init__(self, timestep_list):
        self.timestep_list = timestep_list
    
    def show_results(self, maximums_pure, maximums_hybrid, save_fig=False):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(1, 2)
        ax0 = plt.subplot(gs[0, 0])
        ax0.set_title('Delta_Theta Gen. 3')
        ax0.grid()
        ax1 = plt.subplot(gs[0, 1])
        ax1.set_title('Voltage magnitude Gen. 3')
        ax1.grid()
        ax0.plot(self.timestep_list, maximums_pure[:, 0], label='Pure solver')
        ax0.plot(self.timestep_list, maximums_hybrid[:, 0], label='Hybrid solver')
        ax0.set_ylabel('l1 errors')
        ax0.legend()
        ax1.plot(self.timestep_list, maximums_pure[:, 1], label='Pure solver')
        ax1.plot(self.timestep_list, maximums_hybrid[:, 1], label='Hybrid solver')
        ax1.set_ylabel('l1 errors')
        ax1.legend()
        plt.tight_layout()

        if save_fig:
            plt.savefig('overviewfinal')

        plt.show()