import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class trajectories_overview:
    def __init__(self, sim_time, t_test_rk_pinn_boost, states_rk_pinn_boost, t_test_rk, states_pure_rk, t_test_assimulo, states_assimulo) -> None:
        assert sim_time > 0
        assert t_test_rk_pinn_boost[-1] == sim_time
        self.simulation_range = sim_time
        self.t_test_rk_pinn_boost = t_test_rk_pinn_boost
        self.states_rk_pinn_boost = states_rk_pinn_boost
        self.t_test_rk = t_test_rk
        self.states_pure_rk = states_pure_rk
        self.t_test_assimulo = t_test_assimulo
        self.states_assimulo = states_assimulo

    def currents_calculation(self, states_simulation):
        complex_currents_1 = (states_simulation[:,  6] +(1j)* states_simulation[:,  7])*(1*np.exp((-1j)*states_simulation[:,9]))            
        complex_currents_2 = (states_simulation[:, 16] +(1j)* states_simulation[:, 17])*(1*np.exp((-1j)*states_simulation[:,9]))
        complex_currents_3 = (states_simulation[:, 26] +(1j)* states_simulation[:, 27])*(1*np.exp((-1j)*states_simulation[:,9]))
        return complex_currents_1, complex_currents_2, complex_currents_3
    
    def compute_gradient_delta_theta(self, states_sim, no_machine):
        ind_delta = 2+10*(no_machine-1)
        ind_theta = 9+10*(no_machine-1)
        res_trajectory = states_sim[:, ind_delta] - states_sim[:, ind_theta]
        return res_trajectory
    
    def compute_omegas(self, states_sim, no_machine):
        ind_omega = 3+10*(no_machine-1)
        res_trajectory = states_sim[:, ind_omega]
        return res_trajectory
    
    def compute_theta_reference(self, states_sim, no_machine1, no_machine2):
        ind_theta1 = 9+10*(no_machine1-1)
        ind_theta2 = 9+10*(no_machine2-1)
        res_trajectory = states_sim[:, ind_theta1]-states_sim[:, ind_theta2]

        return res_trajectory

    def compute_results(self, pure_rk_scheme=False, assimulo_states=False):
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(3, 2)

        ax0 = plt.subplot(gs[0, 0])
        ax0_m1_pinn, = ax0.plot(self.t_test_rk_pinn_boost, self.compute_gradient_delta_theta(self.states_rk_pinn_boost, 1), 'r-') 
        ax0_m2_pinn, = ax0.plot(self.t_test_rk_pinn_boost, self.compute_gradient_delta_theta(self.states_rk_pinn_boost, 2), 'g-') 
        ax0_m3_pinn, = ax0.plot(self.t_test_rk_pinn_boost, self.compute_gradient_delta_theta(self.states_rk_pinn_boost, 3), 'b-') 
        ax0.set_title('Gradient delta-theta evolution')

        ax1 = plt.subplot(gs[0, 1])
        ax1_m1_pinn, = ax1.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 1), 'r-')
        ax1_m2_pinn, = ax1.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 2), 'g-')
        ax1_m3_pinn, = ax1.plot(self.t_test_rk_pinn_boost, self.compute_omegas(self.states_rk_pinn_boost, 3), 'b-')
        ax1.set_title('Omegas evolution')

        complex_currents_pinn1, complex_currents_pinn2, complex_currents_pinn3 = self.currents_calculation(self.states_rk_pinn_boost)
        
        ax2 = plt.subplot(gs[1, 0])
        ax2.plot(self.t_test_rk_pinn_boost, np.real(complex_currents_pinn1), 'r-', label='Machine 1')
        ax2.plot(self.t_test_rk_pinn_boost, np.real(complex_currents_pinn2), 'g-', label='Machine 2')
        ax2.plot(self.t_test_rk_pinn_boost, np.real(complex_currents_pinn3), 'b-', label='Machine 3')
        ax2.set_title('ID evolution')
        ax2.legend()

        ax3 = plt.subplot(gs[1, 1])
        ax3.plot(self.t_test_rk_pinn_boost, np.imag(complex_currents_pinn1), 'r-', label='Machine 1')
        ax3.plot(self.t_test_rk_pinn_boost, np.imag(complex_currents_pinn2), 'g-', label='Machine 2')
        ax3.plot(self.t_test_rk_pinn_boost, np.imag(complex_currents_pinn3), 'b-', label='Machine 3')
        ax3.set_title('IQ evolution')
        ax3.legend()

        ax4 = plt.subplot(gs[2, 0])
        ax4_m1_pinn, = ax4.plot(self.t_test_rk_pinn_boost, self.states_rk_pinn_boost[:,  8], 'r-')
        ax4_m2_pinn, = ax4.plot(self.t_test_rk_pinn_boost, self.states_rk_pinn_boost[:, 18], 'g-')
        ax4_m3_pinn, = ax4.plot(self.t_test_rk_pinn_boost, self.states_rk_pinn_boost[:, 28], 'b-')
        ax4.set_title('Vm evolution')

        ax5 = plt.subplot(gs[2, 1])
        ax5_m1_pinn, = ax5.plot(self.t_test_rk_pinn_boost, self.compute_theta_reference(self.states_rk_pinn_boost, 1, 2), 'r-')                   
        ax5_m2_pinn, = ax5.plot(self.t_test_rk_pinn_boost, self.compute_theta_reference(self.states_rk_pinn_boost, 1, 3), 'g-')         
        ax5_m3_pinn, = ax5.plot(self.t_test_rk_pinn_boost, self.compute_theta_reference(self.states_rk_pinn_boost, 2, 3), 'b-')         
        ax5.set_title('Thetas evolution')

        if pure_rk_scheme:
            assert self.t_test_rk[-1] == self.simulation_range
            ax0.plot(self.t_test_rk, self.compute_gradient_delta_theta(self.states_pure_rk, 1), 'r--')  
            ax0.plot(self.t_test_rk, self.compute_gradient_delta_theta(self.states_pure_rk, 2), 'g--')
            ax0.plot(self.t_test_rk, self.compute_gradient_delta_theta(self.states_pure_rk, 3), 'b--')
            
            ax1.plot(self.t_test_rk, self.compute_omegas(self.states_pure_rk, 1), 'r--')
            ax1.plot(self.t_test_rk, self.compute_omegas(self.states_pure_rk, 2), 'g--')
            ax1.plot(self.t_test_rk, self.compute_omegas(self.states_pure_rk, 3), 'b--')
            
            complex_currents_rk_1, complex_currents_rk_2, complex_currents_rk_3 = self.currents_calculation(self.states_pure_rk)

            ax2.plot(self.t_test_rk, np.real(complex_currents_rk_1), 'r--')
            ax2.plot(self.t_test_rk, np.real(complex_currents_rk_2), 'g--')
            ax2.plot(self.t_test_rk, np.real(complex_currents_rk_3), 'b--')

            ax3.plot(self.t_test_rk, np.imag(complex_currents_rk_1), 'r--')
            ax3.plot(self.t_test_rk, np.imag(complex_currents_rk_2), 'g--')
            ax3.plot(self.t_test_rk, np.imag(complex_currents_rk_3), 'b--')
            
            ax4.plot(self.t_test_rk, self.states_pure_rk[:,  8], 'r--')
            ax4.plot(self.t_test_rk, self.states_pure_rk[:, 18], 'g--')
            ax4.plot(self.t_test_rk, self.states_pure_rk[:, 28], 'b--')
        
            ax5.plot(self.t_test_rk, self.compute_theta_reference(self.states_pure_rk, 1, 2), 'r--')                                 
            ax5.plot(self.t_test_rk, self.compute_theta_reference(self.states_pure_rk, 1, 3), 'g--')           
            ax5.plot(self.t_test_rk, self.compute_theta_reference(self.states_pure_rk, 2, 3), 'b--') 

        if assimulo_states:
            assert self.t_test_assimulo[-1] == self.simulation_range
            ax0.plot(self.t_test_assimulo, self.compute_gradient_delta_theta(self.states_assimulo, 1), 'k-.')
            ax0.plot(self.t_test_assimulo, self.compute_gradient_delta_theta(self.states_assimulo, 2), 'k-.')
            ax0.plot(self.t_test_assimulo, self.compute_gradient_delta_theta(self.states_assimulo, 3), 'k-.')

            ax1.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 1),   'k-.')
            ax1.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 2), 'k-.')
            ax1.plot(self.t_test_assimulo, self.compute_omegas(self.states_assimulo, 3), 'k-.')

            complex_currents_assimulo1, complex_currents_assimulo2, complex_currents_assimulo3 = self.currents_calculation(self.states_assimulo)
            
            ax2.plot(self.t_test_assimulo, np.real(complex_currents_assimulo1), 'k-.')
            ax2.plot(self.t_test_assimulo, np.real(complex_currents_assimulo2), 'k-.')
            ax2.plot(self.t_test_assimulo, np.real(complex_currents_assimulo3), 'k-.')
            
            ax3.plot(self.t_test_assimulo, np.imag(complex_currents_assimulo1), 'k-.')
            ax3.plot(self.t_test_assimulo, np.imag(complex_currents_assimulo2), 'k-.')
            ax3.plot(self.t_test_assimulo, np.imag(complex_currents_assimulo3), 'k-.')

            ax4.plot(self.t_test_assimulo, self.states_assimulo[:,  8], 'k-.')
            ax4.plot(self.t_test_assimulo, self.states_assimulo[:, 18], 'k-.')
            ax4.plot(self.t_test_assimulo, self.states_assimulo[:, 28], 'k-.')

            ax5.plot(self.t_test_assimulo, self.compute_theta_reference(self.states_assimulo, 1, 2), 'k-.')
            ax5.plot(self.t_test_assimulo, self.compute_theta_reference(self.states_assimulo, 1, 3), 'k-.')
            ax5.plot(self.t_test_assimulo, self.compute_theta_reference(self.states_assimulo, 2, 3), 'k-.')

            ax0_m1_pinn.set_label('Machine 1')
            ax0_m2_pinn.set_label('Machine 2')
            ax0_m3_pinn.set_label('Machine 3')
            ax0.legend()

            ax1_m1_pinn.set_label('Machine 1')
            ax1_m2_pinn.set_label('Machine 2')
            ax1_m3_pinn.set_label('Machine 3')
            ax1.legend()

            ax4_m1_pinn.set_label('Machine 1')    
            ax4_m2_pinn.set_label('Machine 2')    
            ax4_m3_pinn.set_label('Machine 3')            
            ax4.legend()

            ax5_m1_pinn.set_label('Machines 1-2')
            ax5_m2_pinn.set_label('Machines 1-3')
            ax5_m3_pinn.set_label('Machines 2-3')
            ax5.legend()

    def show_results(self, save_fig=False):
        plt.tight_layout()

        if save_fig:
            plt.savefig('overviewfinal')

        plt.show()