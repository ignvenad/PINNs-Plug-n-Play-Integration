import sys
import torch
from torch.autograd.functional import jacobian

class TDS_simulation():
    def __init__(self, damping, freq, H_vec, Xdp_vec, Y_adm, pg_pf, ini_cond, t_final: float, step_size: float, pinn_boost=None, pinn_weights=None, pinn_limits=None)  -> None:
        assert t_final > 0; assert step_size > 0
        self.t_final = t_final
        self.damping_machines = torch.tensor(damping, dtype=torch.float64)
        self.step_size = torch.tensor(step_size, dtype=torch.float64)
        self.initial_state = ini_cond
        self.freq = torch.tensor(freq, dtype=torch.float64)
        self.H    = torch.tensor(H_vec, dtype=torch.float64)
        self.xd_prime = torch.tensor(Xdp_vec, dtype=torch.float64)
        self.Yadmittance = Y_adm
        self.pg_pf = torch.tensor(pg_pf, dtype=torch.float64)
        self.pinn_boost = pinn_boost
        if type(pinn_boost) == int:
            assert pinn_boost in [1,2,3]
            assert pinn_weights is not None and pinn_limits is not None
            self.upload_pinn_to_dae(pinn_weights, pinn_limits)
            
    def upload_pinn_to_dae(self, pinn_weights, pinn_limits):
        self.pinn_check = pinn_weights
        self.limits_v = pinn_limits[0]
        self.limits_theta = pinn_limits[1]
        self.limits_delta = pinn_limits[2]
        self.limits_omega = pinn_limits[3]
    
    def stator_equations_re(self, states_stator, no_machine) -> torch.Tensor:
        Eq_prime1, Vm1, delta1, Theta1 = states_stator
        id1 = 1/self.xd_prime[no_machine] *(Eq_prime1-Vm1*torch.cos(delta1-Theta1))
        return id1
    
    def stator_equations_im(self, states_stator, no_machine) -> torch.Tensor:
        Ed_prime1, Vm1, delta1, Theta1 = states_stator
        iq1 = 1/self.xd_prime[no_machine] *(Ed_prime1-Vm1*torch.sin(delta1-Theta1))
        return iq1
    
    def reference_update(self, states_stator) -> torch.Tensor:
        delta, Id, Iq = states_stator
        currents = (Id+1j*Iq)*torch.exp(1j*(delta-torch.pi/2))
        return currents
    
    def network_equation(self, states_net, no_machine) -> torch.Tensor:
        Vm1, Theta1, Vm2, Theta2, Vm3, Theta3 = states_net
        i_inj = self.Yadmittance[no_machine,0]*Vm1 * torch.exp(1j * Theta1) + self.Yadmittance[no_machine,1] * Vm2 * torch.exp(1j * Theta2) + self.Yadmittance[no_machine,2] * Vm3 * torch.exp(1j * Theta3)
        return i_inj
    
    def dif_equation_deq(self, diff_states, input_states, no_machine) -> torch.Tensor:
        Eq = diff_states
        d_Eq = torch.zeros_like(Eq)
        return d_Eq
    
    def dif_equation_ded(self, diff_states, input_states, no_machine) -> torch.Tensor:
        Ed = diff_states
        d_Ed = torch.zeros_like(Ed)
        return d_Ed
    
    def dif_equation_ddelta(self, diff_states, input_states, no_machine) -> torch.Tensor:
        omega = input_states[0]
        d_delta = omega*2*torch.pi*self.freq
        return d_delta
    
    def dif_equation_dw(self, diff_states, input_states, no_machine) -> torch.Tensor:
        omega = diff_states
        Eq_prime, Ed_prime, Id, Iq = input_states
        d_omega = (self.pg_pf[no_machine] - Ed_prime * Id - Eq_prime*Iq - self.damping_machines[no_machine] * omega)/(2*self.H[no_machine])
        return d_omega
    
    @torch.enable_grad()
    def trapezoidal_rule_func(self, diffstate, input_states, func, no_machine) -> torch.Tensor:
        state_0, state_1 = diffstate
        states_0_inputs = input_states[::2]
        states_1_inputs = input_states[1::2]
        d_state_0 = func(state_0, states_0_inputs, no_machine)
        d_state_1 = func(state_1, states_1_inputs, no_machine)
        computation_trapz = state_1 - state_0 - 0.5*self.step_size*(d_state_0+d_state_1)
        return computation_trapz
    
    @torch.enable_grad()
    def backward_euler_func(self, diffstate, input_states, func, no_machine) -> torch.Tensor:
        state_0, state_1 = diffstate
        states_1_inputs = input_states[1::2]
        d_state_1 = func(state_1, states_1_inputs, no_machine)
        computation_be = state_1 - state_0 - self.step_size*d_state_1
        return computation_be
    
    def check_pinn_limits(self, pinn_inputs) -> None:
        value_vm_check, value_theta_check, value_delta_check, value_omega_check = pinn_inputs
        if self.limits_v[0] > value_vm_check or self.limits_v[1] < value_vm_check:
            print('Voltage_Careful', self.limits_v, value_vm_check)
        if self.limits_theta[0] > value_theta_check or self.limits_theta[1] < value_theta_check:
            print('Theta_Careful', self.limits_theta, value_theta_check)
        if self.limits_delta[0] > value_delta_check or self.limits_delta[1] < value_delta_check:
            print('Delta_Careful', self.limits_delta, value_delta_check)
        if self.limits_omega[0] > value_omega_check or self.limits_omega[1] < value_omega_check:
            print('Omega_Careful', self.limits_omega, value_omega_check)
    
    def rk_integration_scheme(self, diffstate, input_states, func, no_machine) -> torch.Tensor:
        if self.integration_scheme == 'trapezoidal':
            output_computation = self.trapezoidal_rule_func(diffstate, input_states, func, no_machine)
        elif self.integration_scheme == 'backward_euler':
            output_computation = self.backward_euler_func(diffstate, input_states, func, no_machine)
        return output_computation
    
    def pinn_integration_scheme(self, pinn_input) -> tuple:
        preds_pinn = self.pinn_check(pinn_input)
        d_delta = preds_pinn[:, 0:1][0][0]
        d_omega = preds_pinn[:, 1:2][0][0]
        return d_delta, d_omega
    
    def calculate_new_reference(self, theta_0, theta_1, delta_0):
        theta_pend = (theta_1-theta_0)/self.step_size
        omicron_0 = delta_0 - theta_0
        return theta_pend, omicron_0
    
    def update_function(self, states_x0, states_x1) -> torch.Tensor:

        Eq_prime1_0, Eq_prime2_0, Eq_prime3_0 = states_x0[0], states_x0[10], states_x0[20]
        Ed_prime1_0, Ed_prime2_0, Ed_prime3_0 = states_x0[1], states_x0[11], states_x0[21]
        delta1_0, delta2_0, delta3_0          = states_x0[2], states_x0[12], states_x0[22]
        omega1_0, omega2_0, omega3_0          = states_x0[3], states_x0[13], states_x0[23]
        Id1_0, Id2_0, Id3_0                   = states_x0[4], states_x0[14], states_x0[24]
        Iq1_0, Iq2_0, Iq3_0                   = states_x0[5], states_x0[15], states_x0[25]
        Id_g_1_0, Id_g_2_0, Id_g_3_0          = states_x0[6], states_x0[16], states_x0[26]
        Iq_g_1_0, Iq_g_2_0, Iq_g_3_0          = states_x0[7], states_x0[17], states_x0[27]
        Vm1_0, Vm2_0, Vm3_0                   = states_x0[8], states_x0[18], states_x0[28]
        Theta1_0, Theta2_0, Theta3_0          = states_x0[9], states_x0[19], states_x0[29]     
        Eq_prime1, Eq_prime2, Eq_prime3 = states_x1[0], states_x1[10], states_x1[20]
        Ed_prime1, Ed_prime2, Ed_prime3 = states_x1[1], states_x1[11], states_x1[21]
        delta1, delta2, delta3          = states_x1[2], states_x1[12], states_x1[22]
        omega1, omega2, omega3          = states_x1[3], states_x1[13], states_x1[23]
        Id1, Id2, Id3                   = states_x1[4], states_x1[14], states_x1[24]
        Iq1, Iq2, Iq3                   = states_x1[5], states_x1[15], states_x1[25]
        Id_g_1, Id_g_2, Id_g_3          = states_x1[6], states_x1[16], states_x1[26]
        Iq_g_1, Iq_g_2, Iq_g_3          = states_x1[7], states_x1[17], states_x1[27]
        Vm1, Vm2, Vm3                   = states_x1[8], states_x1[18], states_x1[28]
        Theta1, Theta2, Theta3          = states_x1[9], states_x1[19], states_x1[29]       
        res_0 =  self.rk_integration_scheme((Eq_prime1_0, Eq_prime1), (), self.dif_equation_deq, 0)
        res_1 =  self.rk_integration_scheme((Ed_prime1_0, Ed_prime1), (), self.dif_equation_ded, 0)
        if self.pinn_boost == 1:
            theta_pend, omicron_0 = self.calculate_new_reference(Theta1_0, Theta1, delta1_0)
            inputs_pinn = [Vm1_0.item(), theta_pend.item(), omicron_0.item(), omega1_0.item()]
            self.check_pinn_limits(inputs_pinn)
            pinn_input_data = torch.cat([Vm1_0.view(-1,1), Vm1.view(-1,1), theta_pend.view(-1,1), omicron_0.view(-1,1), omega1_0.view(-1,1), self.step_size.view(-1,1)], dim=1)
            d_values_pinn = self.pinn_integration_scheme(pinn_input_data)
            res_2 = delta1 - self.step_size*d_values_pinn[0] - omicron_0 - theta_pend*self.step_size - Theta1_0
            res_3 = omega1 - omega3_0 -self.step_size*d_values_pinn[1]
        else:
            res_2 =  self.rk_integration_scheme((delta1_0, delta1), (omega1_0, omega1), self.dif_equation_ddelta, 0)
            res_3 =  self.rk_integration_scheme((omega1_0, omega1), (Eq_prime1_0, Eq_prime1, Ed_prime1_0, Ed_prime1, Id1_0, Id1, Iq1_0, Iq1), self.dif_equation_dw, 0)
        res_4 =  Id1 - self.stator_equations_re([Eq_prime1, Vm1, delta1, Theta1], 0)
        res_5 =  Iq1 + self.stator_equations_im([Ed_prime1, Vm1, delta1, Theta1], 0)
        current_updates_m1 = self.reference_update([delta1, Id1, Iq1])
        res_6 =  Id_g_1 - torch.real(current_updates_m1)
        res_7 =  Iq_g_1 - torch.imag(current_updates_m1)
        current_inj_m1 = self.network_equation([Vm1, Theta1, Vm2, Theta2, Vm3, Theta3], 0)
        res_8 =  Id_g_1 - torch.real(current_inj_m1)
        res_9 =  Iq_g_1 - torch.imag(current_inj_m1)
        res_10 = self.rk_integration_scheme((Eq_prime2_0, Eq_prime2), (), self.dif_equation_deq, 1)
        res_11 = self.rk_integration_scheme((Ed_prime2_0, Ed_prime2), (), self.dif_equation_ded, 1)
        if self.pinn_boost == 2:
            theta_pend, omicron_0 = self.calculate_new_reference(Theta2_0, Theta2, delta2_0)
            inputs_pinn = [Vm2_0.item(), theta_pend.item(), omicron_0.item(), omega2_0.item()]
            self.check_pinn_limits(inputs_pinn)
            pinn_input_data = torch.cat([Vm2_0.view(-1,1), Vm2.view(-1,1), theta_pend.view(-1,1), omicron_0.view(-1,1), omega2_0.view(-1,1), self.step_size.view(-1,1)], dim=1)
            d_values_pinn = self.pinn_integration_scheme(pinn_input_data)
            res_12 = delta2 - self.step_size*d_values_pinn[0] - omicron_0 - theta_pend*self.step_size - Theta2_0
            res_13 = omega2 - omega2_0 -self.step_size*d_values_pinn[1]
        else:
            res_12 = self.rk_integration_scheme((delta2_0, delta2), (omega2_0, omega2), self.dif_equation_ddelta, 1)
            res_13 = self.rk_integration_scheme((omega2_0, omega2), (Eq_prime2_0, Eq_prime2, Ed_prime2_0, Ed_prime2, Id2_0, Id2, Iq2_0, Iq2), self.dif_equation_dw, 1)
        res_14 = Id2 - self.stator_equations_re([Eq_prime2, Vm2, delta2, Theta2], 1)
        res_15 = Iq2 + self.stator_equations_im([Ed_prime2, Vm2, delta2, Theta2], 1)
        current_updates_m2 = self.reference_update([delta2, Id2, Iq2])
        res_16 =  Id_g_2 - torch.real(current_updates_m2)
        res_17 =  Iq_g_2 - torch.imag(current_updates_m2)
        current_inj_m2 = self.network_equation([Vm1, Theta1, Vm2, Theta2, Vm3, Theta3], 1)
        res_18 =  Id_g_2 - torch.real(current_inj_m2)
        res_19 =  Iq_g_2 - torch.imag(current_inj_m2)
        res_20 = self.rk_integration_scheme((Eq_prime3_0, Eq_prime3), (), self.dif_equation_deq, 2)
        res_21 = self.rk_integration_scheme((Ed_prime3_0, Ed_prime3), (), self.dif_equation_ded, 2)
        if self.pinn_boost == 3:
            theta_pend, omicron_0 = self.calculate_new_reference(Theta3_0, Theta3, delta3_0)
            inputs_pinn = [Vm3_0.item(), theta_pend.item(), omicron_0.item(), omega3_0.item()]
            self.check_pinn_limits(inputs_pinn)
            pinn_input_data = torch.cat([Vm3_0.view(-1,1), Vm3.view(-1,1), theta_pend.view(-1,1), omicron_0.view(-1,1), omega3_0.view(-1,1), self.step_size.view(-1,1)], dim=1)
            d_values_pinn = self.pinn_integration_scheme(pinn_input_data)
            res_22 = delta3 - self.step_size*d_values_pinn[0] - omicron_0 - theta_pend*self.step_size - Theta3_0
            res_23 = omega3 - omega3_0 -self.step_size*d_values_pinn[1]
        else:
            res_22 = self.rk_integration_scheme((delta3_0, delta3), (omega3_0, omega3), self.dif_equation_ddelta, 2)
            res_23 = self.rk_integration_scheme((omega3_0, omega3), (Eq_prime3_0, Eq_prime3, Ed_prime3_0, Ed_prime3, Id3_0, Id3, Iq3_0, Iq3), self.dif_equation_dw, 2)
        res_24 = Id3 - self.stator_equations_re([Eq_prime3, Vm3, delta3, Theta3], 2)
        res_25 = Iq3 + self.stator_equations_im([Ed_prime3, Vm3, delta3, Theta3], 2)
        current_updates_m3 = self.reference_update([delta3, Id3, Iq3])
        res_26 =  Id_g_3 - torch.real(current_updates_m3)
        res_27 =  Iq_g_3 - torch.imag(current_updates_m3)
        current_inj_m3 = self.network_equation([Vm1, Theta1, Vm2, Theta2, Vm3, Theta3], 2)
        res_28 =  Id_g_3 - torch.real(current_inj_m3)
        res_29 =  Iq_g_3 - torch.imag(current_inj_m3)
        return torch.stack([res_0,res_1,res_2,res_3, res_4, res_5, res_6, res_7, res_8, res_9,
                        res_10,res_11,res_12,res_13, res_14, res_15, res_16, res_17, res_18, res_19,
                        res_20,res_21,res_22,res_23, res_24, res_25, res_26, res_27, res_28, res_29])

    def newton_method(self, states_x0) -> torch.Tensor:
        states_x1 = states_x0.detach().clone()
        it_count = 0
        residual_iteration = torch.ones(30, dtype=torch.float64)
        while torch.max(torch.abs(residual_iteration)) > torch.tensor(1e-6, dtype=torch.float64) and it_count < 10:
            states_x1.requires_grad_(True)
            jacobian_matrix = jacobian(lambda x1: self.update_function(states_x0, x1), states_x1)
            inverse_matrix = torch.linalg.inv(jacobian_matrix)
            states_x1.requires_grad_(False)
            residual_iteration = self.update_function(states_x0, states_x1)
            x_increment = - torch.matmul(inverse_matrix, residual_iteration)
            new_states_x1 = states_x1 + x_increment
            states_x1 = new_states_x1.detach().clone()
            it_count +=1 
        assert it_count < 10
        return states_x1
    
    def simulation_main_loop(self, integration_scheme='trapezoidal') -> tuple:
        no_steps_float = self.t_final/self.step_size
        assert no_steps_float % 1 == 0
        no_steps = int(no_steps_float) + 1
        time_array_sim = torch.arange(no_steps)*self.step_size
        states_array_sim = torch.zeros((no_steps, 30), dtype=torch.float64)
        states_array_sim[0, :] = self.initial_state
        states_iteration = self.initial_state
        self.integration_scheme = integration_scheme
        supported_schemes =['trapezoidal', 'backward_euler']
        try:
            check_scheme(self.integration_scheme, supported_schemes)
        except Scheme_Not_Supported as e:
            print(e)
            sys.exit(1)
        for i in range(1, no_steps):
            states_next_time_step = self.newton_method(states_iteration)
            states_array_sim[i, :] = states_next_time_step
            states_iteration = states_next_time_step
        return time_array_sim.numpy(), states_array_sim.numpy()
    
class Scheme_Not_Supported(Exception):

    def __init__(self, supported_schemes):
        self.supported_schemes = supported_schemes
        super().__init__(f"Invalid scheme. Must be one of: {', '.join(supported_schemes)}")

def check_scheme(scheme, supported_schemes):

    if scheme not in supported_schemes:
        raise Scheme_Not_Supported(supported_schemes)