# Enhanced Tokamak Simulation Code with Vector xi and integration with old GS solver
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
os.makedirs('gs_plots', exist_ok=True)
# --- Import Old Grad-Shafranov Solver ---
import grad_shafranov_solver_old as gs_module # Using the old solver
gs_solver_instance = gs_module.grad_shafranov_solver() # Instantiate GS solver

# --- Constants ---
MU0 = 4 * np.pi * 1e-7

# --- Tokamak Parameters and Circuit Configuration ---
num_pf_coils = 4
num_oh_coils = 1
idx_plasma = num_pf_coils + num_oh_coils
num_circuits = num_pf_coils + num_oh_coils + 1
NUM_XI_PARAMS = 6
IDX_R0, IDX_A, IDX_KAPPA, IDX_DELTA, IDX_LI, IDX_BETAP = 0, 1, 2, 3, 4, 5

# --- GS Solver Cache and Control ---
global_gs_config = {
    "last_call_time": -1.0,
    "data": None, # Cached GS results [R0, a, kappa, delta, li, beta_p]
    "call_interval": 0.002, # Default, will be made adjustable
    "min_Ip_for_gs": 1.0e4,
    "instance": gs_solver_instance
}

R_coil_values = np.array([0.01]*num_pf_coils + [0.005]*num_oh_coils)
R_plasma_nominal = 1e-5
R_matrix_base = np.diag(np.concatenate((R_coil_values, [R_plasma_nominal])))

N_matrix = np.zeros((num_circuits, num_pf_coils + num_oh_coils))
for i in range(num_pf_coils + num_oh_coils):
    N_matrix[i, i] = 1.0

_L_COILS_BASE = np.random.rand(num_pf_coils + num_oh_coils, num_pf_coils + num_oh_coils) * 2e-3
_L_COILS_BASE = (_L_COILS_BASE + _L_COILS_BASE.T) / 2
diag_vals = np.array([10e-3]*num_pf_coils + [100e-3]*num_oh_coils)
np.fill_diagonal(_L_COILS_BASE, np.diag(_L_COILS_BASE) + diag_vals)

# --- Wrapper para o solver antigo ---
def run_gs_solver(gs_instance, input_params=None):
    """
    Wrapper para o solver antigo que não possui o método run_solver.
    Implementa a mesma interface que o método run_solver do solver novo.
    
    Parâmetros:
    -----------
    gs_instance : grad_shafranov_solver
        Instância do solucionador de Grad-Shafranov
    input_params : dict, opcional
        Dicionário com parâmetros de entrada para o solucionador.
        Pode conter 'Ip_target' para escalar os perfis de pressão e corrente.
        
    Retorna:
    --------
    list
        Lista com os parâmetros calculados [R0, a, kappa, delta, li, beta_p]
    """
    # Configurar a grade se ainda não estiver configurada
    if gs_instance.R_grid is None or gs_instance.Z_grid is None:
        gs_instance._setup_grid()
        
    # Ajustar parâmetros com base em input_params
    if input_params is not None and 'Ip_target' in input_params:
        Ip_target = input_params['Ip_target']
        # Escalar os perfis de pressão e corrente com base na corrente alvo
        scale_factor = abs(Ip_target) / 1e6  # Normalizar para MA
        gs_instance.P0_TANH = 1.0e3 * (1.0 + scale_factor)
        gs_instance.FF0_PRIME_POLY_GS = 0.01 * (1.0 + 0.5 * scale_factor)
        print(f"Ajustando perfis para Ip_target={Ip_target:.2e} A, P0={gs_instance.P0_TANH:.2e}, FF'={gs_instance.FF0_PRIME_POLY_GS:.2e}")
    
    # Inicializar psi se necessário
    if gs_instance.psi is None:
        gs_instance.psi = gs_instance._initialize_psi()
        gs_instance.psi = gs_instance._apply_boundary_conditions(gs_instance.psi)
    
    # Resolver a equação de Grad-Shafranov iterativamente
    gs_instance.iteration_count = 0
    error_rms = float('inf')
    
    while gs_instance.iteration_count < gs_instance.MAX_ITERATIONS_GS and error_rms > gs_instance.CONVERGENCE_CRITERION_GS:
        gs_instance.iteration_count += 1
        
        # Armazenar iterações anteriores para atualização adaptativa de omega
        if gs_instance.iteration_count > gs_instance.min_iter_for_omega_update:
            gs_instance._psi_prev_prev_for_rho = gs_instance._psi_prev_for_rho
            gs_instance._psi_prev_for_rho = np.copy(gs_instance.psi)
        
        # Resolver uma iteração
        psi_new, error_rms = gs_instance._solve_iteration(gs_instance.psi)
        gs_instance.psi = psi_new
        
        # Atualizar omega adaptativamente
        if gs_instance.iteration_count > gs_instance.min_iter_for_omega_update + 1:
            gs_instance._update_omega_adaptive(gs_instance.psi, gs_instance._psi_prev_for_rho, gs_instance._psi_prev_prev_for_rho)
        
        # Aplicar condições de contorno
        gs_instance.psi = gs_instance._apply_boundary_conditions(gs_instance.psi)
        
        # Verificar convergência
        if error_rms <= gs_instance.CONVERGENCE_CRITERION_GS:
            print(f"GS convergiu após {gs_instance.iteration_count} iterações com erro RMS = {error_rms:.6e}")
            break
            
    if gs_instance.iteration_count >= gs_instance.MAX_ITERATIONS_GS:
        print(f"GS não convergiu após {gs_instance.MAX_ITERATIONS_GS} iterações. Erro RMS final = {error_rms:.6e}")
    
    # Extrair geometria e parâmetros
    gs_instance._extract_geometry_and_parameters()
    
    # Retornar os parâmetros calculados como uma lista
    return [gs_instance.R0_calc, gs_instance.a_calc, gs_instance.kappa_calc, 
            gs_instance.delta_calc, gs_instance.li_calc, gs_instance.beta_p_calc]

# --- Model Functions ---
def u_input_scenario(t):
    u_applied = np.zeros(num_pf_coils + num_oh_coils)
    idx_oh_channel = num_pf_coils
    if t < 0.1:
        u_applied[idx_oh_channel] = -50.0
    elif t < 1.0:
        u_applied[idx_oh_channel] = 100.0
    else:
        u_applied[idx_oh_channel] = 5.0
    if t > 0.5:
        u_applied[0] = 10 * np.sin(1 * np.pi * (t - 0.5))
        u_applied[1] = -8 * np.cos(1 * np.pi * (t - 0.5))
        if num_pf_coils > 2:
             u_applied[2] = 5 * np.sin(0.5 * np.pi * (t-0.5))
    return u_applied

def call_gs_solver_periodically(t, current_I_plasma, gs_config, force_run=False):
    ip_target_for_gs = abs(current_I_plasma)
    should_call_gs = force_run or \
                     ((t - gs_config["last_call_time"] >= gs_config["call_interval"]) and \
                      ip_target_for_gs > gs_config["min_Ip_for_gs"])

    if should_call_gs:
        print(f"Calling full GS Solver at t={t:.3f}s for Ip={ip_target_for_gs/1e6:.3f} MA (Interval: {gs_config['call_interval']}s)")
        try:
            # Pass the current Ip to GS solver for p_prime and FF_prime scaling
            # Usando o wrapper para o solver antigo
            gs_output = run_gs_solver(gs_config["instance"], input_params={"Ip_target": ip_target_for_gs})
            gs_config["data"] = np.array(gs_output[:NUM_XI_PARAMS]) # R0, a, kappa, delta, li, beta_p
            gs_config["last_call_time"] = t
            save_gs_plot(gs_config["instance"], t, gs_config["instance"].iteration_count)
        except Exception as e:
            print(f"ERROR during GS solver call at t={t:.3f}s: {e}. Using previous/default xi.")
            if gs_config["data"] is None:
                 # Fallback to some reasonable initial values if first call fails
                 gs_config["data"] = np.array([1.7, 0.5, 1.7, 0.3, 0.9, 0.2]) 
    
    if gs_config["data"] is None: # If still None (e.g. first step, low Ip, or call failed before)
        if (t < gs_config["call_interval"] / 2.0 and ip_target_for_gs > gs_config["min_Ip_for_gs"]) or force_run:
             print(f"Initial/Forced GS call at t={t:.3f}s as cache is None.")
             # Recursive call, ensure it's handled carefully to avoid infinite loops if GS always fails
             call_gs_solver_periodically(t, current_I_plasma, gs_config, force_run=True)
        
        if gs_config["data"] is None: # If still None after trying a forced call
            print(f"CRITICAL WARNING: GS data is None at t={t:.3f}s. Using default initial_xi_values.")
            gs_config["data"] = np.array([1.7, 0.5, 1.7, 0.3, 0.9, 0.2]) # Default initial values
            if ip_target_for_gs <= gs_config["min_Ip_for_gs"] and not force_run:
                gs_config["last_call_time"] = t # Update time to prevent immediate re-call if Ip is low

    return gs_config["data"]

def get_plasma_state_vector_and_Rp(t, states, gs_config_dict):
    I_plasma = states[idx_plasma]
    gs_derived_xi = call_gs_solver_periodically(t, I_plasma, gs_config_dict)
    xi_vector = np.array(gs_derived_xi) # R0, a, kappa, delta, li, beta_p from GS

    if abs(I_plasma) < 1e3:
        R_p = 2e-3
    else:
        # More advanced Rp model could use T_e from GS if available
        R_p = 1e-7 + 5e-5 / (1 + (abs(I_plasma) / 5e5)**2)
        R_p = max(R_p, 1e-8)
    return xi_vector, R_p

def get_dxi_vector_dt(t, states, xi_vector_current, gs_config_dict):
    """
    Calculates d(xi_vector)/dt for each component.
    R0, a, kappa, delta are now primarily determined by the GS solver's equilibrium.
    Their d/dt terms here should be zero, as their evolution happens discretely via GS calls.
    li and beta_p are also provided by the GS solver.
    If we assume li and beta_p from GS are the 'true' values at that time step,
    their d/dt can also be considered zero on the faster timescale of the ODE solver,
    as their evolution is captured by the periodic GS updates.
    """
    dxi_dt = np.zeros(NUM_XI_PARAMS)
    # dxi_dt[IDX_R0] = 0.0 # Determined by GS
    # dxi_dt[IDX_A]  = 0.0 # Determined by GS
    # dxi_dt[IDX_KAPPA] = 0.0 # Determined by GS
    # dxi_dt[IDX_DELTA] = 0.0 # Determined by GS
    # dxi_dt[IDX_LI] = 0.0 # Determined by GS
    # dxi_dt[IDX_BETAP] = 0.0 # Determined by GS
    return dxi_dt

def L_matrix_model(xi_vector, base_L_coils):
    L = np.zeros((num_circuits, num_circuits))
    L[0:idx_plasma, 0:idx_plasma] = base_L_coils
    idx_oh_in_coils = num_pf_coils
    R0, a, kappa, delta, li, beta_p = xi_vector

    M_oh_p = (4e-3 * (R0/1.7) - 1e-3 * (a/0.5))
    M_oh_p = max(1e-4, M_oh_p)
    L[idx_oh_in_coils, idx_plasma] = M_oh_p
    L[idx_plasma, idx_oh_in_coils] = M_oh_p

    for i in range(num_pf_coils):
        m_pf_p_i = (0.3e-3 * (R0/1.7) - 0.1e-3 * (a/0.5) + np.random.uniform(-0.05e-3,0.05e-3)) * ((-1)**i)
        L[i, idx_plasma] = m_pf_p_i
        L[idx_plasma, i] = m_pf_p_i

    if R0 > 0 and a > 0 and R0/a > 1:
        term_in_log = 8 * R0 / a
        if term_in_log <= 1.0: term_in_log = 1.01
        L_p = MU0 * R0 * (np.log(term_in_log) + li/2.0 - 2.0 + 0.1 * (kappa-1) + 0.2 * delta)
    else:
        L_p = MU0 * 1.7 * (np.log(8*1.7/0.5) + 0.9/2.0 - 2.0)
    L_p = max(L_p, 1e-7)
    L[idx_plasma, idx_plasma] = L_p
    np.fill_diagonal(L, np.diag(L) + 1e-9)
    return L

def dL_dxi_k_model_list(xi_vector, base_L_coils):
    # Since dxi_dt for geometric params, li, beta_p is now zero (handled by GS updates),
    # the dL/dt term from these components will be zero. 
    # However, the L_matrix_model still depends on xi, so dL/dxi_k is still needed if dxi_dt is non-zero for any reason.
    # For now, we keep the function, but its product with dxi_dt will be zero for most components.
    dL_dxi_list = [np.zeros((num_circuits, num_circuits)) for _ in range(NUM_XI_PARAMS)]
    idx_oh_in_coils = num_pf_coils
    R0, a, kappa, delta, li, beta_p = xi_vector

    dM_oh_p_dR0 = 4e-3 / 1.7
    dL_dxi_list[IDX_R0][idx_oh_in_coils, idx_plasma] = dM_oh_p_dR0
    dL_dxi_list[IDX_R0][idx_plasma, idx_oh_in_coils] = dM_oh_p_dR0
    for i in range(num_pf_coils):
        dm_pf_p_i_dR0 = (0.3e-3 / 1.7) * ((-1)**i)
        dL_dxi_list[IDX_R0][i, idx_plasma] = dm_pf_p_i_dR0
        dL_dxi_list[IDX_R0][idx_plasma, i] = dm_pf_p_i_dR0
    if R0 > 0 and a > 0 and R0/a > 1:
        term_in_log = 8 * R0 / a; term_in_log = max(term_in_log, 1.01)
        dL_p_dR0 = MU0 * (np.log(term_in_log) + li/2.0 - 2.0 + 0.1*(kappa-1) + 0.2*delta + 1.0)
    else: dL_p_dR0 = 0
    dL_dxi_list[IDX_R0][idx_plasma, idx_plasma] = dL_p_dR0

    dM_oh_p_da = -1e-3 / 0.5
    dL_dxi_list[IDX_A][idx_oh_in_coils, idx_plasma] = dM_oh_p_da
    dL_dxi_list[IDX_A][idx_plasma, idx_oh_in_coils] = dM_oh_p_da
    for i in range(num_pf_coils):
        dm_pf_p_i_da = (-0.1e-3 / 0.5) * ((-1)**i)
        dL_dxi_list[IDX_A][i, idx_plasma] = dm_pf_p_i_da
        dL_dxi_list[IDX_A][idx_plasma, i] = dm_pf_p_i_da
    if a > 0 and R0 > 0: dL_p_da = -MU0 * R0 / a
    else: dL_p_da = 0
    dL_dxi_list[IDX_A][idx_plasma, idx_plasma] = dL_p_da

    if R0 > 0:
        dL_p_dkappa = MU0 * R0 * 0.1
    else:
        dL_p_dkappa = 0
    dL_dxi_list[IDX_KAPPA][idx_plasma, idx_plasma] = dL_p_dkappa
    if R0 > 0:
        dL_p_ddelta = MU0 * R0 * 0.2
    else:
        dL_p_ddelta = 0
    dL_dxi_list[IDX_DELTA][idx_plasma, idx_plasma] = dL_p_ddelta
    if R0 > 0:
        dL_p_dli = MU0 * R0 * (1.0/2.0)
    else:
        dL_p_dli = 0
    dL_dxi_list[IDX_LI][idx_plasma, idx_plasma] = dL_p_dli
    return dL_dxi_list

def system_dynamics_tokamak(t, states, R0_matrix, N_ext_matrix, u_func,
                            L_func, dLdxi_k_list_f, plasma_state_func, dxi_dt_func,
                            l_coils_base_param, gs_cfg):
    I_vector = states
    xi_vector_val, R_p_val = plasma_state_func(t, states, gs_cfg)
    R_eff_matrix = np.copy(R0_matrix)
    R_eff_matrix[idx_plasma, idx_plasma] = R_p_val
    dxi_vector_dt_val = dxi_dt_func(t, states, xi_vector_val, gs_cfg)
    L_m = L_func(xi_vector_val, l_coils_base_param)
    dL_dxi_k_matrices = dLdxi_k_list_f(xi_vector_val, l_coils_base_param)

    dLdt_m = np.zeros_like(L_m)
    # Since dxi_dt is now zero for all components updated by GS, dLdt_m will be zero.
    # This simplifies the equation: L dI/dt = V_source - R_eff * I
    for k in range(NUM_XI_PARAMS):
        dLdt_m += dL_dxi_k_matrices[k] * dxi_vector_dt_val[k]

    u_vector_coils = u_func(t)
    V_source_vector = N_ext_matrix @ u_vector_coils

    try: L_inv_m = np.linalg.inv(L_m)
    except np.linalg.LinAlgError:
        print(f"Alerta: Matriz L singular em t={t:.4f}. Usando pseudo-inversa.")
        L_inv_m = np.linalg.pinv(L_m, rcond=1e-10)

    dI_dt = L_inv_m @ (V_source_vector - R_eff_matrix @ I_vector - dLdt_m @ I_vector)
    return dI_dt


# --- Função para salvar plots do Grad-Shafranov em cada chamada ---
def save_gs_plot(gs_instance, t, iteration):
    plt.figure(figsize=(8, 7))
    plt.contourf(gs_instance.R_vec, gs_instance.Z_vec, gs_instance.psi.T, levels=50, cmap='viridis')
    plt.colorbar(label='$\psi(R,Z)$ [Wb/rad]')
    if gs_instance.lcfs_contour_R is not None:
        plt.plot(gs_instance.lcfs_contour_R, gs_instance.lcfs_contour_Z, 'r-', linewidth=2)
    plt.title(f'Grad-Shafranov Solution at t={t:.3f}s')
    plt.xlabel('R [m]'); plt.ylabel('Z [m]')
    plt.axis('equal'); plt.grid(True)
    plt.savefig(f'gs_plots/gs_plot_t_{t:.3f}_iter_{iteration}.png')
    plt.close()

# --- Simulation Configuration ---
SIM_CONFIG = {
    "t_start": 0.0,
    "t_end": 2.0,
    "t_eval_points": 200,
    "initial_Ip": 1e2,
    "gs_call_interval": 0.01, # Default 0.1s
    "rtol": 1e-4,
    "atol": 1e-6
}
global_gs_config["call_interval"] = SIM_CONFIG["gs_call_interval"] # Update global GS config

# --- Simulation ---
t_start = SIM_CONFIG["t_start"]
t_end = SIM_CONFIG["t_end"]
t_eval = np.linspace(t_start, t_end, SIM_CONFIG["t_eval_points"])
initial_currents = np.zeros(num_circuits)
initial_currents[idx_plasma] = SIM_CONFIG["initial_Ip"]

print(f"Pre-simulation: Initializing GS cache with call_interval={global_gs_config['call_interval']}s...")
_ = call_gs_solver_periodically(t_start, initial_currents[idx_plasma], global_gs_config, force_run=True)

args_for_solver = (R_matrix_base, N_matrix, u_input_scenario,
                   L_matrix_model, dL_dxi_k_model_list,
                   get_plasma_state_vector_and_Rp, get_dxi_vector_dt,
                   _L_COILS_BASE, global_gs_config) # Pass gs_config dict

print("Starting Tokamak Simulation with Old GS Integration...")
sol = solve_ivp(
    system_dynamics_tokamak,
    (t_start, t_end),
    initial_currents,
    args=args_for_solver,
    t_eval=t_eval,
    method='BDF',
    rtol=SIM_CONFIG["rtol"],
    atol=SIM_CONFIG["atol"]
)
print("Simulation finished.")

# --- Plotting Results ---
print("Plotting results...")
fig, axs = plt.subplots(4, 1, figsize=(12, 18), sharex=True)

ax1 = axs[0]
for i in range(num_pf_coils): ax1.plot(sol.t, sol.y[i]/1e3, label=f'I_PF{i+1} (kA)')
for i in range(num_oh_coils): ax1.plot(sol.t, sol.y[num_pf_coils+i]/1e3, label=f'I_OH{i+1} (kA)', linestyle='--')
ax1.set_ylabel('Coil Currents (kA)')
ax1.set_title(f'Tokamak Sim w/ Old GS (GS Interval: {global_gs_config["call_interval"]}s)')
ax1.legend(loc='center left', bbox_to_anchor=(1,0.5)); ax1.grid(True)

ax2 = axs[1]
ax2.plot(sol.t, sol.y[idx_plasma]/1e3, label='I_plasma (kA)', color='k', linewidth=2)
ax2.set_ylabel('Plasma Current (kA)'); ax2.legend(loc='center left', bbox_to_anchor=(1,0.5)); ax2.grid(True)

ax3 = axs[2]
xi_labels = ['R0 (m)', 'a (m)', 'kappa', 'delta', 'li', 'beta_p']
# Re-initialize cache for plotting to get fresh values based on solution
global_gs_config["last_call_time"] = -1.0; global_gs_config["data"] = None
if sol.t[0] == t_start: _ = call_gs_solver_periodically(sol.t[0], sol.y[idx_plasma,0], global_gs_config, force_run=True)
xi_values_ts = np.array([get_plasma_state_vector_and_Rp(t, sol.y[:,i_t], global_gs_config)[0] for i_t, t in enumerate(sol.t)]).T
for i in range(NUM_XI_PARAMS): ax3.plot(sol.t, xi_values_ts[i], label=xi_labels[i])
ax3.set_ylabel('xi params (from GS)'); ax3.legend(loc='center left', bbox_to_anchor=(1,0.5)); ax3.grid(True)

ax4 = axs[3]
Rp_values_ts = np.array([get_plasma_state_vector_and_Rp(t, sol.y[:,i_t], global_gs_config)[1] for i_t,t in enumerate(sol.t)])
color_rp = 'tab:blue'
ax4.plot(sol.t, Rp_values_ts * 1e6, color=color_rp, linestyle='--', label=r'$R_p (\mu\Omega)$')
ax4.set_ylabel(r'$R_p (\mu\Omega)$', color=color_rp); ax4.tick_params(axis='y', labelcolor=color_rp)
ax4.legend(loc='center left', bbox_to_anchor=(1,0.5)); ax4.grid(True)
ax4.set_xlabel('Time (s)')
ax4.set_title('Plasma Resistance $R_p$')

plt.tight_layout(rect=[0,0,0.88,1])
output_plot_filename = './sim_tokamak_old_gs.png'
plt.savefig(output_plot_filename)
print(f"Simulação com GS antigo concluída. Gráfico salvo em {output_plot_filename}")
def create_video_from_plots():
    images = []
    filenames = sorted([f for f in os.listdir('gs_plots') if f.startswith('gs_plot')])
    
    for filename in filenames:
        images.append(imageio.imread(os.path.join('gs_plots', filename)))  # Usa imageio.v2
    
    # Salvar vídeo com FFmpeg (certifique-se de que está instalado)
    imageio.mimsave('gs_evolution.mp4', images, fps=10, codec='libx264')  # Especificar codec
    print("Vídeo salvo como 'gs_evolution.mp4'")

# Chamar a função após a simulação
create_video_from_plots()