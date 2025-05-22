import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.integrate import simpson
from skimage.measure import find_contours
from skimage.draw import polygon2mask

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class grad_shafranov_solver:
    def __init__(self):
        self.MU0 = 4 * np.pi * 1e-7
        self.NR_GS = 100
        self.NZ_GS = 100
        self.R_min_GS = 0.8
        self.R_max_GS = 3.2
        self.Z_min_GS = -1.2
        self.Z_max_GS = 1.2
        self.MAX_ITERATIONS_GS = 500
        self.CONVERGENCE_CRITERION_GS = 1e-4
        
        self.PRESSURE_PROFILE_TYPE = 'tanh'
        self.P0_TANH = 1.0e3 
        self.ALPHA_TANH = 5.0
        self.PSI_NORM_PEDESTAL_TANH = 0.8

        self.P0_PRIME_POLY_GS = 1.0e3
        self.FF0_PRIME_POLY_GS = 0.01
        self.ALPHA_P_POLY_GS = 1.0 
        self.ALPHA_F_POLY_GS = 1.0

        self.PSI_AXIS_TARGET_GS = -0.05 
        self.PSI_BOUNDARY_GS = 0.0
        
        self.R_grid = None
        self.Z_grid = None
        self.dR = None
        self.dZ = None
        self.R_vec = None
        self.Z_vec = None
        self.psi = None
        self.iteration_count = 0
        self.R0_calc = None
        self.a_calc = None
        self.kappa_calc = None
        self.delta_calc = None
        self.li_calc = None
        self.beta_p_calc = None
        self.Ip_calc = None
        self.lcfs_contour_R = None
        self.lcfs_contour_Z = None

        self.OMEGA_RELAXATION_GS = 1.0 
        self.min_iter_for_omega_update = 20 
        self.omega_estimated_flag = False
        self._psi_prev_for_rho = None
        self._psi_prev_prev_for_rho = None
        self.rho_J_estimated = None
        self.DEBUG_LOGGING_ITER_INTERVAL = 20

    def _setup_grid(self):
        self.R_vec = np.linspace(self.R_min_GS, self.R_max_GS, self.NR_GS)
        self.Z_vec = np.linspace(self.Z_min_GS, self.Z_max_GS, self.NZ_GS)
        self.dR = self.R_vec[1] - self.R_vec[0]
        self.dZ = self.Z_vec[1] - self.Z_vec[0]
        self.R_grid, self.Z_grid = np.meshgrid(self.R_vec, self.Z_vec, indexing='ij')

    def _p_profile(self, psi_val, psi_axis, psi_boundary):
        if np.abs(psi_axis - psi_boundary) < 1e-9:
            return np.zeros_like(psi_val)
        psi_norm = (psi_val - psi_boundary) / (psi_axis - psi_boundary)
        psi_norm = np.clip(psi_norm, 0, 1)
        if self.PRESSURE_PROFILE_TYPE == 'tanh':
            f_raw_psi_norm = 1.0 - np.tanh(self.ALPHA_TANH * (psi_norm - self.PSI_NORM_PEDESTAL_TANH))
            f_raw_0 = 1.0 - np.tanh(self.ALPHA_TANH * (0.0 - self.PSI_NORM_PEDESTAL_TANH))
            f_raw_1 = 1.0 - np.tanh(self.ALPHA_TANH * (1.0 - self.PSI_NORM_PEDESTAL_TANH))
            denominator = f_raw_1 - f_raw_0
            if np.abs(denominator) < 1e-9:
                pressure = self.P0_TANH * psi_norm if self.ALPHA_TANH == 0 else np.zeros_like(psi_norm)
            else:
                pressure = self.P0_TANH * (f_raw_psi_norm - f_raw_0) / denominator
            return pressure
        elif self.PRESSURE_PROFILE_TYPE == 'polynomial':
            if self.ALPHA_P_POLY_GS <= -1.0:
                logging.error("ALPHA_P_POLY_GS must be > -1.0 for polynomial pressure profile.")
                raise ValueError("ALPHA_P_POLY_GS must be > -1.0 for polynomial pressure profile.")
            return self.P0_PRIME_POLY_GS * (psi_norm ** (self.ALPHA_P_POLY_GS + 1.0))
        else:
            raise ValueError(f"Unknown PRESSURE_PROFILE_TYPE: {self.PRESSURE_PROFILE_TYPE}")

    def _p_prime_profile(self, psi_val, psi_axis, psi_boundary):
        if np.abs(psi_axis - psi_boundary) < 1e-9:
            return np.zeros_like(psi_val)
        psi_norm = (psi_val - psi_boundary) / (psi_axis - psi_boundary)
        psi_norm = np.clip(psi_norm, 0, 1)
        if self.PRESSURE_PROFILE_TYPE == 'tanh':
            f_raw_0 = 1.0 - np.tanh(self.ALPHA_TANH * (0.0 - self.PSI_NORM_PEDESTAL_TANH))
            f_raw_1 = 1.0 - np.tanh(self.ALPHA_TANH * (1.0 - self.PSI_NORM_PEDESTAL_TANH))
            denominator = f_raw_1 - f_raw_0
            if np.abs(denominator) < 1e-9:
                dp_dpsi_norm = self.P0_TANH * np.ones_like(psi_norm) if self.ALPHA_TANH == 0 else np.zeros_like(psi_norm)
            else:
                sech_sq_term = 1.0 - np.tanh(self.ALPHA_TANH * (psi_norm - self.PSI_NORM_PEDESTAL_TANH))**2
                df_raw_dpsi_norm = -self.ALPHA_TANH * sech_sq_term
                dp_dpsi_norm = self.P0_TANH * df_raw_dpsi_norm / denominator
            return dp_dpsi_norm
        elif self.PRESSURE_PROFILE_TYPE == 'polynomial':
            if self.ALPHA_P_POLY_GS <= -1.0:
                logging.error("ALPHA_P_POLY_GS must be > -1.0 for polynomial pressure profile.")
                raise ValueError("ALPHA_P_POLY_GS must be > -1.0 for polynomial pressure profile.")
            return -self.P0_PRIME_POLY_GS * (psi_norm ** self.ALPHA_P_POLY_GS)
        else:
            raise ValueError(f"Unknown PRESSURE_PROFILE_TYPE: {self.PRESSURE_PROFILE_TYPE}")

    def _FF_prime_profile(self, psi_val, psi_axis, psi_boundary):
        if np.abs(psi_axis - psi_boundary) < 1e-9:
            return np.zeros_like(psi_val)
        psi_norm = (psi_val - psi_boundary) / (psi_axis - psi_boundary)
        psi_norm = np.clip(psi_norm, 0, 1)
        return self.FF0_PRIME_POLY_GS * (psi_norm ** self.ALPHA_F_POLY_GS)

    def _initialize_psi(self):
        R0_guess = (self.R_min_GS + self.R_max_GS) / 2.0
        a_guess = (self.R_max_GS - self.R_min_GS) / 3.5
        kappa_guess = 1.7
        psi_initial = self.PSI_AXIS_TARGET_GS * (1.0 - ((self.R_grid - R0_guess)/a_guess)**2 - ((self.Z_grid)/(a_guess*kappa_guess))**2)
        psi_initial = np.clip(psi_initial, self.PSI_AXIS_TARGET_GS, self.PSI_BOUNDARY_GS) 
        return psi_initial

    def _apply_boundary_conditions(self, psi_array):
        psi_array[0, :] = self.PSI_BOUNDARY_GS
        psi_array[-1, :] = self.PSI_BOUNDARY_GS
        psi_array[:, 0] = self.PSI_BOUNDARY_GS
        psi_array[:, -1] = self.PSI_BOUNDARY_GS
        return psi_array

    def _update_omega_adaptive(self, psi_current_iter, psi_prev_iter, psi_prev_prev_iter):
        if psi_prev_iter is None or psi_prev_prev_iter is None: return
        delta_psi_current = (psi_current_iter - psi_prev_iter)[1:-1, 1:-1]
        delta_psi_prev = (psi_prev_iter - psi_prev_prev_iter)[1:-1, 1:-1]
        norm_delta_psi_current = np.linalg.norm(delta_psi_current)
        norm_delta_psi_prev = np.linalg.norm(delta_psi_prev)
        if norm_delta_psi_prev < 1e-12: return
        self.rho_J_estimated = norm_delta_psi_current / norm_delta_psi_prev
        if not (0 <= self.rho_J_estimated < 1.0):
            logging.warning(f"Estimated rho_J = {self.rho_J_estimated:.4f} invalid. Omega not updated.")
            return 
        if (1.0 - self.rho_J_estimated**2) <= 1e-9:
            omega_optimal = 1.99 
        else:
            omega_optimal = 2.0 / (1.0 + np.sqrt(1.0 - self.rho_J_estimated**2))
        self.OMEGA_RELAXATION_GS = np.clip(omega_optimal, 1.0, 1.99) 
        self.omega_estimated_flag = True
        logging.info(f"Iter {self.iteration_count}: Updated OMEGA to {self.OMEGA_RELAXATION_GS:.4f} (rho_J={self.rho_J_estimated:.4f})")

    def _solve_iteration(self, psi_old_full_step):
        psi_new_iter = np.copy(psi_old_full_step)
        error_sum_sq = 0.0
        psi_axis_current = np.min(psi_old_full_step)
        psi_boundary_current = self.PSI_BOUNDARY_GS
        psi_diff_norm = psi_axis_current - psi_boundary_current
        if np.abs(psi_diff_norm) < 1e-9: 
            psi_diff_norm = -1e-9 if psi_axis_current < psi_boundary_current else 1e-9
        
        for i in range(1, self.NR_GS-1):
            for j in range(1, self.NZ_GS-1):
                R_ij = self.R_grid[i, j]
                p_prime_norm_val = self._p_prime_profile(psi_old_full_step[i,j], psi_axis_current, psi_boundary_current)
                FF_prime_norm_val = self._FF_prime_profile(psi_old_full_step[i,j], psi_axis_current, psi_boundary_current)
                dp_dpsi = p_prime_norm_val / psi_diff_norm
                FdF_dpsi = FF_prime_norm_val / psi_diff_norm
                source_term = -self.MU0 * (R_ij**2) * dp_dpsi - FdF_dpsi
                psi_updated_val_jacobi = (
                    (psi_old_full_step[i+1,j]+psi_old_full_step[i-1,j])/self.dR**2 + 
                    (psi_old_full_step[i,j+1]+psi_old_full_step[i,j-1])/self.dZ**2 - 
                    (1.0/R_ij)*(psi_old_full_step[i+1,j]-psi_old_full_step[i-1,j])/(2.0*self.dR) - source_term 
                ) / (2.0/self.dR**2 + 2.0/self.dZ**2)
                psi_new_val_relaxed = (1.0 - self.OMEGA_RELAXATION_GS)*psi_old_full_step[i,j] + \
                                      self.OMEGA_RELAXATION_GS*psi_updated_val_jacobi
                error_sum_sq += (psi_new_val_relaxed - psi_old_full_step[i,j])**2
                psi_new_iter[i,j] = psi_new_val_relaxed
        
        if self.iteration_count == 1 or self.iteration_count % self.DEBUG_LOGGING_ITER_INTERVAL == 0:
            max_abs_source = np.max(np.abs(source_term)) if 'source_term' in locals() else 0
            logging.info(f"Iter {self.iteration_count} Debug: psi_axis={psi_axis_current:.3e}, psi_diff={psi_diff_norm:.3e}, max|src|={max_abs_source:.3e}")

        num_interior_points = (self.NR_GS-2)*(self.NZ_GS-2)
        error_rms = np.sqrt(error_sum_sq / num_interior_points) if num_interior_points > 0 else 0.0
        return psi_new_iter, error_rms

    def _extract_geometry_and_parameters(self):
        if self.psi is None: logging.error("Psi solution not available."); return

        psi_axis_val = np.min(self.psi)
        axis_idx_flat = np.argmin(self.psi)
        axis_idx = np.unravel_index(axis_idx_flat, self.psi.shape)
        R_axis = self.R_grid[axis_idx]
        Z_axis = self.Z_grid[axis_idx]
        self.R0_calc = R_axis

        contours = find_contours(self.psi, self.PSI_BOUNDARY_GS)
        plasma_mask = np.zeros(self.psi.shape, dtype=bool) # Initialize plasma_mask
        lcfs_contour_idx = None # Initialize lcfs_contour_idx

        if not contours:
            logging.warning("LCFS contour not found. Using fallback geometry and psi-based mask.")
            self.a_calc = (self.R_max_GS - self.R_min_GS) / 4.0
            self.kappa_calc = 1.0
            self.delta_calc = 0.0
            self.lcfs_contour_R, self.lcfs_contour_Z = None, None
            plasma_mask = (self.psi < self.PSI_BOUNDARY_GS) & (self.psi >= psi_axis_val)
        else:
            lcfs_contour_idx = max(contours, key=len)
            self.lcfs_contour_R = np.interp(lcfs_contour_idx[:, 0], np.arange(self.NR_GS), self.R_vec)
            self.lcfs_contour_Z = np.interp(lcfs_contour_idx[:, 1], np.arange(self.NZ_GS), self.Z_vec)

            R_lcfs_min = np.min(self.lcfs_contour_R)
            R_lcfs_max = np.max(self.lcfs_contour_R)
            Z_lcfs_min = np.min(self.lcfs_contour_Z)
            Z_lcfs_max = np.max(self.lcfs_contour_Z)
            self.a_calc = (R_lcfs_max - R_lcfs_min) / 2.0
            self.kappa_calc = (Z_lcfs_max - Z_lcfs_min) / (2.0 * self.a_calc + 1e-9) if self.a_calc > 1e-3 else 1.0
            
            R_at_Z_max = self.lcfs_contour_R[np.argmax(self.lcfs_contour_Z)]
            R_at_Z_min = self.lcfs_contour_R[np.argmin(self.lcfs_contour_Z)]
            delta_upper = (self.R0_calc - R_at_Z_max) / (self.a_calc + 1e-9)
            delta_lower = (self.R0_calc - R_at_Z_min) / (self.a_calc + 1e-9)
            self.delta_calc = (delta_upper + delta_lower) / 2.0
            
            # Create plasma mask using polygon2mask from LCFS contour indices
            if lcfs_contour_idx is not None:
                try:
                    lcfs_polygon_indices_for_mask = np.array([lcfs_contour_idx[:,0], lcfs_contour_idx[:,1]]).T.astype(int)
                    lcfs_polygon_indices_for_mask[:,0] = np.clip(lcfs_polygon_indices_for_mask[:,0], 0, self.NR_GS - 1)
                    lcfs_polygon_indices_for_mask[:,1] = np.clip(lcfs_polygon_indices_for_mask[:,1], 0, self.NZ_GS - 1)
                    plasma_mask = polygon2mask(self.psi.shape, lcfs_polygon_indices_for_mask)
                    if not plasma_mask[axis_idx]:
                        logging.warning("Magnetic axis outside LCFS-derived mask. Using psi < psi_boundary fallback.")
                        plasma_mask = (self.psi < self.PSI_BOUNDARY_GS) & (self.psi >= psi_axis_val)
                except Exception as e:
                    logging.error(f"Error creating mask from LCFS: {e}. Using psi < psi_boundary fallback.")
                    plasma_mask = (self.psi < self.PSI_BOUNDARY_GS) & (self.psi >= psi_axis_val)
            else: # Should not happen if contours were found, but as a safeguard
                plasma_mask = (self.psi < self.PSI_BOUNDARY_GS) & (self.psi >= psi_axis_val)

        if not np.any(plasma_mask):
            logging.warning("No plasma volume found (mask is all false). Parameters will be zero.")
            self.li_calc = 0.0; self.beta_p_calc = 0.0; self.Ip_calc = 0.0
            return

        psi_diff_norm_ip = psi_axis_val - self.PSI_BOUNDARY_GS
        if np.abs(psi_diff_norm_ip) < 1e-9: psi_diff_norm_ip = -1e-9
        p_prime_norm_grid = self._p_prime_profile(self.psi, psi_axis_val, self.PSI_BOUNDARY_GS)
        FF_prime_norm_grid = self._FF_prime_profile(self.psi, psi_axis_val, self.PSI_BOUNDARY_GS)
        dp_dpsi_grid = p_prime_norm_grid / psi_diff_norm_ip
        FdF_dpsi_grid = FF_prime_norm_grid / psi_diff_norm_ip
        J_phi_integrand = (self.R_grid * dp_dpsi_grid) + (FdF_dpsi_grid / (self.MU0 * self.R_grid + 1e-12))
        J_phi_integrand_masked = J_phi_integrand * plasma_mask
        integral_J_phi_dZ = simpson(J_phi_integrand_masked, dx=self.dZ, axis=1)
        self.Ip_calc = simpson(integral_J_phi_dZ, dx=self.dR, axis=0)

        p_profile_grid = self._p_profile(self.psi, psi_axis_val, self.PSI_BOUNDARY_GS)
        p_profile_masked = p_profile_grid * plasma_mask
        integrand_p_vol = p_profile_masked * 2 * np.pi * self.R_grid
        integral_p_vol_dZ = simpson(integrand_p_vol, dx=self.dZ, axis=1)
        integral_p_dV = simpson(integral_p_vol_dZ, dx=self.dR, axis=0)
        
        integrand_vol = (2 * np.pi * self.R_grid) * plasma_mask
        integral_vol_dZ = simpson(integrand_vol, dx=self.dZ, axis=1)
        plasma_volume = simpson(integral_vol_dZ, dx=self.dR, axis=0)
        avg_p_vol = integral_p_dV / (plasma_volume + 1e-12) if plasma_volume > 1e-9 else 0.0

        if abs(self.Ip_calc) < 1e-1 or self.a_calc < 1e-3 or plasma_volume < 1e-9:
            self.beta_p_calc = 0.0
        else:
            L_pol = 0.0
            if self.lcfs_contour_R is not None and len(self.lcfs_contour_R) > 1:
                L_pol = np.sum(np.sqrt(np.diff(self.lcfs_contour_R)**2 + np.diff(self.lcfs_contour_Z)**2))
            if L_pol < 1e-3:
                 L_pol = 2 * np.pi * self.a_calc * np.sqrt((1 + self.kappa_calc**2)/2.0)
            B_pa_sq = (self.MU0 * self.Ip_calc / (L_pol + 1e-9))**2
            self.beta_p_calc = avg_p_vol / (B_pa_sq / (2 * self.MU0) + 1e-12) if B_pa_sq > 1e-12 else 0.0

        dpsi_dR_full, dpsi_dZ_full = np.gradient(self.psi, self.dR, self.dZ)
        Bp_sq_grid = (1.0 / (self.R_grid**2 + 1e-12)) * (dpsi_dR_full**2 + dpsi_dZ_full**2)
        Bp_sq_masked = Bp_sq_grid * plasma_mask
        integrand_W_pol = (Bp_sq_masked / (2 * self.MU0)) * 2 * np.pi * self.R_grid
        integral_W_pol_dZ = simpson(integrand_W_pol, dx=self.dZ, axis=1)
        W_pol = simpson(integral_W_pol_dZ, dx=self.dR, axis=0)

        if abs(self.Ip_calc) < 1e-1 or self.R0_calc < 1e-3 or plasma_volume < 1e-9:
            self.li_calc = 0.0
        else:
            denominator_li = self.R0_calc * self.MU0 * self.Ip_calc**2
            self.li_calc = np.clip((4 * W_pol) / (denominator_li + 1e-12), 0.1, 5.0) if denominator_li > 1e-12 else 0.0

        logging.info(f"LCFS Geom: R0={self.R0_calc:.3f}m, a={self.a_calc:.3f}m, kappa={self.kappa_calc:.3f}, delta={self.delta_calc:.3f}")
        logging.info(f"Plasma Params (Simpson): Ip={self.Ip_calc/1e6:.3f}MA, beta_p={self.beta_p_calc:.3f}, li={self.li_calc:.3f}, Vol={plasma_volume:.3f}m^3")

    def solve(self):
        self._setup_grid()
        self.psi = self._initialize_psi()
        self.psi = self._apply_boundary_conditions(self.psi)
        self._psi_prev_for_rho = np.copy(self.psi)
        self._psi_prev_prev_for_rho = np.copy(self.psi)
        self.omega_estimated_flag = False
        self.OMEGA_RELAXATION_GS = 1.0 
        logging.info(f"Starting GS solver. Max iter: {self.MAX_ITERATIONS_GS}, Conv: {self.CONVERGENCE_CRITERION_GS:.1e}")
        logging.info(f"P0_TANH={self.P0_TANH:.1e}, FF0_PRIME_POLY_GS={self.FF0_PRIME_POLY_GS:.1e}")

        for k_iter in range(self.MAX_ITERATIONS_GS):
            self.iteration_count = k_iter + 1
            psi_before_iter_full_step = np.copy(self.psi)
            self.psi, error_rms = self._solve_iteration(self.psi) 
            self.psi = self._apply_boundary_conditions(self.psi)

            if not self.omega_estimated_flag and self.iteration_count >= self.min_iter_for_omega_update:
                self._update_omega_adaptive(self.psi, self._psi_prev_for_rho, self._psi_prev_prev_for_rho)
            
            if not self.omega_estimated_flag or self.iteration_count < self.min_iter_for_omega_update + 5:
                self._psi_prev_prev_for_rho = np.copy(self._psi_prev_for_rho)
                self._psi_prev_for_rho = np.copy(psi_before_iter_full_step)

            if self.iteration_count % self.DEBUG_LOGGING_ITER_INTERVAL == 0 or self.iteration_count == 1:
                 logging.info(f"Iter: {self.iteration_count}, RMS Err: {error_rms:.2e}, Omega: {self.OMEGA_RELAXATION_GS:.3f}, Psi Axis: {np.min(self.psi):.3e}")

            if error_rms < self.CONVERGENCE_CRITERION_GS:
                logging.info(f"Converged in {self.iteration_count} iter with RMS err {error_rms:.2e}")
                break
        else: 
            logging.warning(f"Max iter ({self.MAX_ITERATIONS_GS}) reached. RMS err {error_rms:.2e}")
        self._extract_geometry_and_parameters()
        return self.psi, self.R_grid, self.Z_grid, self.iteration_count, self.R0_calc, self.a_calc, self.kappa_calc, self.delta_calc, self.li_calc, self.beta_p_calc, self.Ip_calc

    def plot_solution(self, filename="gs_solution_adv_geom.png"):
        if self.psi is None: logging.error("Solution not available for plotting."); return
        plt.figure(figsize=(8, 7))
        plt.contourf(self.R_vec, self.Z_vec, self.psi.T, levels=50, cmap='viridis')
        plt.colorbar(label='$\psi(R,Z)$ [Wb/rad]')
        if self.lcfs_contour_R is not None and self.lcfs_contour_Z is not None:
            plt.plot(self.lcfs_contour_R, self.lcfs_contour_Z, 'r-', linewidth=2, label='LCFS (found)')
        else:
            plt.contour(self.R_vec, self.Z_vec, self.psi.T, levels=[self.PSI_BOUNDARY_GS], colors='r', linewidths=2, linestyles='--')
        
        if self.R0_calc is not None:
            axis_idx_flat = np.argmin(self.psi)
            axis_idx = np.unravel_index(axis_idx_flat, self.psi.shape)
            Z0_plt = self.Z_grid[axis_idx] 
            plt.plot(self.R0_calc, Z0_plt, 'kx', markersize=10, label=f'Mag. Axis ({self.R0_calc:.2f}, {Z0_plt:.2f})')
        
        plt.xlabel('R [m]'); plt.ylabel('Z [m]'); plt.title('Grad-Shafranov Solution: $\psi(R,Z)$')
        plt.axis('equal'); plt.grid(True); plt.legend()
        try:
            plt.savefig(filename)
            logging.info(f"Solution plot saved to {filename}")
        except Exception as e:
            logging.error(f"Error saving plot: {e}")
        plt.close()

if __name__ == '__main__':
    solver = grad_shafranov_solver()
    solver.NR_GS = 65
    solver.NZ_GS = 65
    logging.info("Running example GS solver with advanced geometry extraction...")
    results = solver.solve()
    logging.info(f"Solver finished in {results[3]} iterations.")
    logging.info(f"Final Params: R0={results[4]:.3f}m, a={results[5]:.3f}m, kappa={results[6]:.3f}, delta={results[7]:.3f}")
    logging.info(f"Ip={results[10]/1e6:.3f}MA, li={results[8]:.3f}, beta_p={results[9]:.3f}")
    logging.info(f"Psi_axis = {np.min(results[0]):.4e}, Psi_boundary = {solver.PSI_BOUNDARY_GS:.4e}")
    solver.plot_solution()

