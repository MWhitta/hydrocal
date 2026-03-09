import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import nnls
from scipy.optimize import curve_fit

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import PowerTransformer

from mpl_toolkits.axes_grid1.inset_locator import inset_axes


class Analyze():
    """ Analyze hydration calorimetry data 
        - uses Data class to load data from specified directory
        - uses Plot class to plot raw data and preprocessing results
    """
    
    #class attributes
    def __init__(self, preprocess_object) -> None:
        #--- define subclasses
        self.preprocess = preprocess_object
        self.laplace_transform = dict({})


    def n_exponential(self, t, *params):
        """
        """
        n_exp = len(params) // 2 

        y = np.zeros_like(t, dtype=float)
        for i in range(n_exp):
            A = params[2*i]
            tau = params[2*i + 1]
            y += A * np.exp(-t / tau)
        return y
    

    def n_exponential_linear(self, t, *params):
        """
        """
        n_exp = len(params[-2:]) // 2 

        y = np.zeros_like(t, dtype=float)
        for i in range(n_exp):
            A = params[2*i]
            tau = params[2*i + 1]
            y += A * np.exp(-t / tau)
        # add linear background
        a, b = params[-2:]
        y += a * t + b
        return y


    def inverse_laplace(self, t, y, tau, lam=1):
        """https://doi.org/10.1002/mrc.5453
        """
        A = np.exp(-np.outer(t, 1.0/tau))
        b = -y + 1
        min = np.min(-y)
        C = np.vstack((A, lam * np.eye(A.shape[1])))
        p = np.hstack((-y - min, np.zeros(A.shape[1])))
        x, rnorm = nnls(C,p)
        mc = A@x
        f = x
        return (tau, f, mc, rnorm)


    def decay_rates(self, decay_array):
        """ words
        """
        decay_peaks = (decay_array[:, 1:-1] > decay_array[:, :-2]) & \
                      (decay_array[:, 1:-1] > decay_array[:, 2:])
        return decay_peaks


    def inverse_laplace_fit(self, name, n=1000, t_lo=0, t_hi=6, lam=0.01):
        """ words
        """
        tau = np.logspace(t_lo, t_hi, n)

        #--- define input data from Preprocess object
        ndata_array_raw_weight = self.preprocess.fpca[name]['ndata_raw_Weight']
        ndata_array_raw_weight -= ndata_array_raw_weight[:, 0, None]
        ndata_array_raw_HF     = self.preprocess.fpca[name]['ndata_raw_HF']
        ndata_array_iHF        = np.cumsum(ndata_array_raw_HF - ndata_array_raw_HF[0], axis=1)
        
        stime                  = self.preprocess.fpca[name]['fpca_stime']
        fpca_weight_array      = np.array(self.preprocess.fpca[name]['fpca_inverse_transform_Weight'](stime)[:,:,0])
        fpca_iHF_array         = np.array(self.preprocess.fpca[name]['fpca_inverse_transform_HF'](stime)[:,:,0])

        heat_limits            = self.preprocess.fpca[name]['fpca_heat_limits'].astype(int)
        
        #--- initialize output arrays
        shape0 = fpca_iHF_array.shape[0]
        laplace_array_weight = np.zeros((shape0, len(tau)))
        laplace_array_heat   = np.zeros((shape0, len(tau)))

        #--- loop over data and compute inverse Laplace transform
        for i, (y1, w, y2, h, lim) in enumerate(zip(fpca_weight_array, ndata_array_raw_weight, fpca_iHF_array, ndata_array_iHF, heat_limits)):
            s1 = w[lim] / y1[lim]
            s2 = h[lim] / y2[lim]
            _, l1, _, _ = self.inverse_laplace(stime, s1*y1, tau, lam=lam)
            _, l2, _, _ = self.inverse_laplace(stime, s2*y2, tau, lam=lam)
            laplace_array_weight[i] = l1
            laplace_array_heat[i]   = l2
        
        tau_array = np.tile(tau[1:-1], (len(laplace_array_weight), 1))
        heat_decay_mask   = self.decay_rates(laplace_array_heat)
        weight_decay_mask = self.decay_rates(laplace_array_weight)

        #--- write results
        weight_decay = tau_array * weight_decay_mask
        weight_amp   = laplace_array_weight[:, 1:-1] * weight_decay_mask
        heat_decay   = tau_array * heat_decay_mask
        heat_amp     = laplace_array_heat[:, 1:-1] * heat_decay_mask
        
        #--- save results to dictionary
        self.laplace_transform[name] = dict({})
        self.laplace_transform[name].update({f'tau_array':         tau_array,
                                               f'weight_decay': weight_decay,
                                               f'weight_amp':     weight_amp,
                                               f'heat_decay':     heat_decay,
                                               f'heat_amp':         heat_amp})

        #--- plot results
        fig, axs = plt.subplots(1, 4, figsize=(30,5))
        ax1, ax2, ax3, ax4 = axs

        ax1.hist(weight_decay[weight_decay>0], bins=50, color='k')
        ax1.set_xlabel(r'$\tau_{weight}$ [s]')
        ax1.set_ylabel('count')
        
        ax2.hist(heat_decay[    heat_decay>0], bins=50, color='r')
        ax2.set_xlabel(r'$\tau_{heat}$ [s]')
        ax2.set_ylabel('count')

        ax3.plot(tau, laplace_array_weight[19], color='k')
        ax3.set_xscale('log')
        ax3.set_xlabel(r'$\tau_{weight}$ [s]')
        ax3.set_ylabel('amplitude')
        
        ax4.plot(tau, laplace_array_heat[19], color='r')
        ax4.set_xscale('log')
        ax4.set_xlabel(r'$\tau_{heat}$ [s]')
        ax4.set_ylabel('amplitude')

        plt.tight_layout()
        plt.show()


    def power_transform(self, bins=50):
        """ words
        """
        keys = self.laplace_transform.keys()
        self.yj_parameters = dict({})

        all_heat_decay   = []
        all_heat_amp     = []
        all_weight_decay = []
        all_weight_amp   = []

        for i, key in enumerate(keys):
            lt1 = self.laplace_transform[key]['heat_decay'].ravel()
            lt2 = self.laplace_transform[key]['heat_amp'].ravel()
            lt3 = self.laplace_transform[key]['weight_decay'].ravel()
            lt4 = self.laplace_transform[key]['weight_amp'].ravel()
            
            nz1 = np.flatnonzero(lt1)
            nz2 = np.flatnonzero(lt2)
            nz3 = np.flatnonzero(lt3)
            nz4 = np.flatnonzero(lt4)

            all_heat_decay.append(  lt1[nz1])
            all_heat_amp.append(    lt2[nz2])
            all_weight_decay.append(lt3[nz3])
            all_weight_amp.append(  lt4[nz4])

        all_heat_decay   = np.concatenate(  all_heat_decay)
        all_heat_amp     = np.concatenate(    all_heat_amp)
        all_weight_decay = np.concatenate(all_weight_decay)
        all_weight_amp   = np.concatenate(  all_weight_amp)

        pt1 = PowerTransformer(method='yeo-johnson')
        pt2 = PowerTransformer(method='yeo-johnson')
        pt3 = PowerTransformer(method='yeo-johnson')
        pt4 = PowerTransformer(method='yeo-johnson')

        heat_decay_transform   = pt1.fit_transform(all_heat_decay.reshape(-1, 1)  )
        heat_amp_transform     = pt2.fit_transform(all_heat_amp.reshape(-1, 1)    )
        weight_decay_transform = pt3.fit_transform(all_weight_decay.reshape(-1, 1))
        weight_amp_transform   = pt4.fit_transform(all_weight_amp.reshape(-1, 1)  )

        self.yj_parameters.update({f'heat_decay':    (heat_decay_transform[:,0], pt1),
                                    'heat_amp':        (heat_amp_transform[:,0], pt2),
                                    'weight_decay':(weight_decay_transform[:,0], pt3),
                                    'weight_amp':    (weight_amp_transform[:,0], pt4),
                                    })

        fig, axs = plt.subplots(1,4, figsize=(30,5))
        ax1, ax2, ax3, ax4 = axs

        ax1.hist(heat_decay_transform[:,0],   bins=bins, color='r')
        ax2.hist(heat_amp_transform[:,0],     bins=bins, color='r')
        ax3.hist(weight_decay_transform[:,0], bins=bins, color='k')
        ax4.hist(weight_amp_transform[:,0],   bins=bins, color='k')

        ax1.set_xlabel(r'transformed $\tau$')
        ax1.set_ylabel(r'count')
        ax2.set_xlabel(r'transformed A')
        ax2.set_ylabel(r'count')
        ax3.set_xlabel(r'transformed $\tau$')
        ax3.set_ylabel(r'count')
        ax4.set_xlabel(r'transformed A')
        ax4.set_ylabel(r'count')
        
        for ax in axs.ravel():
            ax.legend()        
        plt.tight_layout()
        plt.show()


    def parameter_guess(self, name):
        """
        """
        self.initial_guess = dict({}) 
        self.bounds        = dict({})

        segments        = len(self.preprocess.fpca[name]['fpca_heat_limits'])
        parameters      = self.yj_parameters
        keys            = list(parameters.keys())
        parameter_count = len(parameters['heat_decay'][0])
        components      = np.max((int(parameter_count / segments), 2)).astype(int)

        for key in keys:
            yj_data, yjf = parameters[key]
            gmm = GaussianMixture(n_components=components, random_state=0).fit(yj_data.reshape(-1,1))

            means = gmm.means_.flatten()
            stds = np.sqrt(gmm.covariances_.flatten())
            weights = gmm.weights_.flatten()

            lower_limit = means - 3 * stds
            upper_limit = means + 2 * stds # too large can create NaN

            means       = yjf.inverse_transform(      means.reshape(-1, 1))
            weights     = yjf.inverse_transform(    weights.reshape(-1, 1))
            lower_limit = yjf.inverse_transform(lower_limit.reshape(-1, 1))
            upper_limit = yjf.inverse_transform(upper_limit.reshape(-1, 1))

            self.initial_guess[key] = means[:, 0]
            self.bounds[key]        = (lower_limit[:,0], upper_limit[:,0])


    def exponential_fit(self, name):
        """ words
        """
        if not hasattr(self, 'initial_guess'):
            self.parameter_guess()

        #--- define input data from Preprocess object
        interseg_shift_array   = self.preprocess.fpca[name]['intersegment_shift_array_HF'].astype(int)[1:]
        ndata_array_raw_weight = self.preprocess.fpca[name]['ndata_raw_Weight']
        ndata_array_raw_HF     = self.preprocess.fpca[name]['ndata_raw_HF']

        stime                  = self.preprocess.fpca[name]['fpca_stime']
        heat_limits            = self.preprocess.fpca[name]['fpca_heat_limits'].astype(int)

        fit_time = stime - stime[0]
        fig, axs = plt.subplots(len(ndata_array_raw_weight)-1, 4, figsize=(30, 3*len(ndata_array_raw_weight)))
        #--- fit exponentials
        for i, (w_dat, h_dat, ishift, lim, ax) in enumerate(zip(ndata_array_raw_weight, 
                                                    ndata_array_raw_HF, 
                                                    interseg_shift_array,
                                                    heat_limits,                                                    
                                                    axs)):
            ax1, ax2, ax3, ax4 = ax

            # --- weight fit
            weight_data = -(w_dat[ishift:] - np.max(w_dat[ishift:]))
            weight_decay_guesses = self.initial_guess['weight_decay']
            weight_amp_guesses   = self.initial_guess['weight_amp']
            weight_decay_bounds  = self.bounds['weight_decay']
            weight_amp_bounds    = self.bounds['weight_amp']
            
            initial_guesses_weight = np.array([element for pair in zip(weight_amp_guesses, weight_decay_guesses) for element in pair])
            bounds_lower_weight  = [element for pair in zip(weight_amp_bounds[0], weight_decay_bounds[0]) for element in pair]
            bounds_upper_weight  = [element for pair in zip(weight_amp_bounds[1], weight_decay_bounds[1]) for element in pair]
            bounds_weight = (bounds_lower_weight, bounds_upper_weight)

            try:
                weight_params, weight_params_covariance = curve_fit(self.n_exponential, fit_time[:lim], weight_data[:lim], p0=initial_guesses_weight, bounds=bounds_weight)
                weight_fitted_curve = self.n_exponential(fit_time, *weight_params)
                weight_residuals = weight_data - weight_fitted_curve
            
            except RuntimeError as e:
                print(f'RuntimeError encountered: {e}')

            # --- heat fit
            heat_data = h_dat[ishift:] - h_dat[ishift:][0]
            heat_decay_guesses = self.initial_guess['heat_decay']
            heat_amp_guesses   = self.initial_guess['heat_amp']
            heat_decay_bounds  = self.bounds['heat_decay']
            heat_amp_bounds    = self.bounds['heat_amp']
            
            initial_guesses_heat = np.array([element for pair in zip(heat_amp_guesses, heat_decay_guesses) for element in pair])

            bounds_lower_heat  = [element for pair in zip(-heat_amp_bounds[0], heat_decay_bounds[0]) for element in pair]
            bounds_upper_heat  = [element for pair in zip( heat_amp_bounds[1], heat_decay_bounds[1]) for element in pair]
            bounds_heat = (bounds_lower_heat, bounds_upper_heat)

            try:
                heat_params, heat_params_covariance = curve_fit(self.n_exponential, fit_time[:], heat_data[:], p0=initial_guesses_heat)
                heat_fitted_curve = self.n_exponential(fit_time, *heat_params)
                heat_residuals = weight_data - heat_fitted_curve
                print(f'segment: {i}')
                print(f'weight_params: {weight_params}')
                print(f'heat_params: {heat_params}')

            except RuntimeError as e:
                print(f'RuntimeError encountered: {e}')

            try:
                #--- plot results
                ax1.plot(fit_time[:lim], weight_data[:lim], color='k')
                ax1.plot(fit_time, weight_data, color='k', alpha=0.3)
                ax1.plot(fit_time, weight_fitted_curve, color='r')

                ax3.plot(fit_time[:lim], heat_data[:lim], color='k')
                ax3.plot(fit_time, heat_data, color='k', alpha=0.3)
                ax3.plot(fit_time, heat_fitted_curve, color='r')
            except UnboundLocalError as u:
                print(f'UnboundLocalError encountered: {u}')

        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('weight [mg]')
        ax1.set_xlabel('t [s]')
        ax1.set_ylabel('HF [mW]')
        plt.tight_layout()
        plt.plot()