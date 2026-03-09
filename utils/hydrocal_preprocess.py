import copy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer

import skfda
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import BSplineBasis

from utils.hydrocal_plot import Plot
from utils.hydrocal_dataloader import Data


class Preprocess():
    """ Preprocess hydration calorimetry data 
        - uses Data class to load data from specified directory
        - uses Plot class to plot raw data and preprocessing results
        - 
    """
    
    #---Preprocess attributes
    def __init__(self, data_directory, method_directory, extension='csv', delimiter=',') -> None:
        #--- define subclasses
        loaded = Data(data_directory, method_directory, extension=extension, delimiter=delimiter)

        #--- initialize subclass attributes
        loaded.parse_method()
        loaded.load_data()
        
        #--- define Preprocess class attributes
        self.metadata = loaded.metadata
        self.data = loaded.data
        self.plot = Plot(loaded)
        
        self.stats    = dict({})
        self.gradient = dict({})
        self.fft      = dict({})
        self.pca      = dict({})
        self.fftpca   = dict({})
        self.fpca     = dict({})

        #--- standard colors for data channels
        self.colors = ['0.1', 'k', 'r', 'b']
        plt.rcParams.update({'font.size': 14})


    #---Preprocess methods
    def min_max(self, data):
        """ min max normalization """
        min = np.min(data)
        max = np.max(data)
        delta = max - min
        norm = (data - min) / (max - min)
        
        return norm, delta, min, max
    
    
    def data_select(self, name, headers=['Weight', 'Ts', 'HF', 'RH']):
        """ select data of interest
        """        
        header_indices = np.arange(0, len(self.metadata[f'{name}_headers']), dtype=int)
        column_indices = dict(zip(self.metadata[f'{name}_headers'], header_indices))
        key_headers    = headers
        unit_indices   = [column_indices[h] for h in key_headers]
        key_units      = [self.metadata[f'{name}_units'][i] for i in unit_indices]

        return column_indices, key_headers, key_units
    
    
    def data_view(self, name, save=False, save_path=''):
        """ `data_view` displays key data
        """
        #--- define data of interest        
        column_indices, key_headers, key_units = self.data_select(name)

        _, axs = plt.subplots(4, 1, figsize=(30,10))
        for (ax, h, u, c) in zip(axs, key_headers, key_units, self.colors):
            
            ax.plot(self.data[name][column_indices['t']], self.data[name][column_indices[h]], color=c, lw=0.4)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(h+f' {u}')
            
        plt.tight_layout()
        plt.show

        if save:
            plt.savefig(save_path)
    
    
    def data_stats(self, name, method_name=None):
        """ `data_stats` method calculates standard statistics of the data
            
            Arguments:

            Attributes:

            Methods:
        """
        if method_name is None:
            method_name = self.metadata['method_file_names'][0]
        
        stats_data = self.data[str(name)]
        self.stats[name] = dict({})

        #--- define data of interest        
        column_indices, key_headers, key_units = self.data_select(name)
        data_length = len(stats_data[0])

        #--- loop over key headers and determine statistics
        for h in key_headers:
            self.stats[name].update({h+'_segment_means': []})
            self.stats[name].update({h+'_segment_stds' : []})
            self.stats[name].update({h+'_segment_maxs' : []})
            self.stats[name].update({h+'_segment_mins' : []})
            
            for start, stop in zip(self.metadata[method_name]['start_segment_indices'], self.metadata[method_name]['end_segment_indices']):
                mean = np.mean(stats_data[column_indices[h]][start:stop])
                std  = np.std( stats_data[column_indices[h]][start:stop])
                max  = np.max( stats_data[column_indices[h]][start:stop])
                min  = np.min( stats_data[column_indices[h]][start:stop])

                self.stats[name][h+'_segment_means'].append(mean)
                self.stats[name][h+'_segment_stds' ].append( std)
                self.stats[name][h+'_segment_maxs' ].append( max)
                self.stats[name][h+'_segment_mins' ].append( min)
        
        seg  = np.array(self.metadata[method_name]['segment'], dtype=float)
        
        #--- plot
        fig, axs = plt.subplots(4, 2, figsize=(30,10))
        
        for i, (ax, h, u, c) in enumerate(zip(axs, key_headers, key_units, self.colors)):
            ax0, ax1 = ax

            histogram = ax0.hist(stats_data[column_indices[h]], int(np.sqrt(data_length)), density=True, color=c)
            self.stats.update({h+'_histogram': histogram}) # save histograms to dictionary
            
            mean = np.array(self.stats[name][h+'_segment_means'])
            std  = np.array(self.stats[name][h+'_segment_stds' ])
            mins = np.array(self.stats[name][h+'_segment_mins' ])
            maxs = np.array(self.stats[name][h+'_segment_maxs' ])

            ax0.set_xlabel(    h)
            ax0.set_ylabel('PDF')

            ax1.errorbar(seg, mean, (mean-mins, maxs-mean ), color='r', alpha=0.5, lw=0, elinewidth=1)
            ax1.errorbar(seg, mean, (std, std), color='0.3', capsize=5, lw=0, elinewidth=1)
            ax1.scatter( seg, mean, color='k', marker='o')
            ax1.set_xlabel('segment')
            ax1.set_ylabel(f'{h} {u}')
        
        plt.tight_layout()    
        plt.show()


    def data_gradient(self, name, method_name=None, smoothing_window=[10], save=False, save_path=''):
        """ `data_view` displays key data
        """
        if method_name is None:
            method_name = self.metadata['method_file_names'][0]

        column_indices, key_headers, key_units = self.data_select(name)
        self.gradient[name] = dict({})

        kernel = []
        if smoothing_window:
            lu = len(column_indices)
            lsw = len(smoothing_window)
            if lu==lsw:
                for s in smoothing_window:
                    kernel.append(np.divide(np.ones(s), s, where=s>0, out=np.ones(s)))
            elif lsw == 1:
                kernel.append(np.ones(smoothing_window[0]) / smoothing_window[0])
                kernel = kernel * lu
                smoothing_window = smoothing_window * lu
            else:
                print(f'Smoothing window error. Number of window lengths ({len(smoothing_window)}) does not match number of data columns ({len(column_indices)})')
        else:
            lu = len(column_indices)
            kernel = kernel * lu
            smoothing_window = smoothing_window * lu

        fig, axs = plt.subplots(4, 1, figsize=(30,10))
        for i, (ax, h, u, c, k, w) in enumerate(zip(axs, key_headers, key_units, self.colors, kernel, smoothing_window)):
            smoothed_data = np.convolve(self.data[name][column_indices[h]], k, mode='same')
            gradient = np.gradient(smoothed_data)
            gradient[:w] = 0
            gradient[-w:] = 0
            self.gradient[name].update({h+'_gradient': gradient}) # save gradients to dictionary
            
            ax.plot(self.data[name][column_indices['t']], gradient,color=c, lw=0.1)
            ax.set_xlabel('t [s]')
            ax.set_ylabel(r'$\partial$'+h+r'/$\partial$t '+u)
            
        plt.tight_layout()
        plt.show

        if save:
            plt.savefig(save_path)


    def data_pca(self, name, method_name=None, recon_components=3, intersegment_fraction=0.03):
        """
        """
        if method_name is None:
            method_name = self.metadata['method_file_names'][0]

        column_indices, key_headers, key_units = self.data_select(name, headers=['Weight', 'HF'])
        self.pca[name] = dict({})
        segment_number = len(self.metadata[method_name]['segment_time'][1:])
        segment_lengths = np.zeros(segment_number)

        for i, s in enumerate(np.arange(0, segment_number)):
            segment_lengths[i] = np.sum(np.array(self.metadata[method_name]['segment_labels']) == int(s))
        
        max_length = np.max(segment_lengths).astype(int)

        #--- determine whether segment is increasing or decreasing RH
        starts, ends = self.metadata[method_name]['start_segment_indices'][1:], self.metadata[method_name]['end_segment_indices'][1:]
        startRH, endRH = self.metadata[method_name]['rh_profile'][starts], self.metadata[method_name]['rh_profile'][ends]
        signs = np.sign(startRH - np.roll(startRH, 1))
        self.pca[name].update({f'signs':signs.copy()})

        #--- initialize
        ndata_array = np.zeros((segment_number, max_length))
        dweight = np.zeros((segment_number))
        pca_weight = PCA()
        pca_HF     = PCA()
        pca = [pca_weight, pca_HF]

        fig1, axs1 = plt.subplots(2, 2, figsize=(30,10))
        fig2, axs2 = plt.subplots(2, 1, figsize=(30,10))
        fig3, axs3 = plt.subplots(2, 3, figsize=(30,10))
        for i, (ax1, ax2, ax3, h, u, p) in enumerate(zip(axs1, axs2, axs3, key_headers, key_units, pca)):
            ax11, ax12 = ax1
            ax31, ax32, ax33 = ax3

            #--- loop over segments
            colors_hot = plt.cm.plasma(np.arange(segment_number)/segment_number)
            for ii, (start, stop, sign, length, c) in enumerate(zip(starts, ends, signs, segment_lengths, colors_hot)):
                if intersegment_fraction > 0:
                    intersegment_shift = int(length * intersegment_fraction)
                    start -= intersegment_shift
                    stop -= intersegment_shift

                time    = self.metadata[method_name]['segment_time'][ii]
                segment = self.data[name][column_indices[h]][start:stop+1]
                ndata, delta, _, _ = self.min_max(segment)
                ndata *= sign

                if sign < 0:
                    ndata += 1 # shift negative data above 0
                
                ndata_length = len(ndata)
                if ndata_length < max_length:
                    ndata = np.pad(ndata, max_length-ndata_length)

                ndata_array[ii] = ndata
                dweight[ii] = delta

                ax11.plot(time, ndata, color=c)
                ax11.set_xlabel('time [s]')
                ax11.set_ylabel(f'reduced {h}')

            pca_fit = p.fit_transform(ndata_array.T)

            self.pca[name].update({f'{h}_ndata':ndata_array.copy(),
                                   f'{h}_pca':                   p,
                                   f'{h}_pca_fit':  pca_fit.copy(),
                                   f'{h}_pca_time':    time.copy(),
                                   f'{h}_delta':    dweight.copy()})


            #--- PC clustering
            #--- first, cluster PCs based on explained variance
            explained_variance_ratio         = p.explained_variance_ratio_
            d_explained_variance_ratio_plus  = np.diff(explained_variance_ratio) # discrete derivative
            d_explained_variance_ratio_plus  = np.insert(d_explained_variance_ratio_plus, 0, -1) # add zero as last element to preserve length
            d_explained_variance_ratio_minus = np.roll(d_explained_variance_ratio_plus, 1) # include slope behind, too
            dd_explained_variance_ratio      = d_explained_variance_ratio_minus - d_explained_variance_ratio_plus # second discrete derivative

            dbscan_data = np.array([explained_variance_ratio, d_explained_variance_ratio_plus, d_explained_variance_ratio_minus, dd_explained_variance_ratio]).T

            #--- DBSCAN
            dbscan = DBSCAN(eps=0.1, min_samples=2, metric='euclidean')
            dbscan.fit(dbscan_data)
            labels = dbscan.labels_
            unique_labels = np.sort(np.unique(labels)) # removes duplicates and sorts ascending
            n_clusters = len(unique_labels)
            cluster_indices = np.where(labels == unique_labels[0])[0]

            #--- 
            if labels[0] == -1: # first PC is outlier
                recon1 = (np.dot(pca_fit[:, 1:], p.components_[1:, :]) + p.mean_).T
                recon1_renorm, *rest = self.min_max(recon1)
                pt = PowerTransformer(method='yeo-johnson')
                recon1_renorm_transform = pt.fit_transform(recon1_renorm)

                pca2 = PCA()
                pca_fit2 = pca2.fit_transform(recon1_renorm_transform.T)
                
                self.pca[name].update({f'{h}_pca2': pca2})
                self.pca[name].update({f'{h}_pca_fit2': pca_fit2})


            elif labels[0] == 0: # first PC clusters with others
                # figure this out
                pass



            #--- Plotting
            for idx in cluster_indices:
                ax12.plot(time, pca_fit.T[idx])
                ax12.set_xlabel('time [s]')
                ax12.set_ylabel(f'reduced {h} PC')

            pcs = len(p.explained_variance_ratio_[:])
            pc_range = np.arange(pcs)

            ax31.scatter(pc_range, p.explained_variance_ratio_, color='k')
            ax31.set_xticks(pc_range)
            ax31.set_xlabel('PC')
            ax31.set_yscale('log')
            ax31.set_ylabel('explained variance')

            ax32.plot(p.components_.T[0] * p.singular_values_, label='10% RH')
            ax32.plot(p.components_.transpose()[4] * p.singular_values_, label='50% RH')
            ax32.plot(p.components_.transpose()[9] * p.singular_values_, label='100% RH')
            ax32.set_xticks(pc_range)
            ax32.set_xlabel('PC')
            ax32.set_ylabel('singular value')
            ax32.hlines([0],0,9, linestyles='--', color='0.75')
            ax32.legend(loc=1, fontsize='x-small')

            ax33.plot(p.components_[0], color='k', label="$PC_1$")
            ax33.plot(p.components_[1], color='r', label="$PC_2$")
            ax33.plot(p.components_[2], color='b', label="$PC_3$")
            ax33.plot(p.components_[3], color='0.8', label="$PC_4$")
            ax33.set_xticks(pc_range)
            ax33.set_xlabel('PC')
            ax33.set_ylabel('PC loading')
            ax33.hlines([0], 0, 9, linestyles='--', color='0.75')
            ax33.legend(loc=8, fontsize='x-small')

            recon_component_list = np.arange(recon_components)
            pca_recon = (np.dot(pca_fit[:, recon_component_list], p.components_[recon_component_list, :]) + p.mean_).T
            colors_hot = plt.cm.plasma(pc_range/pcs)
            for i, (pc, c) in enumerate(zip(pca_recon[0::2], colors_hot[0::2])):
                    ax2.plot(pc, color=c)
                    ax2.set_xlabel('time [s]')
                    ax2.set_ylabel('normalized '+h)

            self.pca[name].update({f'{h}_pca_recon': pca_recon.copy()})
            

        plt.show()


    def background_pca(self, name, method_name=None, recon_components=3, intersegment_fraction=0.03):
        """
        """
        if method_name is None:
            method_name = self.metadata['method_file_names'][0]

        column_indices, key_headers, key_units = self.data_select(name, headers=['Weight', 'HF'])
        self.pca[name] = dict({})
        segment_number = len(self.metadata[method_name]['segment_time'][1:])
        segment_lengths = np.zeros(segment_number)

        for i, s in enumerate(np.arange(0, segment_number)):
            segment_lengths[i] = np.sum(np.array(self.metadata[method_name]['segment_labels']) == int(s))

        #--- determine whether segment is increasing or decreasing RH
        starts, ends = self.metadata[method_name]['start_segment_indices'][1:], self.metadata[method_name]['end_segment_indices'][1:]
        startRH, endRH = self.metadata[method_name]['rh_profile'][starts], self.metadata[method_name]['rh_profile'][ends]
        signs = np.sign(startRH - np.roll(startRH, 1))
        self.pca[name].update({f'signs':signs.copy()})

        #--- initialize
        if intersegment_fraction > 0:
            intersegment_shift = int(segment_lengths[1] * intersegment_fraction)
        ndata_array = np.zeros((segment_number, 2*intersegment_shift))
        pca_weight = PCA()
        pca_HF     = PCA()
        pca = [pca_weight, pca_HF]

        fig1, axs1 = plt.subplots(2, 2, figsize=(30,10))
        fig2, axs2 = plt.subplots(2, 1, figsize=(30,10))
        fig3, axs3 = plt.subplots(2, 3, figsize=(30,10))
        for i, (ax1, ax2, ax3, h, u, p) in enumerate(zip(axs1, axs2, axs3, key_headers, key_units, pca)):
            ax11, ax12 = ax1
            ax31, ax32, ax33 = ax3

            #--- loop over segments
            colors_hot = plt.cm.plasma(np.arange(segment_number)/segment_number)
            for ii, (start, stop, sign, length, c) in enumerate(zip(starts, ends, signs, segment_lengths, colors_hot)):
                start -= intersegment_shift
                stop  -= intersegment_shift

                segment = self.data[name][column_indices[h]][start:stop+1]
                segment = np.delete(segment, slice(intersegment_shift, -intersegment_shift))
                ndata, delta, _, _ = self.min_max(segment)
                ndata *= sign

                if sign < 0:
                    ndata += 1 # shift negative data above 0
                
                ndata_array[ii] = ndata

                ax11.plot(ndata, color=c)
                ax11.set_xlabel('time [s]')
                ax11.set_ylabel(f'reduced {h}')

            pca_fit = p.fit_transform(ndata_array.T)

            self.pca[name].update({f'{h}_background_ndata':ndata_array.copy(),
                                   f'{h}_background_pca':                   p,
                                   f'{h}_background_pca_fit':  pca_fit.copy()})


            #--- Plotting
            ax12.plot(pca_fit.T[0])
            ax12.plot(pca_fit.T[1])
            ax12.plot(pca_fit.T[2])
            ax12.plot(pca_fit.T[3])
            ax12.set_xlabel('time [s]')
            ax12.set_ylabel(f'reduced {h} PC')

            pcs = len(p.explained_variance_ratio_[:])
            pc_range = np.arange(pcs)

            ax31.scatter(pc_range, p.explained_variance_ratio_, color='k')
            ax31.set_xticks(pc_range)
            ax31.set_xlabel('PC')
            ax31.set_yscale('log')
            ax31.set_ylabel('explained variance')

            ax32.plot(p.components_.T[0] * p.singular_values_, label='10% RH')
            ax32.plot(p.components_.transpose()[4] * p.singular_values_, label='50% RH')
            ax32.plot(p.components_.transpose()[9] * p.singular_values_, label='100% RH')
            ax32.set_xticks(pc_range)
            ax32.set_xlabel('PC')
            ax32.set_ylabel('singular value')
            ax32.hlines([0],0,9, linestyles='--', color='0.75')
            ax32.legend(loc=1, fontsize='x-small')

            ax33.plot(p.components_[0], color='k', label="$PC_1$")
            ax33.plot(p.components_[1], color='r', label="$PC_2$")
            ax33.plot(p.components_[2], color='b', label="$PC_3$")
            ax33.plot(p.components_[3], color='0.8', label="$PC_4$")
            ax33.set_xticks(pc_range)
            ax33.set_xlabel('PC')
            ax33.set_ylabel('PC loading')
            ax33.hlines([0], 0, 9, linestyles='--', color='0.75')
            ax33.legend(loc=8, fontsize='x-small')

            recon_component_list = np.arange(recon_components)
            pca_recon = (np.dot(pca_fit[:, recon_component_list], p.components_[recon_component_list, :]) + p.mean_).T
            colors_hot = plt.cm.plasma(pc_range/pcs)
            for i, (pc, c) in enumerate(zip(pca_recon[0::2], colors_hot[0::2])):
                    ax2.plot(pc, color=c)
                    ax2.set_xlabel('time [s]')
                    ax2.set_ylabel('normalized '+h)

            self.pca[name].update({f'{h}_background_pca_recon': pca_recon.copy()})
            

        plt.show()

        
        


    
    def data_fpca(self, 
                  name, 
                  method_name=None, 
                  pca_reconstruction_components=2, 
                  fpca_reconstruction_components=1, 
                  basis_type='Bspline', 
                  n_basis=10, 
                  order=4, 
                  regularization_parameter=1, 
                  intersegment_fraction=0.02
                  ):
        """
        """

        if method_name is None:
            method_name = self.metadata['method_file_names'][0]

        column_indices, key_headers, key_units = self.data_select(name, headers=['Weight', 'HF'])
        self.fpca[name] = dict({})
        segment_number = len(self.metadata[method_name]['segment_time'][1:])
        segment_lengths = np.zeros(segment_number)

        for i, s in enumerate(np.arange(0, segment_number)):
            segment_lengths[i] = np.sum(np.array(self.metadata[method_name]['segment_labels']) == int(s))
        
        max_length = np.max(segment_lengths).astype(int)
        ndata_array = np.zeros((segment_number, max_length))
        ndata_array_raw = np.zeros((segment_number, max_length))
        intersegment_shift_array = np.zeros(segment_number)

        starts, ends = self.metadata[method_name]['start_segment_indices'][1:], self.metadata[method_name]['end_segment_indices'][1:]
        startRH, endRH = self.metadata[method_name]['rh_profile'][starts], self.metadata[method_name]['rh_profile'][ends]
        signs = np.sign(startRH - np.roll(startRH, 1))

        fig, axs = plt.subplots(2,3, figsize=(30,5))
        #--- loop over data channel
        for (ax, h) in zip(axs, key_headers):
            ax1, ax2, ax3 = ax
            #--- loop over segments for each data channel
            for ii, (start, stop, sign, length) in enumerate(zip(starts, ends, signs, segment_lengths)):
                if intersegment_fraction > 0:
                    intersegment_shift = int(length * intersegment_fraction)
                    start -= intersegment_shift
                    stop -= intersegment_shift
                else:
                    intersegment_shift = 0 #used for spline knots

                intersegment_shift_array[ii] = intersegment_shift

                time    = self.metadata[method_name]['segment_time'][ii]
                segment = self.data[name][column_indices[h]][start:stop+1]
                ndata, _, _, _ = self.min_max(segment)
                ndata *= sign
                signed_segment = segment * sign

                if sign < 0: # shift negative data above 0
                    ndata += 1 
                    signed_segment -= np.min(signed_segment)

                if h == 'HF': # cumulative sum (approximate instantaneous heatflow integral)
                    ndata = np.cumsum(ndata - ndata[0])  
                    ndata, _, _, max = self.min_max(ndata)
                
                ndata_length = len(ndata)
                if ndata_length < max_length: # for segments with uneven length (may produce questionable results)
                    padding_length = max_length-ndata_length
                    ndata = np.pad(ndata, max_length-ndata_length)
                    print(f'Arrays of unqual length. Padding of length {padding_length} applied. Arrays may be offset in time')

                ndata_array[ii] = ndata
                ndata_array_raw[ii] = signed_segment

            fd = skfda.FDataGrid(
            data_matrix=ndata_array,
            grid_points=time,
            )

            self.fpca[name].update({f'fd_{h}':                                                   fd}) # save to dictionary
            self.fpca[name].update({f'ndata_{h}':                                ndata_array.copy()}) # save to dictionary
            self.fpca[name].update({f'ndata_raw_{h}':                        ndata_array_raw.copy()}) # save to dictionary
            self.fpca[name].update({f'intersegment_shift_array_{h}':intersegment_shift_array.copy()}) # save to dictionary
            
            #--- PCA
            pca = FPCA(n_components=segment_number)
            pca_transform = pca.fit_transform(fd)
            pca_transform[:, pca_reconstruction_components:] = 0 # zero out components not requested (save remaining components in pca object)
            pca_inverse_transform = pca.inverse_transform(pca_transform) # reconstruct with selected components

            self.fpca[name].update({f'pca_{h}':                                    pca}) # save to dictionary
            self.fpca[name].update({f'pca_transform_{h}':                pca_transform}) # save to dictionary
            self.fpca[name].update({f'pca_inverse_transform_{h}':pca_inverse_transform}) # save to dictionary

            #--- fPCA
            regularization = skfda.misc.regularization.L2Regularization(regularization_parameter=regularization_parameter)
            if basis_type=='Bspline': #--- B-spline basis
                #---- define basis
                knots = np.logspace(np.log10(intersegment_shift), np.log10(length), n_basis-order).astype(int)
                knots = np.insert(knots, 0, 1)
                knots = np.insert(knots, 1, intersegment_shift//2)
                basis = skfda.representation.basis.BSplineBasis(domain_range=(1, length), n_basis=n_basis, order=order, knots=knots)
                basis_fd = fd.to_basis(basis)
                #---- apply fPCA
                fpca = FPCA(n_components=segment_number-1, regularization=regularization)
                fpca_transform = fpca.fit_transform(basis_fd)
                fpca_transform[:, fpca_reconstruction_components:] = 0 #zero out coefficients not requested
                fpca_inverse_transform = fpca.inverse_transform(fpca_transform) # reconstruct with requested coefficients
                fpcad = fpca.mean_.derivative()

            elif basis_type=='Fourier': #--- Fourier basis
                #---- define basis
                basis = skfda.representation.basis.FourierBasis(domain_range=(1, length), n_basis=n_basis)
                basis_fd = fd.to_basis(basis)
                #---- apply fPCA
                fpca = FPCA(n_components=segment_number-1)
                fpca_transform = fpca.fit_transform(basis_fd)
                fpca_transform.coefficients[:, fpca_reconstruction_components:] = 0 #zero out coefficients not requested
                fpca_inverse_transform = fpca.inverse_transform(fpca_transform) # reconstruct with requested coefficients
                fpcad = fpca.mean_.derivative()
                
            #--- Save to dictionary
            self.fpca[name].update({f'fpca_{h}':                                    fpca}) # save to dictionary
            self.fpca[name].update({f'fpca_transform_{h}':                fpca_transform}) # save to dictionary
            self.fpca[name].update({f'fpca_inverse_transform_{h}':fpca_inverse_transform}) # save to dictionary

            #--- Plots
            pca.mean_.plot(chart=ax1, color='k')
            pca.components_.plot(chart=ax1,)
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel(f'{h} fPC mag')

            fpca.mean_.plot(chart=ax2, color='k')
            fpca.components_.plot(chart=ax2)
            ax2.set_xlabel('t [s]')
            ax2.set_ylabel(f'{h} fPC mag')

            fpcad.plot(chart=ax3, color='k')
            ax3.set_xlabel('t [s]')
            ax3.set_ylabel(f'{h}')

        plt.show()

        stime = time[intersegment_shift:]
        self.fpca[name].update({f'fpca_stime':stime}) # save to dictionary

        HF_raw = self.fpca[name]['ndata_raw_HF']
        lim_set = self.fpca[name]['fpca_HF'].explained_variance_ratio_[fpca_reconstruction_components]
        fpca_HF_array = self.fpca[name]['fpca_inverse_transform_HF'].derivative()(stime)[:,:,0]
        fpca_iHF_array = self.fpca[name]['fpca_inverse_transform_HF'](stime)[:,:,0]
        fpca_weight_array = self.fpca[name]['fpca_inverse_transform_Weight'](stime)[:,:,0]
        fpca_weight_change = self.fpca[name]['fpca_inverse_transform_Weight'].derivative()(stime)[:,:,0]

        heat_limits = np.zeros(segment_number)
        fig, axs = plt.subplots(segment_number, 3, figsize=(30,60))
        for i, (r, h, dr, dh, hf, ax) in enumerate(zip(fpca_weight_array, fpca_iHF_array, fpca_weight_change, fpca_HF_array, HF_raw, axs)):
            ax1, ax2, ax3 = ax

            scale_time = int(2 * np.pi * np.argmax(dr, keepdims=True)[0])
            
            r_minmax = r - r[0]
            h_minmax = h - h[0]
            scale = np.sum(r_minmax[:scale_time] * h_minmax[:scale_time]) / np.sum(r_minmax[:scale_time]**2)
            h_minmax /= scale

            hr_fraction =  2 * h_minmax[scale_time:] / (r_minmax[scale_time:] + h_minmax[scale_time:])
            hr_lim_arg = np.argmax(np.abs(1-hr_fraction) > lim_set, keepdims=True)
            lim = scale_time + hr_lim_arg
            heat_limits[i] = lim

            hf[intersegment_shift:] -= hf[intersegment_shift]
            hf_norm = hf[intersegment_shift:] / np.max(hf[intersegment_shift:])
            dh /= np.max(dh)

            ax1.plot(stime, r_minmax, color='k', label='weight')
            ax1.plot(stime, h_minmax, color='r', label='heat')
            ax1.set_xlabel('t [s]')
            ax1.set_ylabel('magnitude')
            ax1.legend()

            ax2.plot(stime[scale_time:], hr_fraction, color='k')
            ax2.scatter(stime[lim], hr_fraction[hr_lim_arg], color='r')
            ax2.set_xlabel('t [s]')
            ax2.set_ylabel('heat signal')
            
            ax3.plot(stime, dh, color='r')
            ax3.plot(stime, hf_norm, color='r', alpha=0.2)
            ax3.scatter(stime[lim], dh[lim], color='k')
            ax3.set_xlabel('t [s]')
            ax3.set_ylabel('HF')
        
        plt.tight_layout()
        plt.show()

        self.fpca[name].update({f'fpca_heat_limits': heat_limits}) # save to dictionary


    def data_fft(self, name, method_name=None, smoothing_window=[100], save=False, save_path=''):
        """
        """
        if method_name is None:
            method_name = self.metadata['method_file_names'][0]

        column_indices, key_headers, key_units = self.data_select(name)
        self.fft[name] = dict({})

        kernel = []
        if smoothing_window:
            lu = len(column_indices)
            lsw = len(smoothing_window)
            if lu==lsw:
                for s in smoothing_window:
                    kernel.append(np.divide(np.ones(s), s, where=s>0, out=np.ones(s)))
            elif lsw == 1:
                kernel.append(np.ones(smoothing_window[0]) / smoothing_window[0])
                kernel = kernel * lu
                smoothing_window = smoothing_window * lu
            else:
                print(f'Smoothing window error. Number of window lengths ({len(smoothing_window)}) does not match number of data columns ({len(column_indices)})')
        else:
            lu = len(column_indices)
            kernel = kernel * lu
            smoothing_window = smoothing_window * lu

        fig1, axs1 = plt.subplots(4, 2, figsize=(30,10))
        fig2, axs2 = plt.subplots(4, 2, figsize=(30,10))
        for i, (ax, axx, h, u, c, k, w) in enumerate(zip(axs1, axs2, key_headers, key_units, self.colors, kernel, smoothing_window)):
            data                 = self.data[name][column_indices[h]]
            data_length          = len(data)
            fft_frequency        = np.fft.rfftfreq(data_length) / self.metadata[method_name]['sampling_interval']
            fft_frequency_length = len(fft_frequency)

            if fft_frequency_length % 2 > 0: #odd number of data points
                fft_frequency_length = fft_frequency_length - 1

            fft_data = np.fft.fft(data)
            fft_power = np.abs(fft_data)
            fft_phase = np.angle(fft_data)
            smoothed_phase = np.convolve(fft_phase, k, mode='same')
            smoothed_phase[:w]  = 0
            smoothed_phase[-w:] = 0
            
            #--- save to dictionary
            self.fft[name].update({h+'_fft_frequency': fft_frequency}) # save to dictionary
            self.fft[name].update({h+'_fft_data':           fft_data}) # save to dictionary
            self.fft[name].update({h+'_fft_power':         fft_power}) # save to dictionary
            self.fft[name].update({h+'_fft_phase':         fft_phase}) # save to dictionary
            
            #--- inverse FFTs
            ifft_power = np.fft.ifft(fft_power)
            ifft_phase = np.fft.ifft(fft_phase)
            smoothed_iphase = np.convolve(ifft_phase, k, mode='same')

            #--- save to dictionary
            self.fft[name].update({h+'_ifft_power': ifft_power}) # save to dictionary
            self.fft[name].update({h+'_ifft_phase': ifft_phase}) # save to dictionary
            
            #--- shift fft for visualization
            fft_power_shift  = fft_power[:fft_frequency_length]
            fft_phase_shift  = fft_phase[:fft_frequency_length]
            ifft_power_shift = np.abs(ifft_power[:fft_frequency_length])
            ifft_phase_shift = np.abs(ifft_phase[:fft_frequency_length])
            smoothed_iphase_shift = np.abs(smoothed_iphase[:fft_frequency_length])

            time = self.data[name][column_indices['t']]
            #--- plot ffts
            ax0, ax1 = ax
            ax0.plot(fft_frequency[1:fft_frequency_length], np.log(fft_power_shift[1:]), color=c, lw=0.05)
            ax0.set_xlabel(r'1/t [$s^{-1}$]')
            ax0.set_ylabel(r'|$\mathcal{F}$('+h+')|')
            ax0.set_xscale('log')
            ax0.set_yscale('log')

            ax1.plot(fft_frequency[:fft_frequency_length], np.unwrap(fft_phase_shift[:]), color=c, lw=0.05)
            ax1.plot(fft_frequency[:fft_frequency_length], np.zeros_like(smoothed_phase[:fft_frequency_length]), color='k')
            ax1.set_xlabel(r'1/t [$s^{-1}$]')
            ax1.set_ylabel(r'arg|$\mathcal{F}$('+h+')|')

            #--- plot iffts
            ax0, ax1 = axx
            ax0.plot(time[1:fft_frequency_length], np.log(ifft_power_shift[1:]), color=c, lw=0.5)
            ax0.set_xlabel(r't [s]')
            ax0.set_ylabel(r'|$\mathcal{F}^{-1}$('+h+')|')
            ax0.set_xscale('log')
            ax0.set_yscale('log')

            ax1.plot(time[:fft_frequency_length], ifft_phase_shift[:fft_frequency_length], color=c, lw=0.5)
            ax1.set_xlabel(r't [s]')
            ax1.set_ylabel(r'${F}^{-1}$|$\mathcal{F}$('+h+')|')
            ax1.set_xscale('log')
            ax1.set_yscale('log')
            
        plt.tight_layout()
        plt.show

        if save:
            plt.savefig(save_path)


    def data_fftpca(self, name, method_name=None, recon_components=3):
        """
        """
        if method_name is None:
            method_name = self.metadata['method_file_names'][0]

        column_indices, key_headers, key_units = self.data_select(name, headers=['Weight', 'HF'])
        self.fftpca[name] = dict({})
        segment_number = len(self.metadata[method_name]['segment_time'])
        segment_lengths = np.zeros(segment_number)

        for i, s in enumerate(np.arange(1, segment_number)):
            segment_lengths[i] = np.sum(np.array(self.metadata[method_name]['segment_labels']) == int(s))
        
        max_length = np.max(segment_lengths).astype(int)

        #--- determine whether segment is increasing or decreasing RH
        starts, ends = self.metadata[method_name]['start_segment_indices'], self.metadata[method_name]['end_segment_indices']
        startRH, endRH = self.metadata[method_name]['rh_profile'][starts], self.metadata[method_name]['rh_profile'][ends]
        signs = np.sign(startRH - np.roll(startRH, 1))
        signs[0] = -1

        amp_array = np.zeros((segment_number, max_length))
        phase_array = np.zeros((segment_number, max_length))

        pca_weight_amp, pca_weight_phase = PCA(), PCA()
        pca_HF_amp,     pca_HF_phase     = PCA(), PCA()
        pca = [[pca_weight_amp, pca_weight_phase],[pca_HF_amp, pca_HF_phase]]

        fig1, axs1 = plt.subplots(4, 2, figsize=(30,10))
        fig2, axs2 = plt.subplots(2, 2, figsize=(30,10))
        fig3, axs3 = plt.subplots(4, 3, figsize=(30,10))
        for i, (ax0, ax1, ax2, ax3, ax4, h, u, p, c) in enumerate(zip(axs1[::2], axs1[1::2], axs2, axs3[::2], axs3[1::2], key_headers, key_units, pca, self.colors)):
            ax11, ax12 = ax0
            ax13, ax14 = ax1
            ax21, ax22 = ax2
            ax31, ax32, ax33 = ax3
            ax34, ax35, ax36 = ax4

            #--- loop over segments
            for ii, (start, stop, sign) in enumerate(zip(starts, ends, signs)):
                time    = self.metadata[method_name]['segment_time'][ii]
                segment = self.data[name][column_indices[h]][start:stop+1]
                ndata, delta, _, _ = self.min_max(segment)
                ndata *= sign
                
                if sign < 0:
                    ndata += 1 # shift negative data above 0
                ndata_length = len(ndata)

                if ndata_length < max_length:
                    ndata = np.pad(ndata, max_length-ndata_length)

                data_length          = len(ndata)
                fft_frequency        = np.fft.rfftfreq(data_length) / self.metadata[method_name]['sampling_interval']
                fft_frequency_length = len(fft_frequency)

                if fft_frequency_length % 2 > 0: #odd number of data points
                    fft_frequency_length = fft_frequency_length - 1

                fft_data = np.fft.fft(ndata)
                fft_power = np.abs(fft_data)
                fft_phase = np.angle(fft_data)

                amp_array[ii]   = fft_power
                phase_array[ii] = fft_phase
                
                #--- save to dictionary
                self.fftpca[name].update({h+'_fftpca_frequency': fft_frequency}) # save to dictionary
                self.fftpca[name].update({h+'_fftpca_data':           fft_data}) # save to dictionary
                self.fftpca[name].update({h+'_fftpca_power':         fft_power}) # save to dictionary
                self.fftpca[name].update({h+'_fftpca_phase':         fft_phase}) # save to dictionary

                ax11.plot(fft_frequency[:-1], fft_power[:fft_frequency_length])
                ax11.set_xlabel('1/s')
                ax11.set_ylabel(f'amp pcs {h}')
                ax11.set_xscale('log')
                ax11.set_yscale('log')

                ax12.plot(fft_frequency[:-1], np.unwrap(fft_phase[:fft_frequency_length]))
                ax12.set_xlabel('1/s')
                ax12.set_ylabel(f'phase pcs {h}')
                ax12.set_xscale('log')
            
            p1, p2 = p
            pca_fit_amp   = p1.fit_transform(  amp_array[:].T)
            pca_fit_phase = p2.fit_transform(phase_array[:].T)
            
            ax13.plot(fft_frequency[:-1], pca_fit_amp.T[2][:fft_frequency_length])
            ax13.plot(fft_frequency[:-1], pca_fit_amp.T[1][:fft_frequency_length])
            ax13.plot(fft_frequency[:-1], pca_fit_amp.T[0][:fft_frequency_length])
            ax13.set_xlabel('1/s')
            ax13.set_ylabel(f'amp pcs {h}')
            ax13.set_xscale('log')
            ax13.set_yscale('log')

            ax14.plot(fft_frequency[:-1], pca_fit_phase.T[2][:fft_frequency_length])
            ax14.plot(fft_frequency[:-1], pca_fit_phase.T[1][:fft_frequency_length])
            ax14.plot(fft_frequency[:-1], pca_fit_phase.T[0][:fft_frequency_length])
            ax14.set_xlabel('1/s')
            ax14.set_ylabel(f'phase pcs {h}')
            ax14.set_xscale('log')

            pcs1 = len(p1.explained_variance_ratio_[:])
            pc_range = np.arange(pcs1)

            ax31.scatter(pc_range, p1.explained_variance_ratio_, color='k')
            ax31.set_xticks(pc_range)
            ax31.set_xlabel('PC')
            ax31.set_yscale('log')
            ax31.set_ylabel('explained variance')

            ax34.scatter(pc_range, p1.explained_variance_ratio_, color='k')
            ax34.set_xticks(pc_range)
            ax34.set_xlabel('PC')
            ax34.set_yscale('log')
            ax34.set_ylabel('explained variance')

            ax32.plot(p1.components_.transpose()[0] * p1.singular_values_, label='10% RH')
            ax32.plot(p1.components_.transpose()[1] * p1.singular_values_, label='50% RH')
            ax32.plot(p1.components_.transpose()[2] * p1.singular_values_, label='100% RH')
            ax32.set_xticks(pc_range)
            ax32.set_xlabel('PC')
            ax32.set_ylabel('singular value')
            ax32.hlines([0],0,9, linestyles='--', color='0.75')
            ax32.legend(loc=1, fontsize='x-small')

            ax35.plot(p2.components_.transpose()[0] * p2.singular_values_, label='10% RH')
            ax35.plot(p2.components_.transpose()[1] * p2.singular_values_, label='50% RH')
            ax35.plot(p2.components_.transpose()[2] * p2.singular_values_, label='100% RH')
            ax35.set_xticks(pc_range)
            ax35.set_xlabel('PC')
            ax35.set_ylabel('singular value')
            ax35.hlines([0],0,9, linestyles='--', color='0.75')
            ax35.legend(loc=1, fontsize='x-small')

            ax33.plot(p1.components_[0], color='k', label="$PC_1$")
            ax33.plot(p1.components_[1], color='r', label="$PC_2$")
            ax33.plot(p1.components_[2], color='b', label="$PC_3$")
            ax33.plot(p1.components_[3], color='0.8', label="$PC_4$")
            ax33.set_xticks(pc_range)
            ax33.set_xlabel('RH')
            ax33.set_ylabel('PC loading')
            ax33.hlines([0], 0, 9, linestyles='--', color='0.75')
            ax33.legend(loc=8, fontsize='x-small')

            ax36.plot(p2.components_[0], color='k', label="$PC_1$")
            ax36.plot(p2.components_[1], color='r', label="$PC_2$")
            ax36.plot(p2.components_[2], color='b', label="$PC_3$")
            ax36.plot(p2.components_[3], color='0.8', label="$PC_4$")
            ax36.set_xticks(pc_range)
            ax36.set_xlabel('RH')
            ax36.set_ylabel('PC loading')
            ax36.hlines([0], 0, 9, linestyles='--', color='0.75')
            ax36.legend(loc=8, fontsize='x-small')

            recon_component_list = np.arange(recon_components)
            pca_recon_amp   = (np.dot(pca_fit_amp[:, recon_component_list],   p1.components_[recon_component_list, :]) + p1.mean_).T
            pca_recon_phase = (np.dot(pca_fit_phase[:, recon_component_list], p2.components_[recon_component_list, :]) + p2.mean_).T
            
            colors_hot = plt.cm.plasma(pc_range/pcs1)
            for i, (p, f, c) in enumerate(zip(pca_recon_amp[0::2], pca_recon_phase[0::2], colors_hot[0::2])):
                    ax21.plot(fft_frequency[:-1], p[:fft_frequency_length], color=c)
                    ax21.set_xlabel('1/s')
                    ax21.set_ylabel('amplitude '+h)
                    ax21.set_xscale('log')
                    ax21.set_yscale('log')

                    ax22.plot(fft_frequency[:-1], np.unwrap(f[:fft_frequency_length]), color=c)
                    ax22.set_xlabel('1/s')
                    ax22.set_ylabel('phase '+h)
                    # ax22.set_xscale('log')

        plt.show()