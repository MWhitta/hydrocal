import numpy as np
import matplotlib.pyplot as plt


class Plot():
    """ Preprocess hydration calorimetry data 
        - uses Data class to load data from specified directory
        - 
        - 
    """
    
    #class attributes
    def __init__(self, data_object) -> None:
        self.metadata = data_object.metadata
        self.data = data_object.data


    def plot_method(self, method_name=None, save=False, save_path='', *kwargs):
            """ Plot the temperature and rh methods """

            if method_name is None:
                method_name = self.metadata['method_file_names'][0]
            
            # retrieve variables from method dictionary
            time=self.metadata[f'{method_name}']['time']
            rh_profile=self.metadata[f'{method_name}']['rh_profile']
            temperature_profile=self.metadata[f'{method_name}']['temperature_profile']
            start_segment_indices=self.metadata[f'{method_name}']['start_segment_indices']
            end_segment_indices=self.metadata[f'{method_name}']['end_segment_indices']

            fig, axs = plt.subplots(1,2, figsize=(30,5))
            ax0, ax1 = axs
            
            plt.rcParams.update({'font.size': 14})
            ax0.plot(time, rh_profile, color='b')
            ax0.scatter(time[start_segment_indices], rh_profile[start_segment_indices],    color='g', marker='o')
            ax0.scatter(time[end_segment_indices],   rh_profile[end_segment_indices],      color='r', marker='o')
            ax0.set_xlabel('time [min]')
            ax0.set_ylabel('relative humidity [%]')

            ax1.plot(time, temperature_profile, color='r')
            ax1.scatter(time[start_segment_indices], temperature_profile[start_segment_indices], color='g', marker='o')
            ax1.scatter(time[end_segment_indices],   temperature_profile[end_segment_indices],   color='r', marker='o')
            ax1.set_xlabel('time [min]')
            ax1.set_ylabel(r'reference temperature [$\degree$C]')

            plt.show

    
    def plot_stats(self):
         """ Plot the statistics
         """