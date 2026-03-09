import os
import re
import codecs
import numpy as np

class Data:
    """ 'Data' is a dataloader class
        Files are optionally parsed and sorted, then loaded
        Load can be either into RAM (default) or memory mapped (memmap=True) for large datasets
        
        Argumentss:
            data_dir (str): data directory path from which to load all data contained within
            memmap (bool): whether to use memory mapping. Default is False
            memmap_filename (str): name of memory map file. Default is 'memmap.dat'
            dtype (type): data type. Default is 'float'
        
        Attributes:
            metadata ()
            method ()
            data (numpy array):
        
        Methods:
            extract_order
            order_sort
            load_data
            get_data
            close_memmap
    """

    def __init__(
        self,
        data_directory,
        method_directory='../methods/',
        extension='csv',
        label='',
        delimiter=',',
        memmap=False,
        memmap_filename='memmap.dat',
        dtype=float,
        ):
        
        #--- initialize metadata dictionary
        self.metadata = {'data_directory': data_directory,
                        'method_directory': method_directory,
                        'extension': extension,
                        'label': label,
                        'delimiter': delimiter,
                        'memmap': memmap,
                        'memmap_filename': memmap_filename,
                        'dtype': dtype,
                         }

    
    # --- define method loading functions ------------------------------------------
    def load_method(self):
        lib_path = self.metadata['method_directory']
        method_files = []
        for root, directory, files in os.walk(lib_path):
            for file in files:
                if file.endswith('.' + self.metadata['extension']):
                    method_files.append(os.path.join(root, file))

        sorted_method_files = self.order_sort(method_files)
        self.metadata.update({'method_file_paths': sorted_method_files})

    
    def import_method(self, name=None):
        """ function to import the method file"""
        self.method = dict({})
        self.metadata.update({'method_file_names': []})

        for dir in self.metadata['method_file_paths']:
            n = os.path.basename(dir)
            n = os.path.splitext(n)[0]
            self.metadata['method_file_names'].append(n)
            try: # try reading file
                codecs.open(dir, encoding="utf-8", errors="ignore").readlines()
                self.method[n] = open(dir, encoding='utf-8').read()
            except UnicodeDecodeError: # if file can't be read, convert to utf-8
                with open(dir, 'r', encoding='cp1252') as f:
                    text = f.read()
                with open(dir, 'w', encoding='utf-8') as f:
                    f.write(text)
                    self.method[n] = open(dir, encoding='utf-8').read()
    
    
    def parse_method(self):
        """ Parse a TGA/DSC method file into constituents for vizualization and analysis.
            Methods depict the temperature, relative humidity, and gas flow rate over a specified duration ...
                for a given number of segments
            Global variables are held constant over the course of an experiment
            Segment variables may change (or not) between segments
        """
        if not hasattr(self, 'method'):
            self.load_method()
            self.import_method()
        
        for name in self.metadata['method_file_names']:
            self.metadata[name] = dict({})
            #--- initialize lists for segment metadata
            segment, duration, duration_unit, rh_set, temperature_set = [], [], [], [], []
            
            #--- search method document for segments, find location (element number) and number of segments 
            lines = self.method[name].split('\n')
            for i, line in enumerate(lines):
                if "sampling interval" in line.lower():
                    sampling_interval = float(re.findall(r"\d+\.?\d*",line)[0])
                    sampling_unit = line.split(' ')[-1]
                    self.metadata[name].update({"sampling_interval":sampling_interval, 
                                        "sample_unit":sampling_unit})
                
                if "segment" in line.lower():
                    segment.append(line.split(' ')[-1])
                    self.metadata[name].update({"segment":segment})
                    
                if "duration" in line.lower():
                    duration.append(float(re.findall(r"\d+\.?\d*",line)[0]))
                    duration_unit.append(line.split(' ')[-1])
                    self.metadata[name].update({"duration":duration, 
                                        "duration_unit":duration_unit})
                    
                if "rel. humidity" in line.lower():
                    rh_set.append([re.findall(r"\d+\.?\d*",line)[0]])
                    self.metadata[name].update({"rh_set":rh_set})
                    
                if any(segment):
                    if "temperature" in line.lower():
                        temperature_set.append([re.findall(r"\d+\.?\d*",line)[0]])
                        self.metadata[name].update({"temperature_set":temperature_set})
            
            #--- convert time units to seconds
            for i, dur_unit in enumerate(self.metadata[name]['duration_unit']):
                sample_unit = self.metadata[name]['sample_unit']
                if dur_unit == sample_unit:
                    pass
                elif sample_unit == 's':
                    if dur_unit == 'min':
                        self.metadata[name]['duration'][i] = self.metadata[name]['duration'][i] * 60
                elif sample_unit == 'min': # not sure if this would happen
                    if dur_unit == 's':
                        self.metadata[name]['duration'][i] = self.metadata[name]['duration'][i] / 60

            #--- make list of segment numbers for each datapoint, facilitating segmentwise operations
            segment_labels = [float(num) * np.ones(int(float(dur) / self.metadata[name]['sampling_interval'])) for dur, num in zip(self.metadata[name]['duration'], self.metadata[name]['segment'])]
            segment_labels = [int(t) for seg in segment_labels for t in seg]
            self.metadata[name].update({'segment_labels':segment_labels})
                    
            #--- put segment times into one list, and then take the cumulative sum for the running experiment time
            segment_time = [np.arange(self.metadata[name]['sampling_interval'], dur + 1, self.metadata[name]['sampling_interval']) for dur in self.metadata[name]['duration']]
            self.metadata[name].update({'segment_time':segment_time})
            
            time = [np.ones(int(float(dur) /  self.metadata[name]['sampling_interval'])) for dur in self.metadata[name]['duration']] # this might break for different sample rates
            time = np.array([t for seg in time for t in seg])
            time = np.cumsum(time)
            self.metadata[name].update({'time':time})

            #--- find indeces corresponding to the end of each segment
            end_segment_indices = np.cumsum([len(seg) for seg in self.metadata[name]['segment_time']]).astype(int) - 1 # recast as int for indexing
            self.metadata[name].update({'end_segment_indices':end_segment_indices})
            
            #--- find indeces corresponding to the start of each segment
            start_segment_indices = np.roll(end_segment_indices, 1).astype(int) + 1
            start_segment_indices[0] = 0 # add first datapoint
            self.metadata[name].update({'start_segment_indices':start_segment_indices})
            
            #--- make relative humidity profiles
            rh_profile = []
            for seg, rh in zip(self.metadata[name]['segment_time'], self.metadata[name]['rh_set']):
                rh_profile.append(rh * len(seg))
            rh_profile = np.array([rh for seg in rh_profile for rh in seg]).astype('float64') #flatten
            self.metadata[name].update({'rh_profile':rh_profile})
            
            #--- make temperature profiles
            temperature_profile = []
            for seg, t in zip(self.metadata[name]['segment_time'], self.metadata[name]['temperature_set']):
                temperature_profile.append(t * len(seg))
            temperature_profile = np.array([t for seg in temperature_profile for t in seg]).astype('float64') #flatten
            self.metadata[name].update({'temperature_profile':temperature_profile})


    # --- define data loading functions ------------------------------------------

    def extract_order(self, filename):
        match = re.search(fr'(\d+)-{self.metadata['label']}\.{self.metadata['extension']}$', filename)
        if match:
            return float(match.group(1))
        return 0

    def order_sort(self, names):
        return sorted(names, key=lambda filename: self.extract_order(filename))

    def load_data(self):
        lib_path = self.metadata['data_directory']
        csv_files = []
        for root, directory, files in os.walk(lib_path):
            for file in files:
                if file.endswith('.' + self.metadata['extension']):
                    csv_files.append(os.path.join(root, file))

        sorted_csv_files = self.order_sort(csv_files)
        self.metadata.update({'file_names': sorted_csv_files})
        num_files = len(sorted_csv_files)
        
        #--- find the shape of the data by loading one file
        sample_data = np.loadtxt(open(sorted_csv_files[0],'rt', encoding="cp1252").readlines()[:-1], dtype=str, skiprows=2, encoding="cp1252",)
        n, m = sample_data.shape # features, channels
        self.metadata.update({'shape':(num_files, m, n)})

        if self.metadata['memmap']:
            self.data = np.memmap(self.metadata['memmap_filename'], dtype=self.metadata['dtype]'], mode='w+', shape=self.metadata['shape'])
        else:
            self.data = {}
        
        for i, file_path in enumerate(sorted_csv_files):
            f = open(file_path, 'rt', encoding='cp1252')
            data = np.loadtxt(f.readlines()[:-1], skiprows=2, dtype=None, encoding="cp1252",)
            data_name = os.path.basename(file_path)
            data_name = os.path.splitext(data_name)[0]
            data = np.swapaxes(data, 0, 1) # samples, channels, features
            f.close

            metadata = np.loadtxt(file_path, dtype=str, max_rows=2, encoding="cp1252", comments=None) # needs string dtype to parse
            headers = metadata[0]
            units   = metadata[1]
            
            f = open(file_path, 'rt', encoding='cp1252')
            method = f.readlines()[-1]
            _, date = method.split(',')

            self.data.update({f'{data_name}':data})
            self.metadata.update({f'{data_name}_headers': headers})
            self.metadata.update({f'{data_name}_units':     units})
            self.metadata.update({f'{data_name}_date':       date})
            

    def get_data(self):
        return self.data

    def close_memmap(self):
        if self.metadata['memmap'] and self.data is not None:
            self.data._mmap.close()