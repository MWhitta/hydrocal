import os
import re
import glob
import numpy as np
import pandas as pd

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
        method_extension='txt',
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
                        'method_extension': method_extension,
                        'label': label,
                        'delimiter': delimiter,
                        'memmap': memmap,
                        'memmap_filename': memmap_filename,
                        'dtype': dtype,
                         }
    
    # Constants for file structure
    _HEADER_ROWS = 2
    _ENCODING = 'cp1252'

    
    # --- Helper methods -------------------------------------------------------
    def _find_files_by_extension(self, directory, extension=None):
        """Find all files with target extension in directory recursively."""
        if not os.path.exists(directory):
            return []
        if extension is None:
            extension = self.metadata['extension']
        pattern = os.path.join(directory, '**', f'*.{extension}')
        return glob.glob(pattern, recursive=True)
    
    def _read_file_safe(self, filepath, encoding='utf-8'):
        """Read file with fallback encoding if UTF-8 fails."""
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with cp1252 as fallback
            with open(filepath, 'r', encoding='cp1252') as f:
                content = f.read()
            # Convert file to UTF-8
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return content
    
    # --- Define method loading functions -----------------------------------------
    def load_method(self):
        """Load and sort method file paths from method directory."""
        lib_path = self.metadata['method_directory']
        method_files = self._find_files_by_extension(lib_path, extension=self.metadata['method_extension'])
        sorted_method_files = self.order_sort(method_files)
        self.metadata.update({'method_file_paths': sorted_method_files})

    
    def import_method(self, name=None):
        """Import method files with safe encoding handling."""
        self.method = dict({})
        self.metadata.update({'method_file_names': []})

        for filepath in self.metadata['method_file_paths']:
            method_name = os.path.splitext(os.path.basename(filepath))[0]
            self.metadata['method_file_names'].append(method_name)
            self.method[method_name] = self._read_file_safe(filepath)
    
    
    @staticmethod
    def _parse_setpoint(line):
        """Parse a method line to extract setpoint value(s) and determine if static or dynamic.

        Static example:  'Rel. humidity : 0.00 %'        -> (0.0, 0.0, 'static')
        Dynamic example: 'Rel. humidity : 0.00 - 100.00 %' -> (0.0, 100.0, 'dynamic')

        Returns:
            tuple: (start_value, end_value, segment_type)
        """
        values = re.findall(r"\d+\.?\d*", line)
        if '-' in line.split(':')[-1] and len(values) >= 2:
            return float(values[0]), float(values[1]), 'dynamic'
        else:
            return float(values[0]), float(values[0]), 'static'

    def parse_method(self):
        """ Parse a TGA/DSC method file into constituents for vizualization and analysis.
            Methods depict the temperature, relative humidity, and gas flow rate over a specified duration ...
                for a given number of segments
            Global variables are held constant over the course of an experiment
            Segment variables may change (or not) between segments

            Segments are classified as 'static' or 'dynamic':
                - Static: a single setpoint held constant (e.g., 'Rel. humidity : 0.00 %')
                - Dynamic: a linear ramp between two values (e.g., 'Rel. humidity : 0.00 - 100.00 %')
        """
        if not hasattr(self, 'method'):
            self.load_method()
            self.import_method()

        for name in self.metadata['method_file_names']:
            self.metadata[name] = dict({})
            #--- initialize lists for segment metadata
            segment, duration, duration_unit = [], [], []
            rh_start, rh_end, rh_type = [], [], []
            temp_start, temp_end, temp_type = [], [], []
            segment_type = []  # overall segment type (dynamic if any variable ramps)

            #--- search method document for segments, find location (element number) and number of segments
            lines = self.method[name].split('\n')
            current_seg_rh_type = 'static'
            current_seg_temp_type = 'static'

            for i, line in enumerate(lines):
                if "sampling interval" in line.lower():
                    sampling_interval = float(re.findall(r"\d+\.?\d*",line)[0])
                    sampling_unit = line.split(' ')[-1]
                    self.metadata[name].update({"sampling_interval":sampling_interval,
                                        "sample_unit":sampling_unit})

                if "segment" in line.lower():
                    # Before appending a new segment, save the type of the previous one
                    if segment:
                        seg_kind = 'dynamic' if (current_seg_rh_type == 'dynamic' or current_seg_temp_type == 'dynamic') else 'static'
                        segment_type.append(seg_kind)
                    segment.append(line.split(' ')[-1])
                    current_seg_rh_type = 'static'
                    current_seg_temp_type = 'static'

                if "duration" in line.lower():
                    duration.append(float(re.findall(r"\d+\.?\d*",line)[0]))
                    duration_unit.append(line.split(' ')[-1])

                if "rel. humidity" in line.lower():
                    start, end, kind = self._parse_setpoint(line)
                    rh_start.append(start)
                    rh_end.append(end)
                    rh_type.append(kind)
                    current_seg_rh_type = kind

                if any(segment):
                    if "temperature" in line.lower():
                        start, end, kind = self._parse_setpoint(line)
                        temp_start.append(start)
                        temp_end.append(end)
                        temp_type.append(kind)
                        current_seg_temp_type = kind

            # Save the type of the last segment
            if segment:
                seg_kind = 'dynamic' if (current_seg_rh_type == 'dynamic' or current_seg_temp_type == 'dynamic') else 'static'
                segment_type.append(seg_kind)

            self.metadata[name].update({
                "segment": segment,
                "duration": duration,
                "duration_unit": duration_unit,
                "rh_start": rh_start,
                "rh_end": rh_end,
                "rh_type": rh_type,
                "temp_start": temp_start,
                "temp_end": temp_end,
                "temp_type": temp_type,
                "segment_type": segment_type,
            })

            # Keep backwards-compatible rh_set and temperature_set as list-of-lists
            rh_set = [[s] if t == 'static' else [s, e] for s, e, t in zip(rh_start, rh_end, rh_type)]
            temperature_set = [[s] if t == 'static' else [s, e] for s, e, t in zip(temp_start, temp_end, temp_type)]
            self.metadata[name].update({"rh_set": rh_set, "temperature_set": temperature_set})

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

            #--- make relative humidity profiles (linear ramp for dynamic, constant for static)
            rh_profile = []
            for seg, rs, re_, rt in zip(self.metadata[name]['segment_time'], rh_start, rh_end, rh_type):
                n_points = len(seg)
                if rt == 'dynamic':
                    rh_profile.append(np.linspace(rs, re_, n_points))
                else:
                    rh_profile.append(np.full(n_points, rs))
            rh_profile = np.concatenate(rh_profile)
            self.metadata[name].update({'rh_profile':rh_profile})

            #--- make temperature profiles (linear ramp for dynamic, constant for static)
            temperature_profile = []
            for seg, ts, te_, tt in zip(self.metadata[name]['segment_time'], temp_start, temp_end, temp_type):
                n_points = len(seg)
                if tt == 'dynamic':
                    temperature_profile.append(np.linspace(ts, te_, n_points))
                else:
                    temperature_profile.append(np.full(n_points, ts))
            temperature_profile = np.concatenate(temperature_profile)
            self.metadata[name].update({'temperature_profile':temperature_profile})


    # --- Define data loading functions ------------------------------------------
    def extract_order(self, filepath):
        """Extract numeric order from filename for sorting."""
        filename = os.path.basename(filepath)
        # Look for leading digits in filename
        match = re.match(r'(\d+)', filename)
        return float(match.group(1)) if match else 0

    def order_sort(self, filepaths):
        """Sort filepaths by numeric order in filename."""
        return sorted(filepaths, key=self.extract_order)

    def load_data(self):
        """Load and parse data files from directory."""
        lib_path = self.metadata['data_directory']
        csv_files = self._find_files_by_extension(lib_path)

        sorted_csv_files = self.order_sort(csv_files)
        self.metadata.update({'file_names': sorted_csv_files})
        num_files = len(sorted_csv_files)

        if num_files == 0:
            self.data = {}
            return

        delimiter = self.metadata['delimiter']

        if self.metadata['memmap']:
            # Read first file to determine shape for memmap
            df0 = pd.read_csv(sorted_csv_files[0], delimiter=delimiter,
                              header=None, skiprows=self._HEADER_ROWS,
                              encoding=self._ENCODING)
            num_samples, num_channels = df0.shape
            self.metadata.update({'shape': (num_files, num_channels, num_samples)})
            self.data = np.memmap(self.metadata['memmap_filename'],
                                dtype=self.metadata['dtype'],
                                mode='w+',
                                shape=self.metadata['shape'])
        else:
            self.data = {}

        # Load each data file
        for file_path in sorted_csv_files:
            # Read headers (first two rows)
            header_df = pd.read_csv(file_path, delimiter=delimiter,
                                    header=None, nrows=self._HEADER_ROWS,
                                    encoding=self._ENCODING)
            headers = header_df.iloc[0].fillna('').values.astype(str)
            units = header_df.iloc[1].fillna('').values.astype(str)

            # Fill empty header names using unit labels (e.g., [%] -> RH)
            _UNIT_TO_HEADER = {'[%]': 'RH'}
            for i, h in enumerate(headers):
                if h.strip() == '' and i < len(units) and units[i] in _UNIT_TO_HEADER:
                    headers[i] = _UNIT_TO_HEADER[units[i]]

            # Read data (skip header rows)
            data_df = pd.read_csv(file_path, delimiter=delimiter,
                                  header=None, skiprows=self._HEADER_ROWS,
                                  encoding=self._ENCODING)
            data = data_df.values.astype(float)
            data_name = os.path.splitext(os.path.basename(file_path))[0]
            data = data.T  # Transpose to (channels, samples)

            # Store in data dictionary
            self.data.update({data_name: data})
            self.metadata.update({
                f'{data_name}_headers': headers,
                f'{data_name}_units': units,
            })
        if 'shape' not in self.metadata:
            first_name = os.path.splitext(os.path.basename(sorted_csv_files[0]))[0]
            d = self.data[first_name]
            self.metadata['shape'] = (num_files, d.shape[0], d.shape[1])
            

    def get_data(self):
        return self.data

    def close_memmap(self):
        if self.metadata['memmap'] and self.data is not None:
            self.data._mmap.close()