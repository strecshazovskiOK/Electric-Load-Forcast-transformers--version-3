
# ...existing code...

def _prepare_time_series_data(self, data):
    # Normalize the data
    mean = data.mean()
    std = data.std()
    scaled_data = (data - mean) / std
    
    # Store normalization parameters for later use
    self.mean = mean
    self.std
    
    return scaled_data

# ...existing code...