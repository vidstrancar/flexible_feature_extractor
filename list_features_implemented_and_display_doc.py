from featureExtractor import Extractor
import pandas as pd

data = {"name": "test", "morning_start": 6, "morning_end": 10, "noon_start": 10, "noon_end": 14, "afternoon_start": 14, "afternoon_end": 18,
        "evening_start": 18, "evening_end": 22, "night_start": 1, "night_end": 6, "ht_start": 6, "ht_end": 22, "nt_start": 22, "nt_end": 6, "neighborhood_width": 3}

data['consumption'] = consumption
data['temperature'] = temperature

extractor = Extractor(data)

fs_implemented = [f for f in dir(extractor) if f[0] != '_']
print(f'Features implemented: {fs_implemented}\n')

print(f'MinMax is: {extractor.MinMax.__doc__}')
