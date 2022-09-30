from featureExtractor import Extractor
import pandas as pd

data = {"name": "test", "morning_start": 6, "morning_end": 10, "noon_start": 10, "noon_end": 14, "afternoon_start": 14, "afternoon_end": 18,
        "evening_start": 18, "evening_end": 22, "night_start": 1, "night_end": 6, "ht_start": 6, "ht_end": 22, "nt_start": 22, "nt_end": 6, "neighborhood_width": 3}

consumption = pd.read_csv('data/consumption.csv',
                          parse_dates=True, index_col=0).squeeze()
temperature = pd.read_csv('data/temperature.csv',
                          parse_dates=True, index_col=0).squeeze()

data['consumption'] = consumption
data['temperature'] = temperature

extractor = Extractor(data)

extractor._extract_available()

# ekstrahirane znacilke se shranijo v slovar extractor.extracted
for k in extractor.extracted:
    print(f'{k}: {extractor.extracted[k]}')

# imena neuspesno ekstrahiranih znacilk in razlogi za neuspeh se shranijo v
# slovar extractor._unavailable
for f in extractor.unavailable:
    print(f)
