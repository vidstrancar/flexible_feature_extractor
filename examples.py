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

# EXAMPLE 1: ALL AVAIALBLE
#extractor = Extractor(data)
# extractor._extract_available()

# EXAMPLE 2: SPECIFIC FEATURES EXTRACTION
extractor = Extractor(data)
extractor._extract(['asc', 'skewness', 'kurtosis', 'MinMax', 'lala'])

# EXAMPLE 3: FEATURE EXTRACTION FROM ONE DAY DATA ONLY
# dayOneConsumption = consumption[:96]
# dayOneTemperature = temperature[:96]
# data = {'consumption': dayOneConsumption,
#         'temperature': dayOneTemperature}
# data.update(meta)
# extractor = Extractor(data)
# extractor._extract_available()

# # EXAMPLE 4: GRANULARITY = 1 DAY
# consumption = consumption.resample('D').mean()
# #temperature = temperature.resample('D').mean()
# data = {'consumption': consumption,
#         'temperature': temperature}
# data.update(meta)
# extractor = Extractor(data)
# extractor._extract_available()

# EXAMPLE 5: GRANULARITY = 1 WEEK
# consumption = consumption.resample('W').mean()
# temperature = temperature.resample('W').mean()
# data = {'consumption': consumption,
#         'temperature': temperature}
# data.update(meta)
# extractor = Extractor(data)
# extractor._extract_available()


# print(extractor.extracted)
# print(extractor.unavailable)
#df = pd.DataFrame([extractor.extracted])

for f in extractor.extracted:
    print(f'{f} {extractor.extracted[f]}')
for f in extractor.unavailable:
    print(f'{f}')
