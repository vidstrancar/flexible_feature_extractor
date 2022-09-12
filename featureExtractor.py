import pandas as pd
import json
import csv

from featureExtractorSingleUser import Extractor


#ERROR CHECKING SE CIMPREJ
class featureExtractor:
    def __init__(self, consumptions, temperature, meta, features_to_extract, out):
        self.consumptions = pd.read_csv(consumptions, parse_dates=True, index_col=0)
        self.ids = self.consumptions.columns
        self.temperature = pd.read_csv(temperature, parse_dates=True, index_col=0).squeeze()
        self.meta = json.load(open(meta))
        self.features_to_extract = open(features_to_extract).read().splitlines()
        self.out = out
        self.additional_data = {'temperature': self.temperature,
                                'meta': self.meta}

        
    def extract(self):
        additional_data = self.additional_data
        features_to_extract = self.features_to_extract
        extracted_all = []
        for id in self.ids:
            additional_data['id'] = id
            consumption = self.consumptions[id]
            extractor = Extractor(consumption, additional_data)
            if features_to_extract[0] == 'all_available':
                extractor._extract_available()
            else:
                extractor._extract(features_to_extract)
            extracted = extractor.features
            extracted_all.append(extracted)
        extracted_all = pd.DataFrame(extracted_all)
        if 'id' in extracted_all.columns:
            extracted_all.index = extracted_all['id']
            del extracted_all['id']
        self.extracted_all = extracted_all
        

    def save(self):
        self.extracted_all.to_csv(self.out)
    
        
    def extracted_to_pd_dataframe(self):
        return self.extracted_all


