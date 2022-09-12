from featureExtractor import featureExtractor as Extractor

fe = Extractor('data/2017-2018-consumption-30-users.csv',
               'data/2017-2018-temperature.csv',
               'data/meta.json',
               'data/features_to_extract.csv',
               'data/features_extracted.csv')

fe.extract()
fe.save()
