import logging
import functools

import numpy as np
import pandas as pd
import json
from scipy.optimize import curve_fit
from scipy.signal import correlate, find_peaks, peak_widths, argrelextrema
from scipy.ndimage import uniform_filter1d 


def line(X, k, n):
    return np.array([x*k + n for x in X])

def hockeyStick(X, k1, n1, k2, n2):
    lowpoint = (n2-n1)/(k1-k2)
    return np.array([k1*x + n1 if x < lowpoint else k2*x + n2 for x in X])

def dropna_and_index_intersect(df1, df2):
    i1 = df1.dropna().index
    i2 = df2.dropna().index
    inBoth = i1.intersection(i2)
    return df1.loc[inBoth], df2.loc[inBoth]


class Extractor:
    def __init__(self, data = {}, main='consumption'):

        self.data = data
        self.granularity = (data[main].index[1] - data[main].index[0]).total_seconds()/60.0

        
        self.features = {}
        self.features_min_granularity = {}
        self.features_need_data = {}
        self.features_need_features = {}

        methods_available = [m for m in dir(self) if callable(getattr(self, m))]
        self.features_defined = [m for m in methods_available if not m.startswith('_')]

        if 'id' in self.data:
            self.features['id'] = self.data['id']

        

    # feature extraction
    def _extract_available(self):
        all = [m for m in dir(self) if (callable(getattr(self, m)) and not m.startswith('_'))]
        unavailable = []
        for m in all:
            try:
                getattr(self, m)()
            except Exception as e:
                unavailable.append({m:e})
        self.unavailable = unavailable
        self.extracted = self.features
                                
    def _extract(self, features):
        unavailable = []
        for feature in features:
            try:
                getattr(self, feature)()
            except Exception as e:
                unavailable.append({feature: e})
        self.unavailable = unavailable
        self.extracted = self.features
            

            
    # decorators
    def _min_granularity(min_gran):
        # checks if there is enough granularity for feature
        def decorator_min_granularity(func):
            @functools.wraps(func)
            def wrapper(self):
                self.features_min_granularity.update({func.__name__: min_gran})
                if self.granularity > min_gran:
                    raise Exception(f'{func.__name__} needs granularity less than {min_gran} min')
                return func(self)
            return wrapper
        return decorator_min_granularity
    
    def _import_data(names):
        """ import names from additional data as local variables if exist. 
        if not try create one."""
        def decorator_needs_additional_data(func):
            @functools.wraps(func)
            def wrapper(self):
                self.features_need_data.update({func.__name__: names})
                for name in names:
                    if name in self.data:
                        globals()[name] = self.data[name]
                    elif (hasattr(self, '_'+name)
                        and callable(getattr(self, '_'+name))):
                        logging.info(f'{func.__name__} needs data: {name}. generating.')
                        globals()[name] = getattr(self, '_'+name)()
                    else:
                        raise Exception(f'{func.__name__} needs data: {name}. not defined.')
                result = func(self)
                for name in names:
                    if name in globals():
                        globals().pop(name)
                return result
            return wrapper
        return decorator_needs_additional_data
                    
    def _import_features(features):
        """ import features as local variables if exist. if not try create one. """
        def decorator_needs_features(func):
            @functools.wraps(func)
            def wrapper(self):
                self.features_need_features.update({func.__name__: features})
                for feature in features:
                    if feature in self.features:
                        globals()[feature] = self.features[feature]
                    elif (hasattr(self, feature)
                        and callable(getattr(self, feature))):
                        logging.info(f'{func.__name__} needs {feature}. generating.')
                        # mogoce se ulovi izjeme??
                        getattr(self, feature)()
                        globals()[feature] = self.features[feature]
                    else:
                        raise Exception(f'{func.__name__} needs {feature}. not defined')
                result = func(self)
                for name in features:
                    globals().pop(name)
                return result
            return wrapper
        return decorator_needs_features


    
    # features decorators
    def _check_if_exists_and_save_feature(func):
        """ check if feature exists. if doesn't, extract and save it to dict self.features """
        @functools.wraps(func)
        def wrapper(self):
            if not(func.__name__ in self.features):
                result = func(self)
                self.features[func.__name__] = result
                return result
            else:
                pass
        return wrapper

    

    # data decorators
    def _check_if_exists_and_save_data(func):
        """ check if data exists. """
        @functools.wraps(func)
        def wrapper(self):
            if not(func.__name__ in self.features):
                result = func(self)
                self.data[func.__name__[1:]] = result
                return result
            else:
                pass
        return wrapper


    
    # data generators
    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_check_if_exists_and_save_data
    def _days(self):
        return consumption.index.weekday.values

    
    @_min_granularity(24*60)
    @_import_data(['consumption', 'days'])
    @_check_if_exists_and_save_data
    def _weekdays(self):
        return days < 5     


    @_min_granularity(24*60)
    @_import_data(['consumption', 'days'])
    @_check_if_exists_and_save_data
    def _weekends(self):
        return days >= 5


    @_min_granularity(24*60)
    @_check_if_exists_and_save_data
    def _samples_in_day(self):
        return int(24*60/self.granularity)

    
    @_min_granularity(7*24*60)
    @_import_data(['consumption'])
    @_check_if_exists_and_save_data
    def _samples_in_week(self):
        return int(7*24*60/self.granularity)


    @_import_data(['consumption'])
    @_check_if_exists_and_save_data
    def _n_days(self):
        sid = 24*60/self.granularity
        return int(len(consumption)/sid)

    
    @_import_data(['consumption'])
    @_check_if_exists_and_save_data
    def _n_weeks(self):
        siw = 7*24*60/self.granularity
        return int(len(consumption)/siw)


    @_min_granularity(24*60)
    @_import_data(['consumption', 'n_weeks', 'samples_in_week'])
    @_check_if_exists_and_save_data
    def _average_week(self):
        c = np.nan_to_num(consumption.values)
        c = c[:n_weeks*samples_in_week]
        c = c.reshape((n_weeks, samples_in_week))
        avg_week = np.average(c, axis = 0)
        #use times of first week
        avg_week_times = consumption.index[:samples_in_week]
        avg_week = pd.Series(avg_week)
        avg_week.index = avg_week_times
        return avg_week
        
        
    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_check_if_exists_and_save_data
    def _hours(self):
        return consumption.index.hour.values

    
    @_min_granularity(60)
    @_import_data(['hours', 'morning_start', 'morning_end'])
    @_check_if_exists_and_save_data
    def _mornings(self):
        return (hours >= morning_start) & (hours < morning_end)

    
    @_min_granularity(60)
    @_import_data(['hours', 'noon_start', 'noon_end'])
    @_check_if_exists_and_save_data
    def _noons(self):
        return (hours >= noon_start) & (hours < noon_end)

    
    @_min_granularity(60)
    @_import_data(['hours', 'afternoon_start', 'afternoon_end'])
    @_check_if_exists_and_save_data
    def _afternoons(self):
        return (hours >= afternoon_start) & (hours < afternoon_end)

    
    @_min_granularity(60)
    @_import_data(['hours', 'evening_start', 'evening_end'])
    @_check_if_exists_and_save_data
    def _evenings(self):
        return (hours >= evening_start) & (hours < evening_end)
    

    @_min_granularity(60)
    @_import_data(['hours', 'night_start', 'night_end'])
    @_check_if_exists_and_save_data
    def _nights(self):
        return (hours >= night_start) & (hours < night_end)

    
    @_min_granularity(60)
    @_import_data(['hours', 'nt_start', 'nt_end'])
    @_check_if_exists_and_save_data
    def _nts(self):
        return (hours >= nt_start) & (hours < nt_end)


    @_min_granularity(60)
    @_import_data(['hours', 'ht_start', 'ht_end'])
    @_check_if_exists_and_save_data
    def _hts(self):
        return (hours >= ht_start) & (hours < ht_end)
    
        
    # feature generators
    @_min_granularity(60*24*7)
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def c_week(self):
        """ average consumption throughout the week """
        return consumption.resample('W').sum().mean()
    

    @_min_granularity(60)
    @_import_data(['consumption', 'mornings'])
    @_check_if_exists_and_save_feature
    def c_morning(self):
        """ average morning consumption """
        return consumption[mornings].mean()

    
    @_min_granularity(60)
    @_import_data(['consumption', 'noons'])
    @_check_if_exists_and_save_feature
    def c_noon(self):
        return consumption[noons].mean()
    

    @_min_granularity(60)
    @_import_data(['afternoons', 'consumption'])
    @_check_if_exists_and_save_feature
    def c_afternoon(self):
        return consumption[afternoons].mean()
    

    @_min_granularity(60)
    @_import_data(['consumption', 'evenings'])
    @_check_if_exists_and_save_feature
    def c_evening(self):
        return consumption[evenings].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'nights'])
    @_check_if_exists_and_save_feature
    def c_night(self):
        return consumption[nights].mean()

    
    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekdays'])
    @_check_if_exists_and_save_feature    
    def c_weekday(self):
        return consumption[weekdays].mean()

        
    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'mornings'])
    @_check_if_exists_and_save_feature
    def c_wd_morning(self):
        return consumption[weekdays & mornings].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'noons'])
    @_check_if_exists_and_save_feature
    def c_wd_noon(self):
        return consumption[weekdays & noons].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'afternoons'])
    @_check_if_exists_and_save_feature
    def c_wd_afternoon(self):
        return consumption[weekdays & afternoons].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'evenings'])
    @_check_if_exists_and_save_feature
    def c_wd_evening(self):
        return consumption[weekdays & evenings].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'nights'])
    @_check_if_exists_and_save_feature
    def c_wd_night(self):
        return consumption[weekdays & nights].mean()


    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekends'])
    @_check_if_exists_and_save_feature   
    def c_weekend(self):
        return consumption[weekends].mean()

        
    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'mornings'])
    @_check_if_exists_and_save_feature
    def c_we_morning(self):
        return consumption[weekends & mornings].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'noons'])
    @_check_if_exists_and_save_feature
    def c_we_noon(self):
        return consumption[weekends & noons].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'afternoons'])
    @_check_if_exists_and_save_feature
    def c_we_afternoon(self):
        return consumption[weekends & afternoons].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'evenings'])
    @_check_if_exists_and_save_feature
    def c_we_evening(self):
        return consumption[weekends & evenings].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'nights'])
    @_check_if_exists_and_save_feature
    def c_we_night(self):
        return consumption[weekends & nights].mean()


    @_min_granularity(60)
    @_import_features(['c_min', 'c_evening'])
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def c_evening_no_min(self):
        return c_evening - c_min
    

    @_min_granularity(60)
    @_import_features(['c_min', 'c_morning'])
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def c_morning_no_min(self):
        return c_morning - c_min
        
        
    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_min', 'c_night'])
    @_check_if_exists_and_save_feature
    def c_night_no_min(self):
         return c_night - c_min


    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def c_max(self):
        return consumption.max()

        
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def c_min(self):
        return consumption.min()

    
    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_min', 'c_noon'])
    @_check_if_exists_and_save_feature
    def c_noon_no_min(self):
        return c_noon - c_min


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_min', 'c_afternoon'])
    @_check_if_exists_and_save_feature
    def c_afternoon_no_min(self):
        return c_afternoon - c_min


    @_min_granularity(60)
    @_import_data(['consumption', 'hts'])
    @_check_if_exists_and_save_feature
    def c_ht(self):
        return consumption[hts].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'nts'])
    @_check_if_exists_and_save_feature
    def c_nt(self):
        return consumption[nts].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'hts'])
    @_check_if_exists_and_save_feature
    def c_we_ht(self):
        return consumption[weekends & hts].mean()

  
    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'nts'])
    @_check_if_exists_and_save_feature
    def c_we_nt(self):
        return consumption[weekends & nts].mean()
    

    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'hts'])
    @_check_if_exists_and_save_feature
    def c_wd_ht(self):
        return consumption[weekdays & hts].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'nts'])
    @_check_if_exists_and_save_feature
    def c_wd_nt(self):
        return consumption[weekdays & nts].mean()
    

    @_min_granularity(60*24*7)
    @_import_data(['consumption'])
    @_import_features(['c_week', 'c_max'])
    @_check_if_exists_and_save_feature
    def r_mean_max(self):
        return c_week/c_max


    @_min_granularity(60*24*7)
    @_import_data(['consumption'])
    @_import_features(['c_week', 'c_min'])
    @_check_if_exists_and_save_feature
    def r_min_mean(self):
        return c_min/c_week
    

    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_night', 'c_week'])
    @_check_if_exists_and_save_feature
    def r_night_day(self):
        return c_night/c_week


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_morning', 'c_noon'])
    @_check_if_exists_and_save_feature
    def r_morning_noon(self):
        return c_morning/c_noon


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_evening', 'c_noon'])
    @_check_if_exists_and_save_feature
    def r_evening_noon(self):
        return c_evening/c_noon


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_evening', 'c_noon'])
    @_check_if_exists_and_save_feature
    def r_evening_noon(self):
        return c_evening/c_noon


    @_min_granularity(7*24*60)
    @_import_data(['consumption'])
    @_import_features(['c_week', 'c_max', 'c_min'])
    @_check_if_exists_and_save_feature
    def r_mean_max_no_min(self):
        """ r_mean_max (minimum is deducted in each case) """
        return (c_week - c_min)/(c_max - c_min)


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_evening', 'c_noon', 'c_min'])
    @_check_if_exists_and_save_feature
    def r_evening_noon_no_min(self):
        """ r_evening_noon (minimum is deducted) """
        return (c_evening - c_min)/(c_noon - c_min)


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_morning', 'c_noon', 'c_min'])
    @_check_if_exists_and_save_feature
    def r_morning_noon_no_min(self):
        """ r_morning_noon (minimum is deducted) """
        return (c_morning - c_min)/(c_noon - c_min)


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_night', 'c_week', 'c_min'])
    @_check_if_exists_and_save_feature
    def r_day_night_no_min(self):
        """ r_night_day (minimum is deducted) """
        return (c_night - c_min)/(c_week - c_min)
    

    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekdays'])
    @_check_if_exists_and_save_feature
    def wd_var(self): 
        """ weekday variance """
        return consumption[weekdays].var()


    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekends'])
    @_check_if_exists_and_save_feature
    def we_var(self): 
        """ weekend variance """
        return consumption[weekends].var()

        
    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_import_features(['we_var', 'wd_var'])
    @_check_if_exists_and_save_feature
    def r_var_wd_we(self): 
        """ ratio of variance weekday/weekend """
        return wd_var/we_var

        
    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekdays'])
    @_check_if_exists_and_save_feature
    def wd_min(self): 
        """ minimum of weekdays consumption """
        return consumption[weekdays].min()

        
    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekdays'])
    @_check_if_exists_and_save_feature
    def wd_max(self): 
        """ maximum of weekdays consumption """
        return consumption[weekdays].max()

  
    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekends'])
    @_check_if_exists_and_save_feature
    def we_min(self): 
        """ minimum of weekends consumption """
        return consumption[weekends].min()


    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekends'])
    @_check_if_exists_and_save_feature
    def we_max(self): 
        """ maximum of weekends consumption """
        return consumption[weekends].max()

    
    @_min_granularity(60*24)
    @_import_data(['consumption'])
    @_import_features(['we_min', 'wd_min'])
    @_check_if_exists_and_save_feature
    def r_min_wd_we(self): 
        """ ratio of the minimum weekday/weekend day """
        if we_min == 0:
            raise Exception('r_min_wd_we undefined, we_min equals 0.')
        return wd_min/we_min


    @_min_granularity(60*24)
    @_import_data(['consumption'])
    @_import_features(['we_max', 'wd_max'])
    @_check_if_exists_and_save_feature
    def r_max_wd_we(self):
        """ ratio of the maximum weekday/weekend day """
        return wd_max/we_max


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_evening', 'c_we_evening'])
    @_check_if_exists_and_save_feature
    def r_evening_wd_we(self):
        """ ratio of consumption during evening, weekday/weekend day """
        return c_wd_evening/c_we_evening


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_night', 'c_we_night'])
    @_check_if_exists_and_save_feature
    def r_night_wd_we(self): 
        """ ratio of consumption at night, weekday/weekend """
        return c_wd_night/c_we_night


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_noon', 'c_we_noon'])
    @_check_if_exists_and_save_feature
    def r_noon_wd_we(self):
        """ ratio of consumption during lunchtime, weekday/weekend """
        return c_wd_noon/c_we_noon
    
    
    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_morning', 'c_we_morning'])
    @_check_if_exists_and_save_feature
    def r_morning_wd_we(self):
        """ ratio of consumption in the morning, weekday/weekend day """
        return c_wd_morning/c_we_morning

    
    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_afternoon', 'c_we_afternoon'])
    @_check_if_exists_and_save_feature
    def r_afternoon_wd_we(self):
        """ ratio of consumption in the afternoon, weekday/weekend day """
        return c_wd_afternoon/c_we_afternoon


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_we_night', 'c_weekend'])
    @_check_if_exists_and_save_feature
    def r_we_night_day(self):
        """ ratio c_we_night/c_weekend """
        return c_we_night/c_weekend

    
    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_we_morning', 'c_we_noon'])
    @_check_if_exists_and_save_feature
    def r_we_morning_noon(self): 
        """ ratio c_we_morning/c_we_noon """
        return c_we_morning/c_we_noon


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_we_noon', 'c_we_evening'])
    @_check_if_exists_and_save_feature
    def r_we_evening_noon(self): 
        """ ratio c_we_evening/c_we_noon """
        return c_we_evening/c_we_noon


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_night', 'c_weekday'])
    @_check_if_exists_and_save_feature
    def r_wd_night_day(self):
        """ ratio c_wd_night/c_weekday """
        return c_wd_night/c_weekday


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_morning', 'c_wd_noon'])
    @_check_if_exists_and_save_feature
    def r_wd_morning_noon(self): 
        """ ratio c_wd_morning/c_wd_noon """
        return c_wd_morning/c_wd_noon


    @_min_granularity(60)
    @_import_features(['c_wd_evening', 'c_wd_noon'])
    @_check_if_exists_and_save_feature
    def r_wd_evening_noon(self): 
        """ ratio c_wd_evening/c_wd_noon """
        return c_wd_evening/c_wd_noon


    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'nts'])
    @_check_if_exists_and_save_feature
    def c_wd_nt(self):
        """ average consumption on weekdays during nts """
        return consumption[weekdays & nts].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'nts'])
    @_check_if_exists_and_save_feature
    def c_we_nt(self):
        """ average consumption on weekends during nts """
        return consumption[weekends & nts].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekdays', 'hts'])
    @_check_if_exists_and_save_feature
    def c_wd_ht(self):
        """ average consumption on weekdays during hts """
        return consumption[weekdays & hts].mean()


    @_min_granularity(60)
    @_import_data(['consumption', 'weekends', 'hts'])
    @_check_if_exists_and_save_feature
    def c_we_ht(self):
        """ average consumption on weekends during hts """
        return consumption[weekends & hts].mean()

    
    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_nt', 'c_we_nt'])
    @_check_if_exists_and_save_feature
    def r_nt_wd_we(self): 
        """ ratio of nt consumption weekday/weekend days """
        return c_wd_nt/c_we_nt

    
    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_ht', 'c_we_ht'])
    @_check_if_exists_and_save_feature
    def r_ht_wd_we(self): 
        """ ratio of ht consumption weekday/weekend days """
        return c_wd_ht/c_we_ht


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_ht', 'c_nt'])
    @_check_if_exists_and_save_feature
    def r_nt_ht(self):
        """ ratio of ht/nt consumption """
        return c_ht/c_nt

        
    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_we_ht', 'c_we_nt'])
    @_check_if_exists_and_save_feature
    def r_we_nt_ht(self):
        """ ratio of ht/nt consumption during weekend """ 
        return c_we_ht/c_we_nt


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_wd_ht', 'c_wd_nt'])
    @_check_if_exists_and_save_feature
    def r_wd_nt_ht(self):
        """ ratio of ht/nt consumption during weekend """ 
        return c_wd_ht/c_wd_nt


    @_min_granularity(60)
    @_import_data(['consumption', 'hts'])
    @_check_if_exists_and_save_feature
    def c_ht_max(self):
        """ maximum of ht consumption """
        return consumption[hts].max()


    @_min_granularity(60)
    @_import_data(['consumption', 'nts'])
    @_check_if_exists_and_save_feature
    def c_nt_max(self):
        """ maximum of nt consumption """
        return consumption[nts].max()


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_ht', 'c_ht_max'])
    @_check_if_exists_and_save_feature
    def r_ht_mean_max(self):
        """ medium ht consumption and maximum ratio """
        return c_ht/c_ht_max


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_nt', 'c_nt_max'])
    @_check_if_exists_and_save_feature
    def r_nt_mean_max(self):
        """ medium nt consumption and maximum ratio """
        return c_nt/c_nt_max


    @_min_granularity(60)
    @_import_data(['consumption', 'nts'])
    @_check_if_exists_and_save_feature
    def c_nt_min(self):
        """ minimum of nt consumption """
        return consumption[nts].min()


    @_min_granularity(60)
    @_import_data(['consumption', 'hts'])
    @_check_if_exists_and_save_feature
    def c_ht_min(self):
        """ minimum of nt consumption """
        return consumption[hts].min()


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_ht', 'c_ht_min'])
    @_check_if_exists_and_save_feature
    def r_ht_min_mean(self):
        """ medium ht consumption and minimum ratio """
        return c_ht_min/c_ht


    @_min_granularity(60)
    @_import_data(['consumption'])
    @_import_features(['c_nt', 'c_nt_min'])
    @_check_if_exists_and_save_feature
    def r_nt_min_mean(self):
        """ medium nt consumption and minimum ratio """
        return c_nt_min/c_nt

        
    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week'])
    @_check_if_exists_and_save_feature
    def s_max(self):
        """ maximum in the week """
        return average_week.max()


    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week'])
    @_check_if_exists_and_save_feature
    def s_min(self):
        """ minimum in the average week """
        return average_week.min()

        
    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week', 'samples_in_week', 'weekdays'])
    @_check_if_exists_and_save_feature
    def s_wd_min(self):
        """ minimum in the average week, limited to weekdays (Mon—Fri) """
        return average_week[weekdays[:samples_in_week]].min()


    
    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week', 'samples_in_week', 'weekdays'])
    @_check_if_exists_and_save_feature
    def s_wd_max(self):
        """ maximum in the average week, limited to weekdays (Mon—Fri) """
        return average_week[weekdays[:samples_in_week]].max()
        

    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week', 'samples_in_week', 'weekends'])
    @_check_if_exists_and_save_feature
    def s_we_min(self):
        """ minimum in the average week, limited to weekends """
        return average_week[weekends[:samples_in_week]].min()

    
    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week', 'samples_in_week', 'weekends'])
    @_check_if_exists_and_save_feature
    def s_we_max(self):
        """ maximum in the average week, limited to weekends """
        return average_week[weekends[:samples_in_week]].max()
        

    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def s_sm_variety(self):
        """ 20%-quintile of the deviation from the previous measured value """
        return consumption.diff().abs().quantile(0.2)
        

    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def s_bg_variety(self):
        """ 60%-quintile of the deviation from the previous measured value """
        return consumption.diff().abs().quantile(0.6)

    
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def s_variance(self):
        """ consumption variance """
        return consumption.var()


    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekdays'])
    @_check_if_exists_and_save_feature
    def s_var_wd(self):
        """ variance on weekdays """
        return consumption[weekdays].var()


    @_min_granularity(24*60)
    @_import_data(['consumption', 'weekends'])
    @_check_if_exists_and_save_feature
    def s_var_we(self):
        """ variance on weekends """
        return consumption[weekends].var()


    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def s_diff(self):
        """ total of differences from predecessor (absolute value) """
        return consumption.diff().abs().sum()
        

    @_import_data(['consumption', 'neighborhood_width'])
    @_check_if_exists_and_save_feature
    def s_num_peaks(self):
        """ number of peak (local maximum when considering width_neighborhood measured values """
        peaks = argrelextrema(consumption.values, np.greater_equal, order=neighborhood_width)[0]
        return len(peaks)


    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def s_q1(self):
        """ lower quartile of consumption """
        return consumption.quantile(0.25)

        
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def s_q2(self):
        """ second quartile (median) """
        return consumption.median()
        
        
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def s_q3(self):
        """ upper quartile """
        return consumption.quantile(0.75)
        

    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def c_max_avg(self):
        """ average daily maximum """
        return consumption.resample('D').max().mean()


    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def c_min_avg(self):
        """ average daily minimum """
        return consumption.resample('D').min().mean()


    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def s_number_zeros(self):
        """ number of zero values """
        zeros = consumption[consumption == 0]
        return len(zeros)


    @_import_data(['consumption', 'neighborhood_width'])
    @_check_if_exists_and_save_feature
    def c_sm_max(self):
        """ maximum with simple smoothing """
        consumption_smooth = uniform_filter1d(consumption, neighborhood_width)
        return consumption_smooth.max()

    
    @_min_granularity(60)
    @_import_data(['consumption', 'nts'])
    @_check_if_exists_and_save_feature
    def s_nt_variance(self):
        """ variance of nt consumption """
        return consumption[nts].var()


    @_min_granularity(60)
    @_import_data(['consumption', 'hts'])
    @_check_if_exists_and_save_feature
    def s_ht_variance(self):
        """ variance of ht consumption """
        return consumption[hts].var()


    @_min_granularity(60)
    @_import_data(['consumption', 'nts', 'weekdays'])
    @_check_if_exists_and_save_feature
    def s_nt_var_wd(self):
        """ variance of nt consumption on weekdays """
        return consumption[nts & weekdays].var()


    @_min_granularity(60)
    @_import_data(['consumption', 'hts', 'weekdays'])
    @_check_if_exists_and_save_feature
    def s_ht_var_wd(self):
        """ variance of ht consumption on weekdays """
        return consumption[hts & weekdays].var()


    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week'])
    @_check_if_exists_and_save_feature
    def t_above_mean(self):
        """ number of data points above mean of the week (for the entire week) """
        return len(average_week[average_week > average_week.mean()])


    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week', 'samples_in_day'])
    @_check_if_exists_and_save_feature
    def t_daily_max(self):
        """ time of the first day’s maximum reached (averaged over all weekdays) """
        return np.argmax(average_week[:samples_in_day])
        

    @_min_granularity(24*60)
    @_import_data(['consumption', 'average_week', 'samples_in_day'])
    @_check_if_exists_and_save_feature
    def t_daily_min(self):
        """ time of the first day’s minimum reached (averaged over all weekdays) """
        return np.argmin(average_week[:samples_in_day])


    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def t_width_peaks(self):
        """ average extent of the peak """
        peaks, _ = find_peaks(consumption)
        p_widths, x, _, _ = peak_widths(consumption, peaks)
        return p_widths.sum()/len(p_widths)

    
    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_check_if_exists_and_save_feature
    def c_base_guess(self):
        """ estimated base load """
        return consumption.resample('D').min().median()

        
    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_import_features(['c_base_guess'])
    @_check_if_exists_and_save_feature
    def t_const_time(self):
        """ estimated time of base load """
        return len(consumption[consumption <= c_base_guess])

    
    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_import_features(['c_base_guess'])
    @_check_if_exists_and_save_feature
    def t_first_above_base(self):
        """ first crossing of a threshold assumed as a base load """
        return np.where(consumption.values > c_base_guess)[0][0]

    
    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_import_features(['c_base_guess'])
    @_check_if_exists_and_save_feature
    def t_above_base(self):
        """ number of measuring points above the base load limit """
        base = self.features['c_base_guess']
        return len(consumption[consumption > base])


    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_import_features(['t_above_base'])
    @_check_if_exists_and_save_feature
    def t_percent_above_base(self):
        """ proportion of the measuring points above the base load limit """
        return t_above_base/len(consumption)


    @_min_granularity(24*60)
    @_import_data(['consumption'])
    @_import_features(['c_base_guess'])
    @_check_if_exists_and_save_feature
    def t_value_above_base(self):
        """ sum of the measuring points above the base load limit """
        return consumption[consumption > c_base_guess].sum()


    @_min_granularity(60)
    @_import_data(['consumption', 'nights', 'temperature'])
    @_check_if_exists_and_save_feature
    def w_temp_cor_nighttime(self):
        """ linear relationship between temperature and consumption in the night """
        x = temperature[nights]
        y = consumption[nights]
        x, y = dropna_and_index_intersect(x, y)
        (k, _), _ = curve_fit(line, x, y)
        return  k


    @_min_granularity(60)
    @_import_data(['consumption', 'nights', 'temperature'])
    @_check_if_exists_and_save_feature
    def w_temp_cor_daytime(self):
        """ lin relationship between temperature and consumption during day during weekdays  """
        x = temperature[~nights]
        y = consumption[~nights]
        x, y = dropna_and_index_intersect(x, y)
        (k, _), _ = curve_fit(line, x, y)
        return k


    @_min_granularity(60)
    @_import_data(['consumption', 'evenings', 'temperature'])
    @_check_if_exists_and_save_feature
    def w_temp_cor_evening(self):
        """ linear relationship between temperature and consumption in the evening """
        x = temperature[evenings]
        y = consumption[evenings]
        x, y = dropna_and_index_intersect(x, y)
        (k, _), _ = curve_fit(line, x, y)
        return k


    @_min_granularity(24*60)
    @_import_data(['consumption', 'temperature'])
    @_check_if_exists_and_save_feature
    def w_temp_cor_minima(self):
        """ linear relationship between the daily minima of temperature and power consumption """
        x = temperature.resample('D').min()
        y = consumption.resample('D').min()
        x, y = dropna_and_index_intersect(x, y)
        (k, _), _ = curve_fit(line, x, y)
        return k


    @_min_granularity(24*60)
    @_import_data(['consumption', 'temperature'])
    @_check_if_exists_and_save_feature
    def w_temp_cor_maxmin(self):
        """ lin relationship between the daily maxima of consumption and minima of temperature """
        x = temperature.resample('D').min()
        y = consumption.resample('D').max()
        x, y = dropna_and_index_intersect(x, y)
        (k, _), _ = curve_fit(line, x, y)
        return k
    
      
    @_import_data(['consumption', 'temperature'])
    @_check_if_exists_and_save_feature
    def k1(self):
        #not ideaomatic so we dont have to run the same curve_fit four times
        x, y = dropna_and_index_intersect(temperature, consumption)
        (k1, n1, k2, n2), _ = curve_fit(hockeyStick, x, y)
        lowpoint = (n2-n1)/(k1-k2)
        consumptionAtLowpoint = hockeyStick([lowpoint], k1, n1, k2, n2)[0]
        self.features['n1'] = n1
        self.features['k2'] = k2
        self.features['n2'] = n2
        self.features['lowpoint'] = lowpoint
        self.features['consumptionAtLowpoint'] = consumptionAtLowpoint
        return k1

        
    @_import_features(['k1', 'n1'])
    @_check_if_exists_and_save_feature
    def n1(self):
        # during k1 import we generate also n1, so its available, but could also cause
        # nasty recursive bugs
        return n1

 
    @_import_features(['k1', 'k2'])
    @_check_if_exists_and_save_feature
    def k2(self):
        # during k1 import we generate also k2, so its available, but could also cause
        # nasty recursive bugs
        return k2

    
    @_import_features(['k1', 'n2'])
    @_check_if_exists_and_save_feature
    def n2(self):
        # during k1 import we generate also n2, so its available, but could also cause
        # nasty recursive bugs
        return n2

    
    @_import_features(['k1', 'lowpoint'])
    @_check_if_exists_and_save_feature
    def lowpoint(self):
        # during k1 import we generate and save also lowpoint, so its available,
        # but could also cause nasty recursive bugs
        return lowpoint
    

    @_import_features(['k1'])
    @_check_if_exists_and_save_feature
    def consumptionAtLowpoint(self):
        # during k1 import we generate also n2, so its available, but could also cause
        # nasty recursive bugs
        return consumptionAtLowpoint

    
    @_import_features(['k1', 'n1', 'k2', 'n2'])
    @_import_data(['consumption', 'temperature'])
    @_check_if_exists_and_save_feature
    def hockeyStickErrRel(self):
        x, y = dropna_and_index_intersect(temperature, consumption)
        x = x.values
        y = y.values
        yPred = hockeyStick(x, k1, n1, k2, n2)
        return np.sum(abs((yPred - y)/yPred))/len(y)


    @_import_data(['consumption'])
    @_import_features(['consumptionAtLowpoint'])
    @_check_if_exists_and_save_feature
    def hockeyStickThermalEfficiency(self):
        c = consumption.dropna().values
        return (consumptionAtLowpoint*len(c))/np.sum(c)


    @_import_data(['consumption', 'temperature'])
    @_check_if_exists_and_save_feature
    def k(self):
        # non idiomatic so we dont call curve_fit 2 times on same data
        x, y = dropna_and_index_intersect(temperature, consumption)
        x = x.values
        y = y.values
        (k, n), _ = curve_fit(line, x, y)
        self.features['n'] = n
        return k


    @_import_features(['k', 'n'])
    @_check_if_exists_and_save_feature
    def n(self):
        # n is generated when importing 'k', could also lead to nasty bugs
        return n
        

    @_import_features(['k', 'n'])
    @_import_data(['consumption', 'temperature'])
    @_check_if_exists_and_save_feature
    def linearErrRel(self):
        x, y = dropna_and_index_intersect(temperature, consumption)
        x = x.values
        y = y.values
        yPred = line(x, k, n)
        return np.sum(abs((yPred - y)/yPred))/len(y)

        
    @_import_features(['linearErrRel', 'hockeyStickErrRel'])
    @_check_if_exists_and_save_feature
    def hockeyStickDependency(self):
        return 1 - hockeyStickErrRel/linearErrRel


    @_min_granularity(60)
    @_import_data(['consumption', 'temperature', 'samples_in_day'])
    @_check_if_exists_and_save_feature
    def consumption_temperature_lag(self):
        t, c = dropna_and_index_intersect(temperature, consumption)
        lags = []
        n_days = int(len(c)/samples_in_day)
        for n in range(n_days):
            consumption_daily = t[n*samples_in_day:(n+1)*samples_in_day]
            temperature_daily = c[n*samples_in_day:(n+1)*samples_in_day]
            corr = correlate(consumption_daily, temperature_daily)
            i = np.where(corr == np.max(corr))[0][0]
            lag = min(i%samples_in_day, (-i)%samples_in_day)
            lags.append(lag)
        lag = sum(lags)/len(lags)
        return lag


    
consumption = pd.read_csv('data/consumption.csv', parse_dates=True, index_col=0).squeeze()
temperature = pd.read_csv('data/temperature.csv', parse_dates=True, index_col=0).squeeze()
meta = json.load(open('data/meta.json'))


# EXAMPLE 1: ALL AVAIALBLE
# data = {'consumption': consumption,
#         'temperature': temperature}
# data.update(meta)
# extractor = Extractor(data)
# extractor._extract_available()


# EXAMPLE 2: SPECIFIC FEATURES EXTRACTION
data = {'consumption': consumption,
        'temperature': temperature}
data.update(meta)
extractor = Extractor(data)
extractor._extract(['blabla', 'c_afternoon', 's_num_peaks'])


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




df = pd.DataFrame([extractor.extracted])
print(df)
print(extractor.unavailable)
