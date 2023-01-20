import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def prep_meter_and_space(d):
    """Prepares the meter and space features by interpolating missing values and log transforming the meter_reading.
    Args: 
        df (pandas.DataFrame): The dataset to prepare the meter and space features.
    Returns: 
        pandas.DataFrame: The dataset with the prepared meter and space features.
    """
    dataset = d
    dataset.loc[:,'meter_reading'] = dataset.loc[:,'meter_reading'].astype('float')
    dataset.loc[:,'meter_reading'] = dataset.loc[:,'meter_reading'].interpolate(method='linear', limit_direction='both', axis=0)
    # Transform meter_reading to log scale to reduce the impact of outliers; the reverse transformation
    #  needs to be applied when predicting the meter_reading by np.expm1
    dataset.loc[:,'meter_reading'] = np.float64(np.log1p(dataset.loc[:,'meter_reading']))
    dataset.loc[:,'air_temperature'] = dataset.loc[:,'air_temperature'].interpolate(method='linear', limit_direction='both', axis=0)
    #normalizing temperature data using min-max normalization
    dataset.loc[:,'air_temperature'] = (dataset.loc[:,'air_temperature'] - dataset.loc[:,'air_temperature'].min()) / (dataset.loc[:,'air_temperature'].max() - dataset.loc[:,'air_temperature'].min())
    return dataset

#Zeit-Features transformieren (zyklisches Feature)
#data transformation function for time features
def fill_site_id_with_mode_per_group(d):
    """Fills the site_id with the mode per building_id.
    Args:
        d (pandas.DataFrame): The dataset to fill the site_id.
    Returns:
        pandas.DataFrame: The dataset with the filled site_id."""
    list_of_columns_to_be_filled = ['site_id', 'primary_use_Education', 'primary_use_Entertainment/public assembly',
       'primary_use_Food sales and service', 'primary_use_Healthcare',
       'primary_use_Lodging/residential',
       'primary_use_Manufacturing/industrial', 'primary_use_Office',
       'primary_use_Other', 'primary_use_Parking',
       'primary_use_Public services', 'primary_use_Religious worship',
       'primary_use_Retail', 'primary_use_Services',
       'primary_use_Technology/science', 'primary_use_Utility',
       'primary_use_Warehouse/storage']
    for column in list_of_columns_to_be_filled:
        d[column] = d.groupby('building_id')[column].transform(lambda x: x.fillna(x.mode()[0]))
    return d
    #d['site_id'] = d.groupby('building_id')['site_id'].transform(lambda x: x.fillna(x.mode()[0]))
    #d.loc[:,'site_id'] = d.loc[:,'site_id'].astype('int')
    d = d.assign(site_id=d.groupby('building_id')['site_id'].transform(lambda x: x.fillna(x.mode()[0])))
    d = d.assign(site_id = d.site_id.astype('int32'))
    d = d.assign(primary_use=d.groupby('building_id')['primary_use'].transform(lambda x: x.fillna(x.mode()[0])))
    
    return d

def transform_to_cyclical(train_data):
    """ Transforms the time features to cyclical features.
    Args:
        train_data (pandas.DataFrame): The dataset to transform the time features.
    Returns:
        pandas.DataFrame: The dataset with the transformed time features.
    """

    features_cyc = {'month' : 12, 'weekday' : 7, 'hour' : 24}
    for feature in features_cyc.keys():
        train_data[feature+'_sin'] = np.sin((2*np.pi*train_data[feature])/features_cyc[feature])
        train_data[feature+'_cos'] = np.cos((2*np.pi*train_data[feature])/features_cyc[feature])
    train_data = train_data.drop(features_cyc.keys(), axis=1)
    train_data = train_data.drop(['date'], axis=1)
    return train_data

def transform_time_features(train_data, boolean=True):
    """ Transforms the time features by first extracting day, month, weekday, weekend and hour from the timestamp 
        and then converting them to cyclic features.
    Args: 
        train_data (pandas.DataFrame): The dataset to transform the time features.
        boolean (bool): If True, the time features are transformed to cyclical, if False, the time features are not transformed.
    Returns:
        pandas.DataFrame: The dataset with the transformed time features.

    
    """
    train_data = train_data.assign(date = pd.to_datetime(train_data.loc[:,'timestamp']))
    train_data = train_data.assign(month = pd.to_datetime(train_data.loc[:,'timestamp']).dt.month)
    train_data = train_data.assign(weekday = pd.to_datetime(train_data.loc[:,'timestamp']).dt.weekday)
    train_data = train_data.assign(weekend = np.where((train_data['weekday'] == 5) | (train_data['weekday'] == 6), 1, 0))
    train_data = train_data.assign(hour = pd.to_datetime(train_data.loc[:,'timestamp']).dt.hour)
    

    if boolean == True:
        train_data = transform_to_cyclical(train_data)
    else:
        pass
    return train_data

def add_statistical_features(data: pd.DataFrame) -> pd.DataFrame:
    """ Adds statistical features to the dataset. Important: Only apply to a single building.
    Args:
        data (pandas.DataFrame): The dataset of a single building to add the statistical features.
    Returns:
        pandas.DataFrame: The dataset with the added statistical features.
    """
    series = data.set_index('timestamp')['meter_reading']
    decomposed = seasonal_decompose(series, model='additive', extrapolate_trend='freq', period=24)
    data = data.assign(trend = decomposed._trend.values) #for some reason the trend is all nan
    data = data.assign(seasonal = decomposed._seasonal.values)
    data = data.assign(residual = decomposed._resid.values)
    #data['trend'] = decomposed._trend.values
    #data['seasonal'] = decomposed._seasonal.values
    #data['residual'] = decomposed._resid.values

    return data

def make_last_column_the_first(d):
    cols = d.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    d = d[cols]
    return d

def make_dummy_variables(d):
    """ Makes dummy variables for the categorical features.
    Args:
        d (pandas.DataFrame): The dataset to make the dummy variables.
    Returns:
        pandas.DataFrame: The dataset with the dummy variables.
    """
    d = pd.get_dummies(d, columns=['primary_use'])
    return d


def apply_all_funcs(d):
    """ Applies all functions to the dataset.
    Args:
        d (pandas.DataFrame): The dataset to apply the functions.
    Returns:
        pandas.DataFrame: The dataset with the applied functions.
    """
    d = fill_site_id_with_mode_per_group(d)
    d = transform_time_features(d)
    d = prep_meter_and_space(d)
    if len(d) <= 8784:
        d = add_statistical_features(d)
    else: 
        pass
    #d = make_last_column_the_first(d) # only needed when pivoting the table and using a different format of the dataset.
    return d

def apply_to_list_of_dfs(list_of_dfs):
    """ Applies all functions to a list of datasets, in case the datasets are already separated by building_id or site_id.
    Args:
        list_of_dfs (list): The list of datasets to apply the functions.
    Returns:
        list: The list of datasets with the applied functions.
    """
    for i in range(len(list_of_dfs)):
        list_of_dfs[i] = apply_all_funcs(list_of_dfs[i])
    print('All feature_engineering functions applied to the list of datasets.')
    return list_of_dfs

