import pandas as pd
import numpy as np

def load_and_merge_data():
    """Load the data from the csv file.
    Args:
        None
    Returns:
        pandas.DataFrame: The dataset with all data.
    """
    base_string = r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\kaggle_data' 
    train_data = pd.read_csv(base_string + r'\train.csv', delimiter=',', parse_dates=['timestamp'])
    train_weather = pd.read_csv(base_string + r'\weather_train.csv', delimiter=',', parse_dates=['timestamp'])
    building_metadata = pd.read_csv(base_string + r'\building_metadata.csv', delimiter=',')
    #select only oberservations of electricity meter
    train_data_electricity = train_data.loc[train_data['meter'] == 0]

    #merge training_data with building_metadata
    dataset = pd.merge(train_data_electricity, building_metadata, how='left', on='building_id')
    dataset = pd.merge(dataset, train_weather, how='left', on=['site_id', 'timestamp'])
    print('The shape of the dataset is: ', dataset.shape)

    return dataset

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the columns that are not needed for the model.
    Args:
        df (pandas.DataFrame): The dataset to drop the columns from.
    Returns:
        pandas.DataFrame: The dataset with the dropped columns.
    """
    df = df.drop(columns=['square_feet','year_built', 'floor_count', 'meter','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed'], axis=1)
    print('The remaining columns are: ', df.columns.values)
    print('The shape of the dataset is: ', df.shape)
    return df

def generate_non_participating_set(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a subset of the dataset that will not participate in the federated learning.
    Args:
        df (pandas.DataFrame): The dataset to generate the subset from.
    Returns:
        pandas.DataFrame: The dataset with the excluded test data.
    """

    criteria = 'timestamp>20160101000000'
    building_ids_of_subset = df.groupby('building_id').min().query(criteria).index.tolist()
    subset = df[df.building_id.isin(building_ids_of_subset)]
    excluded_ids = df.groupby('building_id').max().query('timestamp<20161231230000').index.tolist()
    df = df[~df.building_id.isin(building_ids_of_subset)]
    df = df[~df.building_id.isin(excluded_ids)]
    print('The number of buildings in the subset is: ', len(subset.building_id.unique()), 
    'The number of remaining buildings is: ', len(df.building_id.unique()))
    return df, subset

def check_for_missing_data(df: pd.DataFrame, threshold) -> pd.DataFrame:
    """Check for missing data in the dataset.
    Args:
        df (pandas.DataFrame): The dataset to check for missing data.
        threshold (float): The threshold to use to determine if a column has too many missing values.
    Returns:
        pandas.DataFrame: A dataframe with the missing data.
        pandas.DataFrame: A dataframe with the excluded test data.
    """
    threshold = threshold * df.timestamp.nunique()
    below_treshold = df.groupby('building_id').count().query(f'meter_reading<{threshold}').index.tolist()
    df = df[~df['building_id'].isin(below_treshold)]

    return df

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove buildings that have meter_readings with a z-value above |20| from the dataset.
    Args:
        df (pandas.DataFrame): The dataset to remove the outliers from.
    Returns:
        pandas.DataFrame: The dataset with the removed outliers.
    """
    df['z_score'] = df['z_score'] = (df['meter_reading'] - df.groupby("building_id")["meter_reading"].transform('mean')) / df.groupby("building_id")["meter_reading"].transform('std')
    outliers = df[(df['z_score'] > 20) | (df['z_score'] < -20)]
    building_ids = list(outliers['building_id'].unique())
    df = df[~df['building_id'].isin(building_ids)]
    print('The number of buildings with outliers is: ', len(building_ids))
    print('The excluded building_ids are: ', sorted(building_ids))
    print('The affected site_ids are: ', sorted(list(outliers['site_id'].unique())))
    print('The maximum meter_reading is:' , df['meter_reading'].max())
    df.drop(['z_score'], axis=1, inplace=True)
    grouped = df.groupby('building_id')
    def check_zero_values(group):
        return (group['meter_reading'] == 0).mean() > 0.5
    # Apply the custom function on each group
    zero_value_buildings = grouped.apply(check_zero_values)
    # Select the building_ids that have more than 50% zero values
    result = zero_value_buildings[zero_value_buildings == True].index.tolist()
    print('The number of buildings with more than 50% zero values is: ', len(result))
    df = df[~df['building_id'].isin(result)]
    return df

def resample_data(df: pd.DataFrame) -> pd.DataFrame:
    """Resample the data to hourly data. For that the data is grouped by building_id so each building has its own time series.
    Args:
        df (pandas.DataFrame): The dataset to resample.
    Returns:
        pandas.DataFrame: The dataset with the resampled data.
    """
    df = (df.set_index('timestamp')
       .groupby('building_id')
       .resample('H')
       .asfreq()
       .drop('building_id', axis=1)
       .reset_index())
    print('Resampling completed. The shape of the dataset is: ', df.shape)
    return df

def get_all_building_ids(df: pd.DataFrame) -> list:
    """Get all building ids from the dataset.
    Args:
        df (pandas.DataFrame): The dataset to get the building ids from.
    Returns:
        list: A list with all building ids.
    """
    building_ids = df.building_id.unique()
    return building_ids

"""if __name__ == '__main__':
    df = load_and_merge_data()
    df = drop_columns(df)
    df, subset = generate_non_participating_set(df)
    df = check_for_missing_data(df, 0.7)
    df = resample_data(df)
    building_ids = get_all_building_ids(df)
    subset_ids = get_all_building_ids(subset)
    np.save('building_ids.npy', building_ids)
    np.save('subset_ids.npy', subset_ids)"""

if __name__ == '__main__':
    df = load_and_merge_data()
    df = drop_columns(df)
    print(df.shape)
    df, subset = generate_non_participating_set(df)
    print(df.shape, subset.shape)
    df = check_for_missing_data(df, 0.7)
    print(df.shape)
    df = resample_data(df)
    print(df.columns)