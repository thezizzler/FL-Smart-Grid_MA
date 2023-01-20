import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

ROOT = r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_datasets_dummy'

def split_in_buildings(df):
    """Splits the dataset into a list of dataframes, where each dataframe contains the data of one building.
    Args:
        df (pandas.DataFrame): The dataset to split.
    Returns:
        list: A list of dataframes, where each dataframe contains the data of one building.
    """
    buildings = df['building_id'].unique()
    buildings.sort()
    def split_by_building(building_id):
        return df[df['building_id'] == building_id]
    buildings = list(map(split_by_building, buildings))
    #buildings = buildings.tolist()
    #buildings = [df[df['building_id'] == i] for i in buildings]
    return buildings

def reverse_split_in_buildings(buildings):
    """Reverses the splitting of a dataset into a list of dataframes, where each dataframe contains the data of one building.
    Args:
        buildings (list): A list of dataframes, where each dataframe contains the data of one building.
    Returns:
        pandas.DataFrame: The dataset.
    """
    df = pd.concat(buildings)
    return df



def split_in_sites(df):
    """Splits the dataset into a list of dataframes, where each dataframe contains the data of one site.
    Args:
        df (pandas.DataFrame): The dataset to split.
    Returns:
        list: A list of dataframes, where each dataframe contains the data of one site.
    """

    sites = df['site_id'].unique()
    sites.sort()
    sites = sites.tolist()
    sites = [df[df['site_id'] == i] for i in sites]
    return sites

def export_buildings_per_site(buildings, root_path=ROOT, format='csv'):
    """ Writes a list of dataframes to a folder structure, 
        where each site has its own folder and each building has its own csv/pkl file.
    
    Args:
        buildings (list): A list of dataframes, where each dataframe contains the data of one building.
        path (str): The path to the folder where the data should be written to.
        format (str): The format of the files to write. Either 'csv' or 'pkl'.
        
    Returns: 
        None
    """
    def create_folder(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    def get_site_id(df):
        return str(int(df['site_id'].iloc[0]))
    def get_building_id(df):
        return str(int(df['building_id'].iloc[0]))
    def get_path_string(df):
        return root_path + r'\site_' + get_site_id(df) + r'\building_' + get_building_id(df)

    for df in buildings:
        # Extract the site_id for this DataFrame
        site_id = get_site_id(df)
        create_folder(root_path + r'\site_' + site_id)
        filename = get_path_string(df)
        df = df.drop(columns=['timestamp','site_id','building_id'], axis=1)
        if format == 'csv':
            df.to_csv(filename + '.csv', index=False)
        elif format == 'pkl':
            df.to_pickle(filename + '.pkl')
        else:
            raise ValueError('The format must be either "csv" or "pkl".')
    return print('The data has been written to the folder structure.')


# not yet supported; the data has to be pivoted to a wide format first
def export_sites(buildings, root_path=ROOT, format='csv'):
    """Writes a list of dataframes to a csv/pkl file, 
        where each site has its own csv/pkl file.
    Args: 
        sites (list): A list of dataframes, where each dataframe contains the data of one site.
        path (str): The path to the folder where the data should be written to.
        format (str): The format of the files to write. Either 'csv' or 'pkl'.
    Returns:
        None
    """
    df = pd.concat(buildings)
    def create_folder(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    # Extract the building_id for this DataFrame
    def get_site_id(df):
        return str(int(df['site_id'].iloc[0]))
    def get_path_string(df):
        return root_path + r'\cross-silo\site_' + get_site_id(df)


        # drop site_id column
    df_grouped = df.groupby('site_id')

    for site_id, df in df_grouped:
        create_folder(root_path + r'\cross-silo' + r'\site_' + str(site_id))

        filename = get_path_string(df)
        df = df.drop(columns=['site_id', 'timestamp', 'building_id'], axis=1)
        df_train, df_val_test = train_test_split(df, test_size=0.2, shuffle=False)
        df_val, df_test = train_test_split(df_val_test, test_size=0.5, shuffle=False)
        if format == 'csv':
            df_train.to_csv(filename + r'\train.csv', index=False)
            df_val.to_csv(filename + r'\val.csv', index=False)
            df_test.to_csv(filename + r'\test.csv', index=False)
        elif format == 'pkl':
            df_train.to_pickle(filename + r'\train.pkl')
            df_val.to_pickle(filename + r'\val.pkl')
            df_test.to_pickle(filename + r'\test.pkl')
        else:
            raise ValueError('The format must be either "csv" or "pkl".')
    return print('Saving the files is done.')

def export_test_set(testsets, path=ROOT, format='csv'):
    """ Writes a list of dataframes to a csv/pkl file, 
        where each dataframe has its own csv/pkl file.
    Args:
        testsets (list): A list of dataframes, where each dataframe contains the data of one building.
        path (str): The path to the folder where the data should be written to.
        format (str): The format of the files to write. Either 'csv' or 'pkl'.
    Returns:
        None
    """
    #helper function to create folder structure
    def get_building_id(df):
        return str(int(df['building_id'].iloc[0]))

    for df in testsets:
        
        building_id = get_building_id(df)
        # drop building_id and site_id column
        df = df.drop(columns=['timestamp', 'site_id','building_id'], axis=1)
        #create_folder(path + r'site_id_' + building_id)
        # Write the DataFrame to a CSV file in the site's directory
        if format == 'csv':
            df.to_csv(f'{path}/test_{building_id}.csv', index=False)
        elif format == 'pkl':
            df.to_pickle(f'{path}/test_{building_id}.pkl')
        else: print('Please choose a valid format.')


def export_complete_dataset(df, path=ROOT, format='csv'):
    """ Writes a dataframe to a csv/pkl file.
    Args:
        df (pandas.DataFrame): The dataset to write.
        path (str): The path to the folder where the data should be written to.
        format (str): The format of the files to write. Either 'csv' or 'pkl'.
    Returns:
        None
    """
    # drop building_id and site_id column
    #df = df.drop(columns=['site_id','building_id'], axis=1)
    # Write the DataFrame to a CSV file in the site's directory
    if format == 'csv':
        df.to_csv(f'{path}/complete_dataset.csv', index=False)
    elif format == 'pkl':
        df.to_pickle(f'{path}/complete_dataset.pkl')


def export_train_val_test_set_per_building(buildings, root_path=ROOT, format='csv'):
    """ Writes a list of dataframes to a folder structure, 
        where each site has its own folder and each building has its own csv/pkl file.
    
    Args:
        buildings (list): A list of dataframes, where each dataframe contains the data of one building.
        path (str): The path to the folder where the data should be written to.
        format (str): The format of the files to write. Either 'csv' or 'pkl'.
        
    Returns: 
        None
    """
    def create_folder(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass
    def get_site_id(df):
        return str(int(df['site_id'].iloc[0]))
    def get_building_id(df):
        return str(int(df['building_id'].iloc[0]))
    def get_path_string(df):
        return root_path + r'\site_' + get_site_id(df) + r'\building_' + get_building_id(df)

    for df in buildings:
        site_id = get_site_id(df)
        building_id = get_building_id(df)
        create_folder(root_path + r'\site_' + site_id + r'\building_' + building_id)
        filename = get_path_string(df)
        df = df.drop(columns=['timestamp','site_id','building_id'], axis=1)
        df_train, df_val_test = train_test_split(df, test_size=0.2, shuffle=False)
        df_val, df_test = train_test_split(df_val_test, test_size=0.5, shuffle=False)
        if format == 'csv':
            df_train.to_csv(filename + r'\train.csv', index=False)
            df_val.to_csv(filename + r'\val.csv', index=False)
            df_test.to_csv(filename + r'\test.csv', index=False)
        elif format == 'pkl':
            df_train.to_pickle(filename + r'\train.pkl')
            df_val.to_pickle(filename + r'\val.pkl')
            df_test.to_pickle(filename + r'\test.pkl')
        else:
            raise ValueError('The format must be either "csv" or "pkl".')
    return print('The data has been written to the folder structure.')

def export_complete_dataset_train_val_test(buildings):
    train = []
    val = []
    test = []
    for df in buildings:
        df = df.drop(columns=['timestamp','site_id','building_id'], axis=1)
        df_train, df_val_test = train_test_split(df, test_size=0.2, shuffle=False)
        df_val, df_test = train_test_split(df_val_test, test_size=0.5, shuffle=False)
        train.append(df_train)
        val.append(df_val)
        test.append(df_test)
    train_all = reverse_split_in_buildings(train)
    val_all = reverse_split_in_buildings(val)
    test_all = reverse_split_in_buildings(test)
    train_all.to_csv(ROOT + r'\complete_train.csv', index=False)
    val_all.to_csv(ROOT + r'\complete_val.csv', index=False)
    test_all.to_csv(ROOT + r'\complete_test.csv', index=False)
    return print('The data has been written to the folder structure.')