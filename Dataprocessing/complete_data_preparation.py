import pandas as pd
import os
import numpy as np
import pickle
import sys

sys.path.append(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\data_preprocessing')
import load_and_clean as lc
import feature_engineering as fe
import data_splitting as ds
root_folder = ds.ROOT
if __name__ == "__main__":
    dataset = lc.load_and_merge_data() #load three csv files and merge them into one dataframe
    dataset = lc.drop_columns(dataset) #drop columns that are not needed
    dataset = pd.get_dummies(dataset, columns=['primary_use'])
    dataset, testset = lc.generate_non_participating_set(dataset) #generate a test set with clients that have missing data
    dataset = lc.check_for_missing_data(dataset, 0.75) #drop clients with more than 75% missing data
    dataset = lc.remove_outliers(dataset)
    dataset = lc.resample_data(dataset) #resample the data for imputing missing values
    buildings = ds.split_in_buildings(dataset) #split the dataset into a list of dataframes, where each dataframe contains the data of one building
    buildings = fe.apply_to_list_of_dfs(buildings)
    #ds.export_complete_dataset_train_val_test(buildings)
    #buildings_reversed = ds.reverse_split_in_buildings(buildings)
    #ds.export_complete_dataset(buildings_reversed, format='csv') # depreceated function
    #ds.export_buildings_per_site(buildings, format='csv') # depreceated function
    ds.export_train_val_test_set_per_building(buildings, format='csv')
    testset = lc.resample_data(testset)
    testsets = ds.split_in_buildings(testset)
    testsets = fe.apply_to_list_of_dfs(testsets)
    ds.export_test_set(testsets, format='csv')
    #ds.export_sites(buildings, format='csv')





