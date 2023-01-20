from multiprocessing import Process, Pipe
import time
import os
import sys
import ssl
#import math
import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
from datetime import datetime
#from sklearn.preprocessing import MinMaxScaler
from collections import OrderedDict
from typing import List, Tuple, Union, Dict, Optional, Callable
from flwr.server.client_proxy import ClientProxy
from flwr import simulation
from flwr.common import (
    Metrics,
    EvaluateIns, 
    EvaluateRes, 
    FitIns, 
    FitRes, 
    Parameters, 
    Config,
    Scalar,
    NDArray,
    NDArrays,
    ndarrays_to_parameters,
    parameters_to_ndarrays,)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def get_history_to_dataframe(h):
        lc = pd.DataFrame(h.losses_centralized).astype(float)
        ld = pd.DataFrame(h.losses_distributed).astype(float)
        mc = pd.DataFrame(h.metrics_centralized).applymap(lambda x: x[1]).astype(float)
        md = pd.DataFrame(h.metrics_distributed).applymap(lambda x: x[1]).astype(float)

        return lc, ld, mc, md

def get_file_path_dict(root_dir: str)-> Dict:
    """Returns a dictionary with the file paths for each site"""
    file_paths = {}
    NUM_SITES = 16
    for i in range(NUM_SITES):
        site_dir = os.path.join(root_dir, f'site_{i}')
        site_files = [os.path.join(site_dir, file) for file in os.listdir(site_dir)]
        file_paths[f'site_{i}'] = site_files
    return file_paths

def get_building_ids(root_dir: str, site: str):
    """Returns a list of building ids"""
    site_dir = os.path.join(root_dir, f'site_{site}')
    site_files = [os.path.join(site_dir, file) for file in os.listdir(site_dir)]
    return site_files

N_ROUNDS = 50
ROOT_PATH = '/home/azureuser/masterarbeit/ready_datasets_dummy/'
RESULTS_PATH = '/home/azureuser/masterarbeit/hierarchical_FL/results/'
N_FEATURES = 28
try:
    os.mkdir(RESULTS_PATH)
except:
    pass
DICT = get_file_path_dict(root_dir=ROOT_PATH)
""" THE EVALUATION OF THE CENTRAL MODEL IS DONE BY EDGE SERVER 12 (EV 11)"""
BATCH_SIZE = 128
# Determine the fraction of clients that participate in each round depending on the number of clients per site
# Using a MinMaxScaler to normalize the number of clients per site to a range of 4 (minimum value) to 86, which is the mean value of all clients per site
"""scaler = MinMaxScaler(feature_range=(4, 86))
n_customers = [[len(DICT[i])] for i in DICT]
normalized_number = scaler.fit_transform(n_customers,)
site_fraction_fits = {i:round(x,2) for i,x in zip(DICT.keys(), (normalized_number/n_customers).flatten())}
site_fraction_fits = {int(k.replace("site_","")):v for k,v in site_fraction_fits.items()}"""
site_fraction_fits = {"0": 0.34, "1": 0.37, "2": 0.33, "3": 0.32, "4": 0.35, "5": 0.34, "6": 0.39, "7": 0.54, "8": 0.35, "9": 0.34, "10": 0.43, "11": 1.0, "12": 0.39, "13": 0.33, "14": 0.34, "15": 0.34}

def split_sequences_batch_wise(batch_data, n_steps):
    input_data = []
    output_data = []
    for i in range(len(batch_data)):
        end_ix = i + n_steps
        if end_ix > len(batch_data) - 1:
            break
        # gather input and output parts of the pattern
        seq_x = batch_data[i:end_ix]
        seq_y = batch_data[end_ix][:1]
        # pad the sequences so that they have the same length
        pad_length = n_steps - len(seq_x) 
        seq_x = np.pad(seq_x, [(0, pad_length), (0, 0)], mode='constant')
        input_data.append(seq_x)
        output_data.append(seq_y)
    return np.array(input_data), np.array(output_data)


class CSVDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_path, batch_size, split_sequences):
        self.csv_path = csv_path
        self.batch_size = batch_size + 3
        self.split_sequences = split_sequences

    def __len__(self):
        with open(self.csv_path) as f:
            num_rows = sum(1 for _ in f)
        return int(np.ceil(num_rows / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_data = []
        with open(self.csv_path, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader) # if the file contains headers
            for i, row in enumerate(csvreader):
                if i >= idx * self.batch_size and i < (idx + 1) * self.batch_size:
                    batch_data.append(list(map(float, row)))
                if i >= (idx + 1) * self.batch_size:
                    break
        return self.split_sequences(batch_data, 3)

"""GLOBAL HELPER FUNCTIONS"""
def get_data_generator(cid, sid, type):
    path = get_building_ids(ROOT_PATH, int(sid))[int(cid)]+f'/{type}.csv'
    #path = f'{ROOT_PATH}site_{int(sid)}/building_{cid}/{type}.csv'
    return CSVDataGenerator(path, 128, split_sequences_batch_wise)

def get_data_generator_server():
    # path to centralized testset
    path = f'/home/azureuser/masterarbeit/server_testset.csv'
    return CSVDataGenerator(path, 128, split_sequences_batch_wise)



""" global functions for the Client-Instances"""
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences)-1:
			break


		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix,:1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def train_val_test_split_supervised(d,n_steps):
    n = len(d)
    train_df = d[0:int(n*0.7)]
    val_df = d[int(n*0.7):int(n*0.85)]
    test_df = d[int(n*0.85):]
    train_x, train_y = split_sequences(train_df.values, n_steps)
    val_x, val_y = split_sequences(val_df.values, n_steps)
    test_x, test_y = split_sequences(test_df.values, n_steps)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)

def test_model(datagen, model):
            result = model.predict(datagen, verbose=0)
            g = []
            for i in range(len(datagen)):
                g.append(datagen[i][1])
            ground_truth = np.concatenate(g, axis=0)
            error = abs(np.expm1(result) - np.expm1(ground_truth))
            error_percent = (error/np.expm1(result)*100)
            MSE = np.square(np.subtract(np.expm1(result),np.expm1(ground_truth))).mean()
            RMSE = np.sqrt(MSE)
            MAE = error.mean()
            MAPE = abs(error_percent.mean())
            num_examples_test = len(ground_truth)*BATCH_SIZE
            results = {'MAE': MAE,'MAPE':MAPE, 'RMSE': RMSE}
            return MSE, num_examples_test, results, result, ground_truth

def load_data_for_each_client(cid):
    current_path = '/home/adrianz/Masterarbeit/ready_data/site_0'
    try: 
        df = pd.read_pickle(current_path + fr'\building_{cid}.pkl')
    except FileNotFoundError:
        list_of_files = os.listdir(current_path)
        building_ids = [int(file.split('_')[1].split('.')[0]) for file in list_of_files]
        buildings_ids = sorted(building_ids)
        cid = str(min([number for number in building_ids if number >= int(cid)]))
        if (cid == None) or (int(cid) == max(building_ids)):
            i = current_path[-1]
            current_path = current_path[:-1] + str(int(i)+1)
            df = pd.read_pickle(current_path + f'/building_{cid}.pkl')
        else:
            df = pd.read_pickle(current_path + f'/building_{cid}.pkl')
    return df

    
""" global functions for the Server-Instances"""
def get_evaluate_fn(model):
    dataset = pd.read_pickle('/home/adrianz/Masterarbeit/ready_data/test_29.pkl')
    #dataset = dataset.drop(columns=['trend'])
    x_val, y_val = split_sequences(dataset.values, 3)
    
    def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, Scalar]
   ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
       model.set_weights(parameters) # Update model with the latest parameters
       loss, mae = model.evaluate(x_val, y_val) # Evaluate model on validation set
       print(f"Round {server_round} MAE of Server Side Evaluation: {mae}, loss is {loss}")
       return loss, {"mae": mae}
       
    return evaluate
def get_evaluate_fn_gen(model):
    """ This function evaluates the model on the data that is stored on the server"""
    #dataset = pd.read_pickle('/home/adrianz/Masterarbeit/ready_data/test_29.pkl')
    #x_val, y_val = split_sequences(dataset.values, N_STEPS)
    testgen = get_data_generator_server()
    
    def evaluate(
    server_round: int,
    parameters: NDArrays,
    config: Dict[str, fl.common.Scalar]
   ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        model.set_weights(parameters) # Update model with the latest parameters
        #loss, mae = model.evaluate(x_val, y_val) # Evaluate model on validation set
        loss, num_examples_test, results, pred, real = test_model(testgen, model)
        mae = results['MAE']
        mape = results['MAPE']
        rmse = results['RMSE']
        #fig = pd.DataFrame([pred, real]).plot()
        #wandb.log({"Round": server_round, "MAE": mae, "MAPE": mape, "RMSE": rmse})
        #wandb.log({'real vs. pred': fig})
        print(f"Round {server_round} of Server Side Evaluation: MAE: {mae}, MAPE: {mape}, RMSE: {rmse}, loss is {loss}")
        return loss, {'MAE': mae,
                    'MAPE': mape,
                    'RMSE': rmse}
    return evaluate

def fit_config(server_round: int):
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            #"learning_rate": 0.001 if server_round > 2 else 0.001,
            "batch_size": 128,
            "local_epochs": 1, #1 if server_round > 2 else 3,
            "server_round": server_round,
        }
        return config

def eval_config(server_round: int):
    """Return a configuration for validation."""
    val_steps = 5, # if server_round < 4 else 10
    server_round = server_round

    return {"val_steps": val_steps,
    "server_round": server_round}

def eval_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply each metric of each client by the number of samples
    maes = [num_examples * m['mae'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(maes) / sum(examples)}

def eval_weighted_average_gen(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply each metric of each client by the number of samples
    maes = [num_examples * m['MAE'] for num_examples, m in metrics]
    mapes = [num_examples * m['MAPE'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"MAE": sum(maes) / sum(examples),
            "MAPE": sum(mapes) / sum(examples),}

def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply each metric of each client by the number of samples
    maes = [num_examples * m['mae'] for num_examples, m in metrics]
    losses = [num_examples * m['loss'] for num_examples, m in metrics]
    #val_maes = [num_examples * m['val_mae'] for num_examples, m in metrics]
    #val_losses = [num_examples * m['val_loss'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(maes) / sum(examples),
            "loss": sum(losses) / sum(examples),
            }

def get_parameters(model: tf.keras.Model) -> List[np.ndarray]:
    return [layer for layer in model.get_weights()]



def sum_weights(weights):
    # get the weights of the model
    #weights = model.weights

    # flatten the weights
    flat_weights = [w.flatten() for w in weights]

    # sum of all weights
    total_sum = sum([sum(w) for w in flat_weights])

    return total_sum

# ChatGPTs version
arr = [1, 2, 3, 4, 5]
class Worker:
    def __init__(self):
        # Create pipes for communication between processes
        self.cs1, self.es1 = Pipe()
        self.cs2, self.es2 = Pipe()
        self.cs3, self.es3 = Pipe()
        self.cs4, self.es4 = Pipe()
        # create pipes until we reach 13 Pipes
        self.cs5, self.es5 = Pipe()
        self.cs6, self.es6 = Pipe()
        self.cs7, self.es7 = Pipe()
        self.cs8, self.es8 = Pipe()
        self.cs9, self.es9 = Pipe()
        self.cs10, self.es10 = Pipe()
        self.cs11, self.es11 = Pipe()
        self.cs12, self.es12 = Pipe()
        self.cs13, self.es13 = Pipe()

        # Create processes
        self.p0 = Process(target=self.central_server, args=(self.cs1, self.cs2, self.cs3, self.cs4, self.cs5, self.cs6, self.cs7, self.cs8, self.cs9, self.cs10, self.cs11, self.cs12, self.cs13))
        self.p1 = Process(target=self.edge_server1, args=(self.es1,))
        self.p2 = Process(target=self.edge_server2, args=(self.es2,))
        self.p3 = Process(target=self.edge_server3, args=(self.es3,))
        self.p4 = Process(target=self.edge_server4, args=(self.es4,))
        self.p5 = Process(target=self.edge_server5, args=(self.es5,))
        self.p6 = Process(target=self.edge_server6, args=(self.es6,))
        self.p7 = Process(target=self.edge_server7, args=(self.es7,))
        self.p8 = Process(target=self.edge_server8, args=(self.es8,))
        self.p9 = Process(target=self.edge_server9, args=(self.es9,))
        self.p10 = Process(target=self.edge_server10, args=(self.es10,))
        self.p11 = Process(target=self.edge_server11, args=(self.es11,))
        self.p12 = Process(target=self.edge_server12, args=(self.es12,))
        self.p13 = Process(target=self.edge_server13, args=(self.es13,))


    def central_server(self, conn1, conn2, conn3, conn4, conn5, conn6, conn7, conn8, conn9, conn10, conn11, conn12, conn13):
        connections = [conn1, conn2, conn3, conn4, conn5, conn6, conn7, conn8, conn9, conn10, conn11, conn12, conn13]
        # Initialize array
        global_round = 0
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        #arr = [1, 2, 3, 4, 5]
        initial_weights = model.get_weights()
        initial_sum = sum_weights(initial_weights)
        print(f'Starting array:{initial_sum}')
        # Send array to worker2 and worker3
        [conn.send(initial_weights) for conn in connections]
        

        
        """ ROUND 1 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 2 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 3 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 4 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 5 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 6 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 7 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 8 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 9 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 10 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 11 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 12 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 13 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 14 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 15 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 16 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 17 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 18 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 19 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 20 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 21 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 22 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 23 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 24 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 25 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 26 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 27 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 28 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 29 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 30 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 31 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 32 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 33 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 34 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 35 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 36 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 37 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 38 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 39 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 40 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 41 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 42 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 43 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 44 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 45 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 46 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 47 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 48 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 49 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]
        total_sum = sum_weights(new_weights)
        print(f'Central Server: Averaged result Round {global_round}: {total_sum}')
        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)

        """ ROUND 50 """
        global_round += 1
        weights_1, n1 = conn1.recv()
        weights_2, n2 = conn2.recv()
        weights_3, n3 = conn3.recv()
        weights_4, n4 = conn4.recv()
        weights_5, n5 = conn5.recv()
        weights_6, n6 = conn6.recv()
        weights_7, n7 = conn7.recv()
        weights_8, n8 = conn8.recv()
        weights_9, n9 = conn9.recv()
        weights_10, n10 = conn10.recv()
        weights_11, n11 = conn11.recv()
        weights_12, n12 = conn12.recv()
        weights_13, n13 = conn13.recv()
        # Average the results
        prior_weights = list(zip(weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, weights_7, weights_8, weights_9, weights_10, weights_11, weights_12, weights_13))
        new_weights = [sum(x*y for x, y in zip(prior_weights[i], [n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13])) / sum([n1, n2, n3, n4, n5, n6, n7, n8, n9, n10, n11, n12, n13]) for i in range(len(prior_weights))]

        [conn.send(new_weights) for conn in connections]
        np.savez(fr"/home/azureuser/masterarbeit/hierarchical_FL/central weights/central_weights_round_{global_round}.npz", *new_weights)
        time.sleep(.5)



        

       


        

    def edge_server1(self, conn2):
        SITE_ID = 15
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'RMSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server2(self, conn2):
        SITE_ID = 1
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server3(self, conn2):
        SITE_ID = 2
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge Server {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server4(self, conn2):
        SITE_ID = 3
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server5(self, conn2):
        SITE_ID = 4
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server6(self, conn2):
        SITE_ID = 5
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge Server {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server7(self, conn2):
        SITE_ID = 6
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server8(self, conn2):
        SITE_ID = 7
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server9(self, conn2):
        SITE_ID = 8
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge 1: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server10(self, conn2):
        SITE_ID = 13
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge 1: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server11(self, conn2):
        SITE_ID = 10
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    def edge_server12(self, conn2):
        SITE_ID = 11
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        lc.to_csv(f'{RESULTS_PATH}losses_centralized_hier.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        mc.to_csv(f'{RESULTS_PATH}metrics_centralized_hier.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history


    def edge_server13(self, conn2):
        SITE_ID = 14
        NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
        FRACTION_FIT = site_fraction_fits[str(SITE_ID)]

        params = conn2.recv()
        n_steps = 3
        n_features = N_FEATURES
        model = tf.keras.models.Sequential(
        [
                tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                tf.keras.layers.RepeatVector(n_steps),
                tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                tf.keras.layers.Dense(100),
                tf.keras.layers.Dense(1)
            ]
        )
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        model.set_weights(params)
        print(f'Edge {SITE_ID}: Initial weights: {sum_weights(model.get_weights())}')
        

        class SmartMeterClient(fl.client.NumPyClient):
            def __init__(self, cid, model, train, val, test) -> None:
                self.cid = cid
                self.model = model
                self.train = train
                self.val = val
                self.test = test

            def get_parameters(self, config):
                """Get parameters of the local model"""
                return self.model.get_weights()
            
            def get_properties(self):
                """Get properties of client."""
                raise Exception('Not implemented. (get_properties)')

            """def get_parameters(self, config):
                Get parameters of the local model
                raise Exception('Not implemented, server-side parameter intialization. (get_parameters)')"""

            def fit(self, parameters, config):
                """Train parameters on the locally held training set."""

                # Update local Parameters
                self.model.set_weights(parameters)

                # get hyperparameters for this round
                server_round: int = config["server_round"]
                epochs: int = config["local_epochs"]
                batch_size: int = config["batch_size"]
                print(f"[Edge Server {SITE_ID} Client {self.cid}, round {server_round}] fit, config: {config}")

                # Return updated model parameters and results
                history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5 , epochs=epochs, verbose=0)
                parameters_prime = self.model.get_weights()
                num_examples_train = len(self.train)*128
                results = {
                    "loss": history.history["loss"][0],
                    "mae": history.history["mae"][0],
                    "val_loss": history.history["val_loss"][0],
                    "val_accuracy": history.history["val_mae"][0],
                }

                return parameters_prime, num_examples_train, results
            
            def evaluate(self, parameters, config):
                self.model.set_weights(parameters)
                #PERSONALIZATION STEP ONLY IN THE LAST ROUND
                if config['server_round'] == N_ROUNDS:
                    self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
                else: 
                    pass
                loss, num_examples_test, results, _, _ = test_model(self.test, self.model)
                return loss, num_examples_test, results
        
        class SaveModelStrategy(fl.server.strategy.FedAvg):
            def aggregate_fit(
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, FitRes]],
                failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            # Aggregate the training results and save the model weights to disk.

                # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
                aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
                n_examples_fit = sum([fit_res.num_examples for _, fit_res in results])
                if aggregated_parameters is not None:
                    print(f"Edge Server {SITE_ID} Saving round {server_round} aggregated_ndarrays...")
                    # Convert `Parameters` to `List[np.ndarray]`
                    aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(aggregated_parameters)
                    #np.savez(fr"weights\round-{server_round}-weights.npz", *aggregated_ndarrays)
                    conn2.send((aggregated_ndarrays, n_examples_fit))
                    print(f'Edge Server {SITE_ID} Summe: {sum_weights(aggregated_ndarrays)}')
                    #create a list of 
                    #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")
                    new_aggregated_parameters = conn2.recv()
                    new_aggregated_parameters = ndarrays_to_parameters(new_aggregated_parameters) 
                return new_aggregated_parameters, aggregated_metrics

            def aggregate_evaluate(
                #aggregate federated evaluation results from clients
                self,
                server_round: int,
                results: List[Tuple[ClientProxy, EvaluateRes]],
                failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
                if not results:
                    return None, {}
                #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
                aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

                # weigh mae of each client by number of samples

                maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
                mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
                rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
                examples = [r.num_examples for _, r in results]

                aggregated_mae = sum(maes) / sum(examples)
                aggregated_mapes = sum(mapes) / sum(examples)
                aggregated_rmse = sum(rmses) / sum(examples)
                metrics_dict = {'Round Nr' : [server_round],
                                'MAE': [aggregated_mae],
                                'MAPE': [aggregated_mapes],
                                'MSE': [aggregated_rmse]}

                print(f"Edge Server {SITE_ID} Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
            
                return aggregated_loss, {'MAE': aggregated_mae,
                                    'MAPE': aggregated_mapes,
                                    'MSE': aggregated_rmse}

        #NUM_CLIENTS = 3
        strategy = SaveModelStrategy(
            fraction_fit=FRACTION_FIT,  # Sample 100% of available clients for training
            fraction_evaluate=1,  # Sample 10% of available clients for evaluation
            min_fit_clients=2,  # Never sample less than 2 clients for training
            min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
            min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
            #evaluate_fn=get_evaluate_fn_gen(model),
            on_fit_config_fn= fit_config,
            on_evaluate_config_fn=eval_config,
            evaluate_metrics_aggregation_fn=eval_weighted_average_gen,
            fit_metrics_aggregation_fn=fit_weighted_average,
            initial_parameters = ndarrays_to_parameters(get_parameters(model)),
        )
        def client_fn(cid: str) -> fl.client.NumPyClient:
            #load data_partition
            # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
            train = get_data_generator(cid,SITE_ID,"train")
            val = get_data_generator(cid,SITE_ID,"val")
            test = get_data_generator(cid,SITE_ID,"test") 
            #df = pd.read_pickle(r'C:\Users\adria\Documents\Masterarbeit\Federated Learning\FLOWEr\ASHRAE_energy_prediction\ready_data\site_0\building_' + str(cid) + '.pkl')
            n_steps = 3
            # create model
            n_features = N_FEATURES
            model = tf.keras.models.Sequential(
            [
                    tf.keras.layers.LSTM(50, activation='relu',input_shape=(n_steps,n_features)),
                    tf.keras.layers.RepeatVector(n_steps),
                    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=False),
                    tf.keras.layers.Dense(100),
                    tf.keras.layers.Dense(1)
                ]
            )
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])

            return SmartMeterClient(cid, model, train, val, test)
        history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )
        lc, ld, mc, md = get_history_to_dataframe(history)
        #lc.to_csv(f'{RESULTS_PATH}lc_hier_{SITE_ID}.csv')
        ld.to_csv(f'{RESULTS_PATH}ld_hier_{SITE_ID}.csv')
        #mc.to_csv(f'{RESULTS_PATH}mc_hier{SITE_ID}.csv')
        md.to_csv(f'{RESULTS_PATH}md_hier_{SITE_ID}.csv')

        return history

    
    




    def start_workers(self):
        self.p1.start()
        self.p2.start()
        self.p3.start()

    def terminate_workers(self):
        self.p1.terminate()
        self.p2.terminate()
        self.p3.terminate()
import ray
if __name__ == '__main__':
    startTime = datetime.now()
    # Start processes
    worker = Worker()
    worker.p0.start()
    worker.p1.start()
    worker.p2.start()
    worker.p3.start()
    worker.p4.start()
    worker.p5.start()
    worker.p6.start()
    worker.p7.start()
    worker.p8.start()
    worker.p9.start()
    worker.p10.start()
    worker.p11.start()
    worker.p12.start()
    worker.p13.start()

    worker.p0.join()
    worker.p1.join()
    worker.p2.join()
    worker.p3.join()
    worker.p4.join()
    worker.p5.join()
    worker.p6.join()
    worker.p7.join()
    worker.p8.join()
    worker.p9.join()
    worker.p10.join()
    worker.p11.join()
    worker.p12.join()
    worker.p13.join()

    print(datetime.now() - startTime)
    

