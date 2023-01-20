import os
import flwr as fl
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Tuple, Union, Dict, Optional
from flwr.server.client_proxy import ClientProxy
from flwr.common import (Metrics, EvaluateIns, EvaluateRes, FitIns, FitRes, Parameters, Config, Scalar, NDArrays, ndarrays_to_parameters, parameters_to_ndarrays)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import wandb
import csv
pd.options.plotting.backend = "plotly"

SITE_ID = 8
N_ROUNDS = 50
BATCH_SIZE = 128
N_STEPS = 3
N_FEATURES = 28
site_fraction_fits = {"0": 0.34, "1": 0.37, "2": 0.33, "3": 0.32, "4": 0.35, "5": 0.34, "6": 0.39, "7": 0.54, "8": 0.35, "9": 0.34, "10": 0.43, "11": 1.0, "12": 0.39, "13": 0.33, "14": 0.34, "15": 0.34}


"""GLOBAL HELPER FUNCTIONS"""
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
def get_data_generator(cid, type):
    path = get_building_ids(ROOT_PATH, SITE_ID)[int(cid)]+f'/{type}.csv'
    #path = f'{ROOT_PATH}site_{SITE_ID}/building_{cid}/{type}.csv'
    return CSVDataGenerator(path, BATCH_SIZE, split_sequences_batch_wise)

def get_data_generator_server():
    # path to centralized testset
    path = '/home/azureuser/masterarbeit/server_testset.csv'
    return CSVDataGenerator(path, BATCH_SIZE, split_sequences_batch_wise)


def create_model():
    n_steps = N_STEPS
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
    return model
model = create_model()

def test(datagen, model):
            result = model.predict(datagen)
            g = []
            for i in range(len(datagen)):
                g.append(datagen[i][1])
            ground_truth = np.concatenate(g, axis=0)
            error = abs(np.expm1(result) - np.expm1(ground_truth))
            error_percent = (error/np.expm1(result)*100)
            MSE = np.square(np.subtract(np.expm1(result),np.expm1(ground_truth))).mean()
            RMSE = np.sqrt(MSE)
            MAE = error.mean()
            MAPE = error_percent.mean()
            num_examples_test = len(ground_truth)*BATCH_SIZE
            results = {'MAE': MAE,'MAPE':MAPE, 'RMSE': RMSE}
            return MSE, num_examples_test, results, result, ground_truth

#model = create_model()
#model.compile(optimizer='adam', loss='mse', metrics=['mae'])

def get_parameters(model: tf.keras.Model) -> List[np.ndarray]:
    return [layer for layer in model.get_weights()]

"""GLOBAL VARIABLES"""
ROOT_PATH = '/home/azureuser/masterarbeit/ready_datasets_dummy'
DICT = get_file_path_dict(root_dir=ROOT_PATH)
NUM_CLIENTS = len(DICT[f'site_{SITE_ID}'])
#distributed_metrics_df = pd.DataFrame()



"""DEFINING CLIENT"""

class SmartMeterClient(fl.client.NumPyClient):
    def __init__(self, cid, model, train, val, test) -> None:
        self.cid = cid
        self.model = model
        self.train = train
        self.val = val
        self.test = test
    
    def get_properties(self):
        """Get properties of client."""
        raise Exception('Not implemented. (get_properties)')

    def get_parameters(self, config):
        #Get parameters of the local model.
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.model)
    
    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local Parameters
        self.model.set_weights(parameters)

        # get hyperparameters for this round
        server_round: int = config["server_round"]
        epochs: int = config["local_epochs"]
        batch_size: int = config["batch_size"]
        print(f"[Client {self.cid}, round {server_round}] fit, config: {config}")

        # Return updated model parameters and results
        history = self.model.fit(self.train, steps_per_epoch=len(self.train), validation_data=(self.val), validation_steps=5, epochs=epochs, verbose=1)
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.train)*BATCH_SIZE
        results = {
            "loss": history.history["loss"][0],
            "mae": history.history["mae"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_mae"][0],
        }

        return parameters_prime, num_examples_train, results
    
    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        if config['server_round'] == N_ROUNDS:
            self.model.fit(self.train, steps_per_epoch=len(self.train), epochs=1, verbose=0)
            #self.model.save(f'{RESULTS_PATH}/models_{SITE_ID}_{self.cid}.csv')
        else: 
            pass
        loss, num_examples_test, results, _, _ = test(self.test, self.model)
        return loss, num_examples_test, results

def client_fn(cid: str) -> fl.client.NumPyClient:
    # create model
    model = create_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    #load data_partition
    # cid als Argument für read_csv übergeben, so dass die richtig building_id geladen wird
    train = get_data_generator(cid,"train")
    val = get_data_generator(cid,"val")
    test = get_data_generator(cid,"test") 
    
    return SmartMeterClient(cid, model, train, val, test)


"""DEFINING STRATEGY"""

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        print(sum([fit_res.num_examples for _, fit_res in results]))
        if aggregated_parameters is not None:
            print(f"Saving round {server_round} aggregated_ndarrays...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)
            if server_round == N_ROUNDS:
                np.savez(fr"/home/azureuser/masterarbeit/cross-device_pro_EV/weights-{SITE_ID}.npz", *aggregated_ndarrays)
            else:
                pass
            #create a list of 
            #aggregated_ndarrays.save(f"round-{server_round}-weights.h5")          
        return aggregated_parameters, aggregated_metrics
    #distributed_metrics = wandb.Table(columns=["Round Nr", "MAE", "MAPE", "MSE"])
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        #distributed_metrics = distributed_metrics
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}
        #Call aggregate_evaluate from base class (FedAvg) to aggregate loss and metrics
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        # weigh mae of each client by number of samples
        #maes = [r.metrics["mae"] * r.num_examples for _, r in results]
        maes = [r.metrics["MAE"] * r.num_examples for _, r in results]
        mapes = [r.metrics["MAPE"] * r.num_examples for _, r in results]
        rmses = [r.metrics["RMSE"] * r.num_examples for _, r in results]
        #maes = [num_examples * m['MAE'] for num_examples, m in results]
        #mapes = [num_examples * m['MAPE'] for num_examples, m in results]
        examples = [r.num_examples for _, r in results]

        aggregated_mae = sum(maes) / sum(examples)
        aggregated_mapes = sum(mapes) / sum(examples)
        aggregated_rmse = sum(rmses) / sum(examples)
        metrics_dict = {'Round Nr' : [server_round],
                        'MAE': [aggregated_mae],
                        'MAPE': [aggregated_mapes],
                        'MSE': [aggregated_rmse]}
        
        #distributed_metrics.add_data(server_round, aggregated_mae, aggregated_mapes, aggregated_rmse)
        #wandb.log({"Distributed Metrics": distributed_metrics})
        print(f"Round {server_round} Evaluation aggregated from client results: MAE: {aggregated_mae}, MAPE: {aggregated_mapes}, RMSE: {aggregated_rmse}, loss: {aggregated_loss}")
        
        return aggregated_loss, {'MAE': aggregated_mae,
                                 'MAPE': aggregated_mapes,
                                 'MSE': aggregated_rmse}
    
"""DEFINING SERVER"""
def get_evaluate_fn(model):
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
        loss, num_examples_test, results, pred, real = test(testgen, model)
        mae = results['MAE']
        mape = results['MAPE']
        rmse = results['RMSE']
        #fig = pd.DataFrame([pred, real]).plot()
        #wandb.log({"Round": server_round, "MAE": mae, "MAPE": mape, "RMSE": rmse})
        #wandb.log({'real vs. pred': fig})
        print(f"Round {server_round} of Server Side Evaluation: MAE: {mae}, MAPE: {mape}, RMSE: {rmse}, loss is {loss}")
        return loss, {"MAE": mae,
                    "MAPE": mape,
                    "RMSE": rmse}
    return evaluate


def fit_config(server_round: int) -> Dict[str, Scalar]: #stimmt type annotation?
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            #"learning_rate": 0.001 if server_round > 2 else 0.001,
            "batch_size": BATCH_SIZE,
            "local_epochs": 1,
            "server_round": server_round,
        }
        return config

def eval_config(server_round: int) -> Dict[str, Scalar]: 
    """Return a configuration for validation."""
    val_steps = 5 if server_round < 4 else 10
    server_round = server_round

    return {"val_steps": val_steps,
            "server_round": server_round}

def eval_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply each metric of each client by the number of samples
    maes = [num_examples * m['MAE'] for num_examples, m in metrics]
    mapes = [num_examples * m['MAPE'] for num_examples, m in metrics]
    #building_id = metrics[1][1]['building_id']
    #server_round = metrics[1][1]['server_round']
    #{'building_id': int(self.cid), 'MAE': MAE,'MAPE':MAPE, 'Round': int(config['server_round'])}

    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"MAE": sum(maes) / sum(examples),
            "MAPE": sum(mapes) / sum(examples),}
           # "building_id": building_id,
            #"server_round": server_round}

def fit_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply each metric of each client by the number of samples
    maes = [num_examples * m['mae'] for num_examples, m in metrics]
    losses = [num_examples * m['loss'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"mae": sum(maes) / sum(examples),
            "loss": sum(losses) / sum(examples),
            }

strategy = SaveModelStrategy(
        fraction_fit=site_fraction_fits[str(SITE_ID)],  # Sample 100% of available clients for training
        fraction_evaluate=1,  # Sample 10% of available clients for evaluation
        min_fit_clients=2,  # Never sample less than 2 clients for training
        min_evaluate_clients=2,  # Never sample less than 2 clients for evaluation
        min_available_clients=int(NUM_CLIENTS * 0.75),  # Wait until at least 75 clients are available
        evaluate_fn=get_evaluate_fn(create_model()),
        on_fit_config_fn= fit_config,
        on_evaluate_config_fn=eval_config,
        evaluate_metrics_aggregation_fn=eval_weighted_average,
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(create_model())),
    )

if __name__ == '__main__':
    print(f'THE FOLLOWING TRAINING IS FROM SITE ID {SITE_ID}' )
    #run = wandb.init(project="smartmeter", entity="adrianz", group='' name=f"site_{SITE_ID}")
    #os.mkdir(f'/home/azureuser/masterarbeit/cross-device_pro_EV/weights-{SITE_ID}')
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=N_ROUNDS),
        strategy=strategy,
        )

    lc, ld, mc, md = get_history_to_dataframe(history)
    lc.to_csv(f'/home/azureuser/masterarbeit/cross-device_pro_EV/results/lc-site_{SITE_ID}.csv')
    ld.to_csv(f'/home/azureuser/masterarbeit/cross-device_pro_EV/results/ld-site_{SITE_ID}.csv')
    mc.to_csv(f'/home/azureuser/masterarbeit/cross-device_pro_EV/results/mc-site_{SITE_ID}.csv')
    md.to_csv(f'/home/azureuser/masterarbeit/cross-device_pro_EV/results/md-site_{SITE_ID}.csv')
    print(f'Training Site {SITE_ID} done')

    