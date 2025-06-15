import os
import sys
import time
import logging
import pickle
from datetime import datetime
import importlib
import random
import numpy as np
import torch
import yaml
from server_custom import Server
from client import ClientCustom as Client

def fix_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False

def setup_client_selection(server, CS_algo, CS_args):
    client_selection = getattr(server, f"client_selection_{CS_algo}", None)
    if client_selection is None:
        raise ValueError(f"Unknown client selection algorithm: {CS_algo}")
    return client_selection, CS_args

def setup_aggregation(server, Agg_algo, Agg_args):
    aggregation = getattr(server, f"aggregate_{Agg_algo}", None)
    if aggregation is None:
        raise ValueError(f"Unknown aggregation algorithm: {Agg_algo}")
    return aggregation, Agg_args

def main(rounds, seed, client_list, client_selection, CS_args, aggregation, Agg_args, client_train_config, client_test_config, batch_size):  
    fix_seed(seed)
    cumulative_stats = {
        'client_selection_time': 0,
        'actual_training_time': 0,
        'aggregation_time': 0,
        'validation_time': 0
    }
    total_minibatches_per_round = 100
    
    prev_accuracy = None  # Track previous round's global accuracy
    for rnd in range(1, rounds+1):
        CS_args["round"] = rnd
        Agg_args["round"] = rnd
        cs_start_time = time.time()
        selected_cids = client_selection(client_list, CS_args)
        cs_time = time.time() - cs_start_time
        cumulative_stats['client_selection_time'] += cs_time
        
        logger.info(f"SELECTED_CLIENTS:{selected_cids}")
        
        num_selected_clients = len(selected_cids)
        base_minibatches = total_minibatches_per_round // num_selected_clients
        extra_minibatches = total_minibatches_per_round % num_selected_clients
        client_minibatch_limits = {cid: base_minibatches + (1 if i < extra_minibatches else 0) for i, cid in enumerate(selected_cids)}
        
        total_train_time = 0
        for cid in selected_cids:
            train_start_time = time.time()
            client_list[cid].train(round_id=rnd, args=client_train_config, batch_size=batch_size, max_minibatches=client_minibatch_limits[cid])
            cumulative_stats['actual_training_time'] += time.time() - train_start_time
        
        agg_start_time = time.time()
        global_wts, client_list = aggregation(selected_cids=selected_cids, client_list=client_list, round=rnd)
        agg_time = time.time() - agg_start_time
        cumulative_stats['aggregation_time'] += agg_time
        
        server.model.load_state_dict(global_wts)
        
        val_start_time = time.time()
        server.test(round_id=rnd)
        val_time = time.time() - val_start_time
        cumulative_stats['validation_time'] += val_time
        
        # Update rewards for bandit algorithm
        server.update_rewards(selected_cids, client_list, rnd, server, prev_accuracy)
        prev_accuracy = server.test_metrics.get(rnd, {}).get('accuracy', 0)
        
        sim_training_time = max(client_list[cid].time_util[rnd] for cid in selected_cids)
        
        logger.info("=== Time Statistics ===")
        logger.info(f"TOTAL_CLIENT_SELECTION_TIME:{cumulative_stats['client_selection_time']:.4f}")
        logger.info(f"TOTAL_ACTUAL_TRAINING_TIME:{cumulative_stats['actual_training_time']:.4f}")
        logger.info(f"SIM_TRAINING_ROUND_TIME:{sim_training_time:.4f}")
        logger.info(f"TOTAL_AGGREGATION_TIME:{cumulative_stats['aggregation_time']:.4f}")
        logger.info(f"TOTAL_VALIDATION_TIME:{cumulative_stats['validation_time']:.4f}")
        logger.info("===========================")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_exp.py <path_to_yaml>")
    else:
        config_yaml_path = sys.argv[1]
        exp_config = None
        try:
            with open(config_yaml_path, 'r') as file:
                exp_config = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: File not found at {config_yaml_path}")
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")
        
        if exp_config is not None:
            CS_algo = exp_config['FL_config']['CS_algo']
            CS_args = exp_config['FL_config']['CS_args']
            Agg_algo = exp_config['FL_config']['Agg_algo']
            Agg_args = exp_config['FL_config']['Agg_args']
            model_id = exp_config['ML_config']['model_id']
            dataset_id = exp_config['ML_config']['dataset_id']
            
            total_rounds = exp_config['FL_config']['total_rounds']
            total_num_clients = exp_config['FL_config']['total_num_clients']
            num_clients_per_round = exp_config['FL_config']['clients_per_round']
            
            client_train_config = exp_config['ML_config']['train_config']
            client_test_config = exp_config['ML_config']['test_config']
            
            torch_device = "cuda" if torch.cuda.is_available() and exp_config['server_config']['use_gpu'] else "cpu"
            seed = exp_config['server_config']['seed']
            
            batch_sizes = [16]
            
            for batch_size in batch_sizes:
                exp_name = f"{CS_algo}_{model_id}_{dataset_id}_bs{batch_size}"
                
                CS_args = {
                    "round": 0,
                    "total_rounds": total_rounds,
                    "num_clients_per_round": num_clients_per_round,
                }
                if 'CS_args' in exp_config:
                    for key, value in exp_config['FL_config']['CS_args'].items():
                        CS_args[key] = value
                
                Agg_args = {}
                if 'Agg_args' in exp_config:
                    for key, value in exp_config['FL_config']['Agg_args'].items():
                        Agg_args[key] = value
                
                try:
                    parent_dir = os.path.dirname(os.path.dirname(exp_config['ML_config']['model_file_path']))
                    if parent_dir not in sys.path:
                        sys.path.append(parent_dir)
                    
                    model_file = os.path.basename(exp_config['ML_config']['model_file_path'])
                    module_name = os.path.splitext(model_file)[0]
                    module_path = f"models.{module_name}"
                    
                    module = importlib.import_module(module_path)
                    model_class = getattr(module, model_id)
                except (ModuleNotFoundError, AttributeError) as e:
                    print(f"Error: loading {model_id} from {exp_config['ML_config']['model_id']}: {e}")
                    continue
                
                initial_model_path = os.path.join(exp_config['ML_config']['initial_model_path'], f"{model_id}.pth")
                if not os.path.exists(initial_model_path):
                    model = model_class(cid="Initial Model", args=exp_config['ML_config']['model_args'])
                    torch.save(model, initial_model_path)
                
                os.makedirs(os.path.join(exp_config['server_config']['save_path'], exp_name), exist_ok=True)
                filename = datetime.now().strftime("%d-%m-%Y %H-%M-%S") + f"_bs{batch_size}.log"
                
                logging.basicConfig(
                    filename=os.path.join(exp_config['server_config']['save_path'], exp_name, filename),
                    filemode="a",
                    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S",
                    level=logging.DEBUG,
                )
                logger = logging.getLogger("test")
                
                logger.info("=== EXPERIMENT CONFIGURATION ===")
                for key, value in exp_config.items():
                    if isinstance(value, dict):
                        logger.info(f"{key}:")
                        for subkey, subvalue in value.items():
                            logger.info(f"  {subkey}: {subvalue}")
                    else:
                        logger.info(f"{key}: {value}")
                logger.info(f"Batch Size: {batch_size}")
                logger.info("=============================")
                
                client_list = []
                for i in range(total_num_clients):
                    client_obj = Client(
                        logger=logger, 
                        cid=i, 
                        device=torch_device,
                        model_class=model_class, 
                        model_args=exp_config['ML_config']['model_args'], 
                        data_path=exp_config['ML_config']['dataset_dir'], 
                        dataset_id=dataset_id, 
                        train_batch_size=batch_size, 
                        test_batch_size=exp_config['ML_config']['test_config']['test_bs'],
                        minibatch_time=exp_config['client_config']['minibatch_time'],
                    )
                    client_obj.model.load_state_dict(torch.load(initial_model_path, weights_only=False).state_dict())
                    client_list.append(client_obj)
                
                server = Server(
                    logger=logger,
                    device=torch_device,
                    model_class=model_class,
                    model_args=exp_config['ML_config']['model_args'],
                    data_path=exp_config['ML_config']['dataset_dir'],
                    dataset_id=dataset_id,
                    test_batch_size=exp_config['ML_config']['test_config']['test_bs'],
                )
                server.model.load_state_dict(torch.load(initial_model_path, weights_only=False).state_dict())
                
                client_selection, CS_args = setup_client_selection(server, CS_algo, CS_args)
                aggregation, Agg_args = setup_aggregation(server, Agg_algo, Agg_args)
                
                main(total_rounds, seed, client_list, client_selection, CS_args, aggregation, Agg_args, client_train_config, client_test_config, batch_size)