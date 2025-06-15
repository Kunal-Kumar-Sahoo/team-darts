import os
import yaml
import torch
import psutil
import time

class Client:
    def __init__(self, logger, cid, device, model_class, model_args, data_path, dataset_id, train_batch_size, test_batch_size, minibatch_time):
        self.cid = cid
        self.device = device
        self.logger = logger
        self.minibatch_time = minibatch_time
        
        self.model = model_class(cid, model_args)
            
        with open(
            os.path.join(
                data_path, f"part_{cid}", dataset_id, "train_dataset_config.yaml"
            ),
            "r",
        ) as m:
            meta = yaml.safe_load(m)
            self.num_items = meta["metadata"]["num_items"]
            self.data_distrb = meta["metadata"]["label_distribution"]
    
        self.train_data, self.test_data = self.model.load_data(logger, data_path, dataset_id, cid, train_batch_size, test_batch_size)
        
        self.roundtime = self.minibatch_time * len(self.train_data)
        
        self.train_metrics = dict()
        self.test_metrics = dict()
        
        self.time_util = dict()
        
    def get_num_model_params(self):
        self.logger.info(f"NUM_MODEL_PARAMS: {self.model.count_parameters()}")
        
    def expected_time_util(self):
        return self.minibatch_time
    
    def update_util(self, round_id, epochs):
        self.time_util[round_id] = self.roundtime * epochs
        
    def train(self, round_id, args, batch_size=16, max_minibatches=100):
        self.train_metrics[round_id] = self.model.train_model(self.logger, self.train_data, args, self.device, batch_size, self.cid, round_id, max_minibatches)
        self.update_util(round_id, args["epochs"])

    def test(self, round_id):
        self.test_metrics[round_id] = self.model.test_model_client(self.logger, self.test_data, self.cid)
        self.last_round_tested = round_id
    
    def test_global_data(self, round_id, data):
        self.test_metrics[round_id] = self.model.test_model_client(self.logger, data, self.cid)
        self.last_round_tested = round_id

class ClientCustom:
    def __init__(self, logger, cid, device, model_class, model_args, data_path, dataset_id, train_batch_size, test_batch_size, minibatch_time):
        self.cid = cid
        self.device = device
        self.logger = logger
        self.minibatch_time = minibatch_time
        
        self.model = model_class(cid, model_args)
            
        with open(
            os.path.join(
                data_path, f"part_{cid}", dataset_id, "train_dataset_config.yaml"
            ),
            "r",
        ) as m:
            meta = yaml.safe_load(m)
            self.num_items = meta["metadata"]["num_items"]
            self.data_distrb = meta["metadata"]["label_distribution"]
    
        self.train_data, self.test_data = self.model.load_data(logger, data_path, dataset_id, cid, train_batch_size, test_batch_size)
        
        self.roundtime = self.minibatch_time * len(self.train_data)
        
        self.train_metrics = dict()
        self.test_metrics = dict()
        
        self.time_util = dict()
        
    def get_num_model_params(self):
        self.logger.info(f"NUM_MODEL_PARAMS: {self.model.count_parameters()}")
        
    def expected_time_util(self):
        return self.minibatch_time
    
    def update_util(self, round_id, epochs):
        self.time_util[round_id] = self.roundtime * epochs
        
    def train(self, round_id, args, batch_size=16, max_minibatches=100):
        self.train_metrics[round_id] = self.model.train_model(self.logger, self.train_data, args, self.device, batch_size, self.cid, round_id, max_minibatches)
        self.update_util(round_id, args["epochs"])

    def test(self, round_id):
        self.test_metrics[round_id] = self.model.test_model_client(self.logger, self.test_data, self.cid)
        self.last_round_tested = round_id
    
    def test_global_data(self, round_id, data):
        self.test_metrics[round_id] = self.model.test_model_client(self.logger, data, self.cid)
        self.last_round_tested = round_id