
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from time import time
from collections import defaultdict
from torchvision import models
import os
import pickle
from math import floor

class MobileNetV2_Model(nn.Module):
    def __init__(self, cid, args: dict = None):
        super(MobileNetV2_Model, self).__init__()
        self.cid = cid
        self.num_classes = args["num_classes"]
        self.model = models.mobilenet_v2(weights=True)
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, self.num_classes)

    def forward(self, x):
        return self.model(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train_model(self, logger, data, args, device):
        epochs = args["epochs"]
        lr = args["lr"] 
        print("Training for client ", self.cid)
        cost = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        optimizer.param_groups.clear()
        optimizer.state.clear()
        optimizer.add_param_group({"params": [p for p in self.parameters()]})

        total_num_minibatches = 0
        accuracy = 0
        total_loss, last_loss = 0.0, 0.0

        self.train()
        start_time = time()
        self.to(device)

        try:
            for _ in range(epochs):
                total_num_minibatches += len(data)
                correct = 0
                total_data = 0
                for train_x, train_label in data:
                    train_x = train_x.to(device)
                    train_label = train_label.to(device)
                    optimizer.zero_grad()
                    predict_y = self(train_x)
                    loss = cost(predict_y, train_label)
                    loss.backward()
                    optimizer.step()
                    correct += (
                        (torch.argmax(predict_y, 1) == train_label).cpu().float().sum()
                    ).item()
                    total_data += len(train_label)
                accuracy = round((correct / total_data) * 100, 3)
                last_loss = loss.cpu().item()  # Store the last loss
                total_loss += last_loss

        except Exception as e:
            print(
                f"Exception in {self.__class__.__name__}.train_model = ", e
            )
            # If an exception occurs, use the last recorded loss
            if last_loss == 0.0:
                last_loss = float('inf')  # Use infinity if no loss was recorded

        logger.info(f"TIME_TAKEN:{time()-start_time}")
        logger.info(f"MINIBATCHES:{total_num_minibatches}")
        logger.info(f"LOSS:{last_loss}")
        logger.info(f"ACCURACY:{accuracy}")
        if len(data) > 0:
            return {'loss': total_loss / len(data), 'accuracy': accuracy}
        return {'loss': None, 'accuracy': None}

    def test_model(self, logger, data):
        self.eval()
        correct_test = 0
        total_test = 0
        cost = torch.nn.CrossEntropyLoss()
        
        # Label-wise metrics containers
        labelwise_correct = defaultdict(int)
        labelwise_total = defaultdict(int)
        total_loss = 0

        accuracy = 0
        with torch.no_grad():
            for inputs, targets in data:
                outputs = self(inputs)
                loss = cost(outputs, targets)

                # Predictions
                _, preds = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                total_loss += loss.item()

                # Update metrics for each label
                for label in range(self.num_classes):
                    mask = targets == label
                    labelwise_correct[label] += (preds[mask] == label).sum().item()
                    labelwise_total[label] += mask.sum().item()

        # Calculate final metrics for each label
        for label in range(self.num_classes):
            accuracy = (
                labelwise_correct[label] / labelwise_total[label]
                if labelwise_total[label] > 0
                else 0
            )
            total_accuracy = sum(list(labelwise_correct.values()))/sum(list(labelwise_total.values()))
            if self.cid == "server":
                print(f"GLOBAL MODEL: Label {label} Accuracy = {accuracy:.2f}")
                #logger.info(f"GLOBAL MODEL: Label {label} Accuracy = {accuracy:.2f}")

        if self.cid == "server":
            print(
                f"GLOBAL MODEL: Total Accuracy = {total_accuracy}, Loss = {total_loss/len(data)}"
            )
            logger.info(
                f"GLOBAL MODEL: Total Accuracy = {total_accuracy}"
            )
            logger.info(f"GLOBAL MODEL: Loss = {total_loss/len(data)}")
        return {'loss': total_loss / len(data), 'accuracy': total_accuracy}


    def load_data(self, logger, dataset_path, dataset_id, cid, train_batch_size, test_batch_size):
        if cid == "server":
            if "coreset" in dataset_path:
                path = dataset_path
            else: 
                path = os.path.join(dataset_path, "test_data.pth")
            if "coreset" in dataset_path:
                try:
                    with open(path, 'rb') as f:
                        dataset = pickle.load(f)
                except Exception as e:
                    print("Exception caught from MobNetV2 dataloader :: ", e)
            if "CIFAR10_NIID3" in dataset_path or "dirichlet" in dataset_path:
                try:
                    dataset = torch.load(path, weights_only=False).dataset
                except Exception as e:
                    print("Exception caught from MobNetV2 dataloader :: ", e)
            else:     
                try:
                    dataset = torch.load(path, weights_only=False)
                except Exception as e:
                    print("Exception caught from MobNetV2 dataloader :: ", e)
                
            train_loader = None
            test_loader = torch.utils.data.DataLoader(
                dataset, shuffle=False, batch_size=test_batch_size
            )
            logger.info(f"GLOBAL_DATA_LOADED, NUM_ITEMS:{len(dataset)}")

        else:
            if "CIFAR10_NIID3" in dataset_path:
                try:
                    dataset = torch.load(os.path.join(dataset_path, f"part_{cid}", dataset_id, "train_data.pth")).dataset
                except Exception as e:
                    print("Exception caught from CNN dataloader :: ", e)
            elif "dirichlet" in dataset_path:
                try:
                    with open(os.path.join(dataset_path, f"part_{cid}", dataset_id, "train_data.pth"), 'rb') as f:
                        dataset = pickle.load(f)
                except Exception as e:
                    print("Exception caught from CNN dataloader :: ", e)    
                
            dataset_len = len(dataset)

            split_idx = floor(0.90 * dataset_len)

            train_dataset = torch.utils.data.Subset(dataset, list(range(0, split_idx)))
            test_dataset = torch.utils.data.Subset(
                dataset, list(range(split_idx, dataset_len))
            )

            train_loader = torch.utils.data.DataLoader(
                train_dataset, shuffle=True, batch_size=train_batch_size, drop_last=False,
            )
            test_loader = torch.utils.data.DataLoader(
                test_dataset, shuffle=True, batch_size=test_batch_size, drop_last=False,
            )
            logger.info(
                f"CID{cid}_DATA_LOADED, NUM_ITEMS:{len(train_dataset)}/{len(test_dataset)}"
            )

        return train_loader, test_loader
