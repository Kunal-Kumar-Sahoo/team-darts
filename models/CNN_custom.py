import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import psutil
import os
import pickle
from math import floor

class CNN_custom(nn.Module):
    def __init__(self, cid, args: dict = None):
        super(CNN_custom, self).__init__()
        self.cid = cid
        self.num_classes = args["num_classes"]

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, padding=1
        )
        self.fc1 = nn.Linear(
            in_features=64 * 4 * 4, out_features=64
        )
        self.fc2 = nn.Linear(in_features=64, out_features=self.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def train_model(self, logger, data, args, device, batch_size, cid, round_id, max_minibatches):
        epochs = args["epochs"]
        lr = args["lr"] 
        cost = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

        optimizer.param_groups.clear()
        optimizer.state.clear()
        optimizer.add_param_group({"params": [p for p in self.parameters()]})

        total_num_minibatches = 0
        total_accuracy = 0
        total_loss = 0.0
        total_time = 0.0
        total_memory = 0.0

        self.train()
        self.to(device)

        process = psutil.Process(os.getpid())
        minibatch_count = 0

        try:
            for _ in range(epochs):
                for train_x, train_label in data:
                    if minibatch_count >= max_minibatches:
                        break
                    minibatch_count += 1
                    start_time = time.time()
                    train_x = train_x.to(device)
                    train_label = train_label.to(device)
                    optimizer.zero_grad()
                    predict_y = self(train_x)
                    loss = cost(predict_y, train_label)
                    loss.backward()
                    optimizer.step()
                    correct = (
                        (torch.argmax(predict_y, 1) == train_label).cpu().float().sum()
                    ).item()
                    batch_accuracy = (correct / len(train_label)) * 100
                    batch_time = time.time() - start_time
                    memory_usage = process.memory_info().rss / (1024 * 1024)
                    total_accuracy += batch_accuracy
                    total_loss += loss.cpu().item()
                    total_time += batch_time
                    total_memory += memory_usage
                    logger.info(f"CID:{cid},ROUND:{round_id},BATCH_SIZE:{batch_size},MINIBATCH:{minibatch_count},LOSS:{loss.item():.4f},ACCURACY:{batch_accuracy:.2f},TIME:{batch_time:.4f},MEMORY:{memory_usage:.2f}MB")
                if minibatch_count >= max_minibatches:
                    break
            total_num_minibatches = minibatch_count

        except Exception as e:
            logger.info(f"Exception in {self.__class__.__name__}.train_model = {e}")
            if total_loss == 0.0:
                total_loss = float('inf')

        avg_loss = total_loss / total_num_minibatches if total_num_minibatches > 0 else float('inf')
        avg_accuracy = total_accuracy / total_num_minibatches if total_num_minibatches > 0 else 0
        avg_time = total_time / total_num_minibatches if total_num_minibatches > 0 else 0
        avg_memory = total_memory / total_num_minibatches if total_num_minibatches > 0 else 0

        logger.info(f"CID:{cid},ROUND:{round_id},BATCH_SIZE:{batch_size},AVG_LOSS:{avg_loss:.4f},AVG_ACCURACY:{avg_accuracy:.2f},AVG_TIME:{avg_time:.4f},AVG_MEMORY:{avg_memory:.2f}MB")
        return {'loss': avg_loss, 'accuracy': avg_accuracy, 'time_per_minibatch': avg_time, 'memory_per_minibatch': avg_memory}

    def test_model(self, logger, data):
        self.eval()
        correct_test = 0
        total_test = 0
        cost = torch.nn.CrossEntropyLoss()
        
        total_loss = 0
        with torch.no_grad():
            for inputs, targets in data:
                outputs = self(inputs)
                loss = cost(outputs, targets)
                _, preds = torch.max(outputs.data, 1)
                total_test += targets.size(0)
                total_loss += loss.item()
                correct_test += (preds == targets).sum().item()

        accuracy = (correct_test / total_test) if total_test > 0 else 0
        avg_loss = total_loss / len(data) if len(data) > 0 else float('inf')
        
        if self.cid == "server":
            logger.info(f"GLOBAL MODEL: Total Accuracy = {accuracy:.4f}, Loss = {avg_loss:.4f}")
        return {'loss': avg_loss, 'accuracy': accuracy}

    def test_model_client(self, logger, data, cid):
        return self.test_model(logger, data)

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