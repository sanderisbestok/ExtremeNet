import os
import torch
import importlib
import torch.nn as nn
import json


from config import system_configs
from models.py_utils.data_parallel import DataParallel
from db.datasets import datasets

torch.manual_seed(317)

class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss  = loss

    def forward(self, xs, ys, **kwargs):
        preds = self.model(*xs, **kwargs)
        loss  = self.loss(preds, ys, **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module
class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)

    def load_my_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 print("unkown name")
                 continue
            # if isinstance(param, Parameter):
            #     print("instance")
            #     # backwards compatibility for serialized parameters
            #     param = param.data
            if (own_state[name].size() == param.size()):
                print("copy")
                own_state[name].copy_(param)
            else:
                print("andere size")

class NetworkFactory(object):
    def __init__(self, db):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        print("module_file: {}".format(module_file))
        nnet_module = importlib.import_module(module_file)

        self.model   = DummyModule(nnet_module.model(db))
        self.loss    = nnet_module.loss
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(self.network, chunk_sizes=system_configs.chunk_sizes)

        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate, 
                momentum=0.9, weight_decay=0.0001
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self, xs, ys, **kwargs):
        xs = [x.cuda(non_blocking=True) for x in xs]
        ys = [y.cuda(non_blocking=True) for y in ys]

        self.optimizer.zero_grad()
        loss = self.network(xs, ys)
        loss = loss.mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def validate(self, xs, ys, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]

            loss = self.network(xs, ys)
            loss = loss.mean()
            return loss

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading pretrain from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            self.model.load_my_state_dict(params, strict=False)

    def load_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("loading model from {}".format(cache_file))
        with open(cache_file, "rb") as f:
            params = torch.load(f)
            self.model.load_state_dict(params)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)

    def calculate_bboxes(self, cfg_file, iteration):
        print("JOEJOE")
        with open(cfg_file, "r") as f:
            configs = json.load(f)

        train_split = system_configs.train_split
        val_split   = system_configs.val_split
        test_split  = system_configs.test_split

        split = {
            "training": train_split,
            "validation": val_split,
            "testing": test_split
        }["validation"]

        dataset = system_configs.dataset

        testing_db = datasets[dataset](configs["db"], split)

        test_file = "test.{}".format(testing_db.data)
        testing = importlib.import_module(test_file).testing

        result_dir = system_configs.result_dir
        result_dir = os.path.join(result_dir, str(iteration), split)

        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        
        testing(testing_db, self, result_dir, iteration)