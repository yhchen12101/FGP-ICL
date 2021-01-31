import torch
from torch import optim
import models
from convnet import resnet, cifar_resnet
from lib import data

def get_optimizer(params, optimizer, lr, weight_decay=0.0):
    if optimizer == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)

    raise NotImplementedError


def get_convnet(convnet_type, **kwargs):
    if convnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    elif convnet_type == "cifar_resnet32":
        return cifar_resnet.resnet32(**kwargs)

    raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))


def get_model(args):
    if args["model"] == "fgp":
        return models.FGP(args)

    raise NotImplementedError(args["model"])


def get_data(args):
    return data.IncrementalDataset(
        dataset_name=args["dataset"],
        random_order=args["random_classes"],
        shuffle=True,
        batch_size=args["batch_size"],
        workers=args["workers"],
        increment = args["increment"],
        validation_split=args["validation"],
        order = args["order"],
        initial_class_num = args["initial"]
    )


def set_device(args):
    device_type = args["device"]

    if device_type == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(device_type))

    args["device"] = device
