import argparse


def get_parser():
    parser = argparse.ArgumentParser("IncLearner",
                                     description="Incremental Learning trainer.")

    # Model related:
    #"cifar_resnet32 resnet18"
    parser.add_argument("-m", "--model", default="fgp", type=str,
                        help="Incremental learner to train.")
    parser.add_argument("-c", "--convnet", default="cifar_resnet32", type=str,
                        help="Backbone convnet.")

    parser.add_argument("--dropout", default=0., type=float,
                        help="Dropout value.")
    parser.add_argument("-he", "--herding", default=None, type=str,
                        help="Method to gather previous tasks' examples.")
    parser.add_argument("-memory", "--memory-size", default=2000, type=int,
                        help="Max number of storable examplars.")

    # Data related:
    #"cifar100 imagenet"
    parser.add_argument("-d", "--dataset", default="cifar100", type=str,
                        help="Dataset to test on.")
    parser.add_argument("-inc", "--increment", default=10, type=int,
                        help="Number of class to add per task.")
    parser.add_argument("-b", "--batch-size", default=128, type=int,
                        help="Batch size.")
    parser.add_argument("-w", "--workers", default=1, type=int,
                        help="Number of workers preprocessing the data.")
    parser.add_argument("-v", "--validation", default=0., type=float,
                        help="Validation split (0. <= x <= 1.).")
    parser.add_argument("-random", "--random-classes", action="store_true", default=False,
                        help="Randomize classes order of increment")
    parser.add_argument("-max-task", "--max-task", default=None, type=int,
                        help="Cap the number of tasks.")
    parser.add_argument("-initial", "--initial", default=10, type=int,
                        help="Number of class of initial task.")

    # Training related:
    #Cifar : lr=2.0, epochs: 70, schedule : epoch 50,64 decay 1/5
    #Imagenet : lr=2.0, epochs : 60, schedule : 20,30,40,50 decay 1/5
    parser.add_argument("-lr", "--lr", default=2., type=float,
                        help="Learning rate.")
    parser.add_argument("-wd", "--weight-decay", default=0.00001, type=float,
                        help="Weight decay.")
    parser.add_argument("-sc", "--scheduling", default=[50,64], nargs="*", type=int,
                        help="Epoch step where to reduce the learning rate.")
    parser.add_argument("-lr-decay", "--lr-decay", default=1/5, type=float,
                        help="LR multiplied by it.")
    parser.add_argument("-opt", "--optimizer", default="sgd", type=str,
                        help="Optimizer to use.")
    parser.add_argument("-e", "--epochs", default=70, type=int,
                        help="Number of epochs per task.")

    # Misc:
    parser.add_argument("--device", default=0, type=int,
                        help="GPU index to use, for cpu use -1.")
    parser.add_argument("--name", default="exp",
                        help="Experience name")
    parser.add_argument("-seed", "--seed", default=[1], type=int, nargs="+",
                        help="Random seed.")
    parser.add_argument("-seed-range", "--seed-range", type=int, nargs=2,
                        help="Seed range going from first number to second (both included).")
    parser.add_argument("-order", "--order", default=[0], type=int, nargs="+",
                        help="data order.")

    return parser
