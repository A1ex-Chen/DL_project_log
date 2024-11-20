import argparse
import flwr as fl
from typing import Dict, List, Tuple, Union, Optional
from flwr.common import Metrics
from models import VAE
import matplotlib.pyplot as plt
from flwr.common.typing import Scalar
import torch
import numpy as np
from client import Fl_Client
from utils import test, load_partition
from torch.utils.data import DataLoader
from collections import OrderedDict
import os


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, Scalar]]:

        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        global DATASET, CLIENT, DP
        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")

            np.savez(f"Results/{DATASET}/{CLIENT}/{DP}/weights/FL_round_{server_round}_weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics














def get_evaluate_fn(args):
    """Return an evaluation function for server-side evaluation."""

    return evaluate


def get_fit_cofig(l_local, batch_size, DP):
    return fit_config


def get_evaluate_fig(batch_size, DP, DATASET):
    return evaluate_fig


def get_client_fn(args):
    return client_fn


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply metrics of each client by number of examples used
    # fid = [m["fid"] for number, m in metrics]
    # fid = np.average(fid)
    # Aggregate and return average fid
    met = {}
    for i, m in enumerate(metrics):
        met[i] = m[1]["fid"]
    #return {"fid": fid}
    return met


def main(args):
    num_client = args.client
    DATASET = args.dataset
    DP = args.dp
    batch_size = args.batch_size
    l_epochs = args.l_epochs
    model = VAE(DATASET)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    strategy = SaveModelStrategy(
        fraction_fit=1,
        fraction_evaluate=1,
        min_fit_clients=num_client,
        min_available_clients=num_client,
        min_evaluate_clients=num_client,
        evaluate_fn=get_evaluate_fn(args),
        on_fit_config_fn=get_fit_cofig(l_epochs, batch_size, DP),
        on_evaluate_config_fn=get_evaluate_fig(batch_size, DP, DATASET),
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        evaluate_metrics_aggregation_fn=weighted_average
    )

    history = fl.server.History()
    num_rounds = args.g_epochs
    history = fl.simulation.start_simulation(
        client_fn=get_client_fn(args),
        num_clients=num_client,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy
    )

    # print(history)
    fid_save_path = f"Results/{DATASET}/{num_client}/{DP}/metrics"
    if not os.path.exists(fid_save_path):
        os.makedirss(fid_save_path)
    fid_save_path = fid_save_path + f"/fid_loss_{num_rounds}.npy"
    np.save(fid_save_path, history)


if __name__ == "__main__":
    # simulation

    parser = argparse.ArgumentParser(
        description="Draw figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        help="Dataset",
    )

    parser.add_argument(
        "--client",
        type=int,
        default=2,
        help="Number of clients, 1 for centralized, 2/3/4/5 for federated learning",
    )

    parser.add_argument(
        "--g_epochs",
        type=int,
        default=200,
        help="Global training epochs",
    )

    parser.add_argument(
        "--l_epochs",
        type=int,
        default=200,
        help="Local training epochs",
    )

    parser.add_argument(
        "--dp",
        type=str,
        default="normal",
        help="Disable privacy training and just train with vanilla type",
    )

    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.1,
        help="The ratio of dataset for test",
    )

    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        metavar="B",
        help="Batch size",
    )

    args = parser.parse_args()

    try:
        if torch.backends.mps.is_built():
            args.device = "mps"
        else:
            if torch.cuda.is_available():
                args.device = "cuda:0"
            else:
                args.device = "cpu"
    except AttributeError:
        if torch.cuda.is_available():
            args.device = "cuda:0"
        else:
            args.device = "cpu"

    main(args)