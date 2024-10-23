from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import sbi.inference.trainers
from sbi import inference as sbi_inference
from sbi.inference.trainers.npe import SNPE_C

from sbi.utils import get_log_root
from sbi.utils.user_input_checks  import prepare_for_sbi



def summary_plcr(prefix):
    logdir = Path(
        get_log_root(),
        prefix,
        datetime.now().isoformat().replace(":", "_"),
    )
    return SummaryWriter(logdir)


def run_inference(simulator, prior, build_nn_posterior, ground_truth,
                  meta_parameters, summary_extractor=None, save_rounds=False,
                  seed=42, device="cpu", num_workers=1, max_num_epochs=None,
                  stop_after_epochs=20, training_batch_size=100):
    
    # set seed for numpy and torch
    np.random.seed(seed)
    torch.manual_seed(seed)

    # make a SBI-wrapper on the simulator object for compatibility
    simulator, prior = prepare_for_sbi(simulator, prior)


    if save_rounds:
        # save the ground truth and the parameters
        folderpath = Path.cwd() / "results" / meta_parameters["label"]
        print(folderpath)
        folderpath.mkdir(exist_ok=True, parents=True)
        path = folderpath / "ground_truth.pkl"
        torch.save(ground_truth, path)
        path = folderpath / "parameters.pkl"
        torch.save(meta_parameters, path)

    # setup the inference procedure
    inference = SNPE_C(
        prior=prior,
        density_estimator=build_nn_posterior,
        show_progress_bars=True,
        device=device,
        summary_writer=summary_plcr(meta_parameters["label"])
    )
    # loop over rounds
    posteriors = []
    proposal = prior
    for round_ in range(meta_parameters["n_rd"]):
        print("ROUND", round_)
        # simulate the necessary data
        theta, x = sbi_inference.simulate_for_sbi(
            simulator, proposal, num_simulations=meta_parameters["n_sr"],
            num_workers=num_workers,
        )
        print('simulation theta', theta.size())
        print('simulation x', x.size())

        if 'cuda' in device:
            torch.cuda.empty_cache()

        # extract summary features
        if summary_extractor is not None:
            x = summary_extractor(x)
        # print("avant train")
        #print(inference._neural_net) #None
        # print("density",inference._build_neural_net) # partial function that builds 2 flows
        # train the neural posterior with the loaded data
        nn_posterior = inference.append_simulations(theta, x, proposal=proposal).train(
            num_atoms=10,
            training_batch_size=training_batch_size,
            use_combined_loss=True,
            discard_prior_samples=True,
            max_num_epochs=max_num_epochs,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=True
        ) # TYPE : FACTORIZED FLOW
        print("append simu finie")
        nn_posterior.zero_grad()
        #nn_posterior=ConditionalDensityEstimator(net=nn_posterior, input_shape=nn_posterior.input_shape, condition_shape=nn_posterior.condition_shape)
        posterior = inference.build_posterior(nn_posterior) #TYPE : DIRECT POSTERIOR
        posteriors.append(posterior)
        # save the parameters of the neural posterior for this round
        if save_rounds:
            path = folderpath / f"nn_posterior_round_{round_:02}.pkl"
            #posterior.net.save_state(path)
            posterior.posterior_estimator.save_state(path)


        # set the proposal prior for the next round
        #print(posterior.sample((10,), ground_truth['observation']))
        proposal = posterior.set_default_x(ground_truth['observation']) # TYPE : DIRECT POSTERIOR
    

    return posteriors
