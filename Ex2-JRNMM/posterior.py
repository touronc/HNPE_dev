
import torch

# Imports for the SBI package
from sbi.neural_nets.net_builders.flow import build_maf
from pyknos.nflows.distributions import base


class AggregateInstances(torch.nn.Module):
    def __init__(self, mean=True):

        super().__init__()
        self.mean = mean

    def forward(self, x):
        if x.shape[2] == 1:
            return x.view(len(x), -1)
        else:
            if self.mean:
                xobs = x[:, :, 0]  # n_batch, n_embed
                xagg = x[:, :, 1:].mean(dim=2)  # n_batch, n_embed
                x = torch.cat([xobs, xagg], dim=1)  # n_batch, 2*n_embed
                return x
            else:
                return x.view(len(x), -1)


class IdentityJRNMM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, n_extra=0):
        return x


class StackContext(torch.nn.Module):
    def __init__(self, embedding_net):
        super().__init__()
        self.embedding_net = embedding_net

    def forward(self, y):
        """
        Parameters
        ----------
        y : torch.Tensor, shape (n_batch, n_times + 1)
            Input of the StackContext layer.

        Returns
        --------
        context : torch.Tensor, shape (n_batch, n_embed + 1)
            Context where the input y has been encoded, except the last entry
            which is pass thru.
        """
        # The embedding net expect an extra dimension to handle n_extra. Add it
        # in x and remove it in x_embeded
        x = y[:, :-1, None]
        x_embed = self.embedding_net(x, n_extra=0)[:, :, 0]
        theta = y[:, -1:]
        return torch.cat([x_embed, theta], dim=1)


class JRNMMFlow_nflows_base(base.Distribution):

    def __init__(self, batch_theta, batch_x, embedding_net, n_layers=10,
                 z_score_theta=True, z_score_x=True, aggregate=True):

        super().__init__()

        embedding_net = torch.nn.Sequential(
            embedding_net,
            AggregateInstances(mean=aggregate)
        )
        self._embedding_net = embedding_net

        # instantiate the flow
        flow = build_maf(batch_x=batch_theta,
                         batch_y=batch_x,
                         z_score_x=z_score_theta,
                         z_score_y=z_score_x,
                         embedding_net=embedding_net,
                         num_transforms=n_layers)

        self._flow = flow

    def _log_prob(self, inputs, context):
        logp = self._flow.log_prob(inputs, context)
        return logp

    def _sample(self, num_samples, context):
        samples = self._flow.sample(num_samples, context)[0]
        return samples

    def save_state(self, filename):
        state_dict = {}
        state_dict['flow'] = self._flow.state_dict()
        torch.save(state_dict, filename)

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location='cpu')
        self._flow.load_state_dict(state_dict['flow'])


class JRNMMFlow_nflows_factorized(base.Distribution):

    def __init__(self, batch_theta, batch_x, embedding_net, n_layers_factor=5,
                 z_score_theta=True, z_score_x=True):

        super().__init__()

        # flow_1 estimates p(gain | x, x1, ..., xn)
        # create a new net that embeds all n+1 observations and then aggregates
        # n of them via a sum operation
        embedding_net_1 = torch.nn.Sequential(
            embedding_net,
            AggregateInstances(mean=True)
        )
        self._embedding_net_1 = embedding_net_1

        # choose whether the embedding of the context should be done inside
        # the flow object or not; this can have an impact over the z-scoring
        batch_theta_1 = batch_theta[:, -1:]
        batch_context_1 = batch_x
        flow_1 = build_maf(batch_x=batch_theta_1,
                           batch_y=batch_context_1,
                           z_score_x=z_score_theta,
                           z_score_y=z_score_x,
                           embedding_net=embedding_net_1,
                           num_transforms=n_layers_factor)

        self._flow_1 = flow_1

        # flow_2 estimates p(C, mu, sigma, ... | x, gain)
        # create a new embedding next that handles the fact of having
        # a context that is a stacking of the embedded observation x
        # and the gain parameter
        embedding_net_2 = StackContext(embedding_net)
        self._embedding_net_2 = embedding_net_2

        batch_theta_2 = batch_theta[:, :-1]
        batch_context_2 = torch.cat(
            [batch_x[:, :, 0], batch_theta[:, -1:]],
            dim=1
        )  # shape (n_batch, n_times+1)
        flow_2 = build_maf(batch_x=batch_theta_2,
                           batch_y=batch_context_2,
                           z_score_x=z_score_theta,
                           z_score_y=z_score_x,
                           embedding_net=embedding_net_2,
                           num_transforms=n_layers_factor)

        self._flow_2 = flow_2

    def _log_prob(self, inputs, context):

        # logprob of the flow that models p(gain | x, x1, ..., xn)
        context_1 = context
        theta_1 = inputs[:, -1:]  # gain is the last parameter
        logp_1 = self._flow_1.log_prob(theta_1, context_1)

        # logprob of the flow that models p(C, mu, sigma | x, gain)
        gain = inputs[:, -1:]
        context_2 = torch.cat([context[:, :, 0], gain], dim=1)
        theta_2 = inputs[:, :-1]
        logp_2 = self._flow_2.log_prob(theta_2, context_2)

        return logp_1 + logp_2

    def _sample(self, num_samples, context):
        """Draw sample from the posterior.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw from the posterior.
        context : torch.Tensor, shape (n_ctx, n_times, 1 + n_extra)
            Conditionning for the draw.

        Returns
        -------
        samples : shape (n_ctx, num_samples, n_params)
            Sample drawn from the distribution.
        """

        context_1 = context
        # shape (n_samples, 1)
        samples_flow_1 = self._flow_1.sample(num_samples, context_1)[0]
        context_2 = torch.cat([context[:, :, 0].repeat(num_samples, 1),
                               samples_flow_1], dim=1)
        context_2 = self._flow_2._embedding_net(context_2)
        noise = self._flow_2._distribution.sample(num_samples)
        samples_flow_2, _ = self._flow_2._transform.inverse(noise,
                                                            context=context_2)

        samples = torch.cat([samples_flow_2, samples_flow_1], dim=1)
        return samples

    def save_state(self, filename):
        state_dict = {}
        state_dict['flow_1'] = self._flow_1.state_dict()
        state_dict['flow_2'] = self._flow_2.state_dict()
        torch.save(state_dict, filename)

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location='cpu')
        self._flow_1.load_state_dict(state_dict['flow_1'])
        self._flow_2.load_state_dict(state_dict['flow_2'])


def build_flow(batch_theta,
               batch_x,
               embedding_net,
               naive=False,
               aggregate=True,
               **kwargs):

    if naive:
        flow = JRNMMFlow_nflows_base(
            batch_theta, batch_x, embedding_net, aggregate=aggregate, **kwargs
        )
    else:
        flow = JRNMMFlow_nflows_factorized(
            batch_theta, batch_x, embedding_net, **kwargs
        )

    return flow


if __name__ == '__main__':

    from summary import summary_JRNMM

    batch_theta = torch.randn(10, 3)
    batch_x = torch.randn(10, 1024, 1)
    embedding_net = summary_JRNMM(0, 33, 1024, 'Fourier')

    flow = build_flow(batch_theta,
                      batch_x,
                      embedding_net,
                      naive=True,
                      aggregate=False)

    logp = flow.log_prob(batch_theta, batch_x)
    samples = flow.sample(100, batch_x[0])
    print(samples)
