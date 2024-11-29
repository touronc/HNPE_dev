import torch

# Imports for the SBI package
from pyknos.nflows.distributions import base
#from sbi.utils.get_nn_models import build_nsf
from sbi.neural_nets.net_builders.flow import build_nsf, build_maf_rqs, build_maf
from sbi.neural_nets.estimators.base import ConditionalDensityEstimator


class IdentityToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, n_extra=0):
        return x


class AggregateInstances(torch.nn.Module):
    """
    Aggregate the observations when nextra >0
    Returns a batch of type [x_0,mean(x_1,...x_N)]
    """
    def __init__(self, aggregate=True):
        super().__init__()
        self._aggregate = aggregate

    def forward(self, x):
        if self._aggregate:
            xobs = x[:, 0][:, None]  # n_batch, n_embed
            xagg = x[:, 1:].mean(dim=1)[:, None]  # n_batch, n_embed
            x = torch.cat([xobs, xagg], dim=1)  # n_batch, 2*
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
        context : torch.Tensor, shape (n_batch, n_embed+1)
            Context where the input y has been encoded, except the last entry
            which is pass thru.
        """
        # The embedding net expect an extra dimension to handle n_extra. Add it
        # in x and remove it in x_embeded
   
        # en entrée on a [x0, beta]
        x = y[:, :-1]

        x_embed = self.embedding_net(x, n_extra=0)[:, :1] # on stocke les x0

        theta = y[:, -1:]
        return torch.cat([x_embed, theta], dim=1)


class ToyModelFlow_factorized_nflows(base.Distribution):

    def __init__(self, batch_theta, batch_x, embedding_net,
                 z_score_theta="independent", z_score_x="independent"):

        super().__init__()

        # flow_1 estimates p(beta | x, x1, ..., xn) (parameter phi_1)
        # create a new net that embeds all n+1 observations and then aggregates
        # n of them via a sum operation (parameter phi_3)
        # computes the mean of the extra obs and stack it next to x

        embedding_net_1 = torch.nn.Sequential(
            embedding_net, AggregateInstances(aggregate=(batch_x.shape[2] > 1)) 
        )
        self._embedding_net_1 = embedding_net_1

        # choose whether the embedding of the context should be done inside
        # the flow object or not; this can have an impact over the z-scoring
        batch_theta_1 = batch_theta[:, -1:] #size (nsim, 1)
        batch_context_1 = batch_x.mean(dim=1) #size (nsim, nextra+1) mean over ntrials

        flow_1 = build_nsf(batch_x=batch_theta_1,
                           batch_y=batch_context_1,
                           z_score_x=z_score_theta,
                           z_score_y=z_score_x,
                           embedding_net=embedding_net_1,
                           num_transforms=5)

        self._flow_1 = flow_1 #TYPE: Nflowsflow

        # flow_2 estimates p(alpha | x, beta)
        # create a new embedding next that handles the fact of having
        # a context that is a stacking of the embedded observation x
        # and the gain parameter
        embedding_net_2 = StackContext(embedding_net)
        self._embedding_net_2 = embedding_net_2

        batch_theta_2 = batch_theta[:, :-1] #size (nsim,1)

        batch_context_2 = torch.cat(
            [batch_x[:, :, :1].mean(dim=1), batch_theta[:, -1:]],
            dim=1
        ) #size (nsim, 2) # shape (n_batch, n_times+1)
 
        flow_2 = build_nsf(batch_x=batch_theta_2,
                           batch_y=batch_context_2,
                           z_score_x=z_score_theta,
                           z_score_y=z_score_x,
                           embedding_net=embedding_net_2,
                           num_transforms=5)
        # flow_2 = build_maf_rqs(batch_x=batch_theta_2,
        #                         batch_y=batch_context_2,
        #                         z_score_x=z_score_theta,
        #                         z_score_y=z_score_x,
        #                         embedding_net=embedding_net_2,
        #                         num_transforms=5)
        # flow_2 = build_maf(batch_x=batch_theta_2,
        #                         batch_y=batch_context_2,
        #                         z_score_x=z_score_theta,
        #                         z_score_y=z_score_x,
        #                         embedding_net=embedding_net_2,
        #                         num_transforms=5)


        self._flow_2 = flow_2 #TYPE: Nflowsflow
        ###ATTENTION
        self.input_shape = batch_theta[0,:].size() #torch.zeros(2,).size() #meta_parameters["n_sr"]
        #print("input shape",self.input_shape)
        self.condition_shape = batch_x[0,:,:].size() #torch.zeros((1,1)).size()
        # A DECOMMENTER POUR LANCER UNE VERSION SEQUENTIELLE
        #self.net = self._flow_2.net

    def log_prob(self, inputs, condition):
        
        # logprob of the flow that models p(beta | x, x1, ..., xn)
        condition_1 = condition.mean(dim=1) # mean over ntrials, size (batchsize, nextra+1)
        
        theta_1 = inputs[:,:,-1:]  # gain (beta) is the last parameter
        # theta_1 = inputs[:,-1:]  # gain is the last parameter
        
        logp_1 = self._flow_1.log_prob(theta_1, condition_1)
        
        # logprob of the flow that models p(C, mu, sigma | x, gain)
        #ATTENTION changement de taille
        beta = inputs[:, :, -1:].squeeze(0)

        #beta = inputs[:, -1:]
        condition_2 = torch.cat([condition[:, :, :1].mean(dim=1), beta], dim=1)
        # ATTENTION changement de taille
        theta_2 = inputs[:,:, :-1] #alpha is the first parameter
        # theta_2 = inputs[:, :-1]
        logp_2 = self._flow_2.log_prob(theta_2, condition_2)

        return logp_1 + logp_2
    
    def loss(self, input, condition):

        return -self.log_prob(input.unsqueeze(0), condition)

    def sample(self, num_samples, condition):

        condition_1 = condition.mean(dim=1)

        # shape (n_samples, 1)
        samples_flow_1 = self._flow_1.sample(num_samples, condition_1).squeeze(2)#[0] #on garde le premier beta seulement ??

        # size (nsamples,2)
        condition_2 = torch.cat([condition[:, :, 0].mean(dim=1).repeat( 
                               num_samples, 1).unsqueeze(1),
                               samples_flow_1], dim=1)
        

        # condition batch passes through a standardizing net then embedding net
        condition_2 = self._flow_2.net._embedding_net(condition_2) #on va chercher le flow du Nflowflows puis l'embedding net
        noise = self._flow_2.net._distribution.sample(num_samples[0])
        samples_flow_2, _ = self._flow_2.net._transform.inverse(noise,
                                                          context=condition_2)
        # sample de la forme (alpha,beta)
        samples = torch.cat([samples_flow_2, samples_flow_1], dim=1).unsqueeze(1) #on ajoute une dimension pour le batch shape (en 2e position)
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


class ToyModelFlow_naive_nflows(base.Distribution):

    def __init__(self, batch_theta, batch_x, embedding_net,
                 z_score_theta="independent", z_score_x="independent", aggregate=True):

        super().__init__()

        embedding_net = torch.nn.Sequential(
            embedding_net, AggregateInstances(aggregate=aggregate)
        )
        self._embedding_net = embedding_net

        # instantiate the flow (neural spline flow)
        flow = build_nsf(batch_x=batch_theta,
                         batch_y=batch_x.mean(dim=1),
                         z_score_x=z_score_theta,
                         z_score_y=z_score_x,
                         embedding_net=embedding_net,
                         num_transforms=10)  # same capacity as factorized

        self._flow = flow
        ###ATTENTION
        self.input_shape = batch_theta[0,:].size() #torch.zeros(2,).size() #meta_parameters["n_sr"]
        #print("input shape",self.input_shape)
        self.condition_shape = batch_x[0,:,:].size() #torch.zeros((1,1)).size()


    def log_prob(self, inputs, condition):
        logp = self._flow.log_prob(inputs, condition.mean(dim=1))
        return logp

    def loss(self, input, condition):
        return -self.log_prob(input.unsqueeze(0), condition)

    def sample(self, num_samples, condition):
        samples = self._flow.sample(num_samples, condition.mean(dim=1))#[0]
        return samples

    def save_state(self, filename):
        state_dict = {}
        state_dict['flow'] = self._flow.state_dict()
        torch.save(state_dict, filename)

    def load_state(self, filename):
        state_dict = torch.load(filename, map_location='cpu')
        self._flow.load_state_dict(state_dict['flow'])


def build_flow(batch_theta, batch_x, embedding_net=torch.nn.Identity(),
               naive=False, aggregate=True):

    if naive:
        print("naive")
        flow = ToyModelFlow_naive_nflows(batch_theta,
                                         batch_x,
                                         embedding_net,
                                         aggregate=aggregate)
    else:
        print("aggregate", aggregate)
        flow = ToyModelFlow_factorized_nflows(batch_theta,
                                              batch_x,
                                              embedding_net)

    return flow
