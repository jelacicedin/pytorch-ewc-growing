from functools import reduce
import torch
from torch import nn
from torch.nn import functional as F
from torch import autograd
from torch.autograd import Variable
import utils


class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=400,
        hidden_layer_num=2,
        hidden_dropout_prob=0.5,
        input_dropout_prob=0.2,
        lamda=40,
    ):
        # Configurations.
        super().__init__()
        self.input_size = input_size
        self.input_dropout_prob = input_dropout_prob
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.hidden_dropout_prob = hidden_dropout_prob
        self.output_size = output_size
        self.lamda = lamda

        # Layers.
        # self.layers = nn.ModuleList(
        #     [
        #         # input
        #         nn.Linear(self.input_size, self.hidden_size),
        #         nn.ReLU(),
        #         nn.Dropout(self.input_dropout_prob),
        #         # hidden
        #         *(
        #             (
        #                 nn.Linear(self.hidden_size, self.hidden_size),
        #                 nn.ReLU(),
        #                 nn.Dropout(self.hidden_dropout_prob),
        #             )
        #             * self.hidden_layer_num
        #         ),
        #         # output
        #         nn.Linear(self.hidden_size, self.output_size),
        #     ]
        # )

        self.linear_input = nn.Linear(self.input_size, self.hidden_size)
        self.relu_input = nn.ReLU()
        self.dropout_input = nn.Dropout(self.input_dropout_prob)
        self.linear_hidden = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu_output = nn.ReLU()
        self.dropout_output = nn.Dropout(self.hidden_dropout_prob)
        self.linear_output = nn.Linear(self.hidden_size, self.output_size)

    @property
    def name(self):
        return (
            "MLP"
            "-lamda{lamda}"
            "-in{input_size}-out{output_size}"
            "-h{hidden_size}x{hidden_layer_num}"
            "-dropout_in{input_dropout_prob}_hidden{hidden_dropout_prob}"
        ).format(
            lamda=self.lamda,
            input_size=self.input_size,
            output_size=self.output_size,
            hidden_size=self.hidden_size,
            hidden_layer_num=self.hidden_layer_num,
            input_dropout_prob=self.input_dropout_prob,
            hidden_dropout_prob=self.hidden_dropout_prob,
        )

    def forward(self, x):
        print(x.size())
        x = self.linear_input(x)
        x = self.relu_input(x)
        x = self.dropout_input(x)
        x = self.linear_hidden(x)
        x = self.relu_output(x)
        x = self.dropout_output(x)
        x = self.linear_output(x)
        return x

    def estimate_fisher(self, dataset, data_loader, sample_size, batch_size=32):
        # sample loglikelihoods from the dataset.
        loglikelihoods = []
        data_loader = torch.utils.data.DataLoader(
            dataset,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=4,
            shuffle=True,
        )
        for x, y in data_loader:
            preds = self(x.cuda()).squeeze()

            x = Variable(x).cuda() if self._is_on_cuda() else Variable(x)
            # y = Variable(y).cuda() if self._is_on_cuda() else Variable(y)
            print (f"Actual batch size: {x.size(0)}")
            loglikelihoods.append(
                F.log_softmax(self(x), dim=0)
            )
            if len(loglikelihoods) >= sample_size // batch_size:
                break
        # estimate the fisher information of the parameters.
        loglikelihoods = torch.cat(loglikelihoods).unbind()
        loglikelihood_grads = zip(
            *[
                autograd.grad(
                    l, self.parameters(), retain_graph=(i < len(loglikelihoods))
                )
                for i, l in enumerate(loglikelihoods, 1)
            ]
        )
        loglikelihood_grads = [torch.stack(gs) for gs in loglikelihood_grads]
        fisher_diagonals = [(g**2).mean(0) for g in loglikelihood_grads]
        param_names = [n.replace(".", "__") for n, p in self.named_parameters()]
        return {n: f.detach() for n, f in zip(param_names, fisher_diagonals)}

    def consolidate(self, fisher):
        for n, p in self.named_parameters():
            n = n.replace(".", "__")
            self.register_buffer("{}_mean".format(n), p.data.clone())
            self.register_buffer("{}_fisher".format(n), fisher[n].data.clone())

    def ewc_loss(self, cuda=False):
        try:
            losses = []
            for n, p in self.named_parameters():
                # retrieve the consolidated mean and fisher information.
                n = n.replace(".", "__")
                mean = getattr(self, "{}_mean".format(n))
                fisher = getattr(self, "{}_fisher".format(n))
                # wrap mean and fisher in variables.
                mean = Variable(mean)
                fisher = Variable(fisher)
                # calculate a ewc loss. (assumes the parameter's prior as
                # gaussian distribution with the estimated mean and the
                # estimated cramer-rao lower bound variance, which is
                # equivalent to the inverse of fisher information)
                losses.append((fisher * (p - mean) ** 2).sum())
            return (self.lamda / 2) * sum(losses)
        except AttributeError:
            # ewc loss is 0 if there's no consolidated parameters.
            return Variable(torch.zeros(1)).cuda() if cuda else Variable(torch.zeros(1))

    def _is_on_cuda(self):
        return next(self.parameters()).is_cuda
