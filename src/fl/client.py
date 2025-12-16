import flwr as fl
import torch
from models.encoder import SiameseEncoder
from models.projector import Projector
from losses.contrastive import contrastive_loss
from losses.prox import prox_loss
from losses.moon import moon_loss

class FLClient(fl.client.NumPyClient):
    def __init__(self, dataloader, method="fedavg", mu=0.01, lambda_moon=1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = SiameseEncoder().to(self.device)
        self.projector = Projector().to(self.device)

        self.dataloader = dataloader
        self.method = method
        self.mu = mu
        self.lambda_moon = lambda_moon

        self.global_encoder = None
        self.prev_encoder = None

        self.opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)

    def get_parameters(self, config=None):
        return [p.detach().cpu().numpy() for p in self.encoder.parameters()]

    def set_parameters(self, parameters):
        for p, w in zip(self.encoder.parameters(), parameters):
            p.data = torch.tensor(w, device=self.device)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.encoder.train()

        self.global_encoder = torch.nn.utils.deepcopy(self.encoder)

        for x1, x2, y in self.dataloader:
            x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

            z1 = self.encoder(x1)
            z2 = self.encoder(x2)

            loss = contrastive_loss(z1, z2, y)

            if self.method == "fedprox":
                loss += self.mu * prox_loss(self.encoder, self.global_encoder)

            if self.method == "moon" and self.prev_encoder is not None:
                loss += self.lambda_moon * moon_loss(
                    self.projector(z1),
                    self.projector(self.global_encoder(x1)),
                    self.projector(self.prev_encoder(x1))
                )

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

        self.prev_encoder = torch.nn.utils.deepcopy(self.encoder)
        return self.get_parameters(), len(self.dataloader), {}
