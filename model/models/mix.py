from model.models import FewShotModelWrapper
import torch
import torch.nn.functional as F


class MixedWrapper(FewShotModelWrapper):
    def _forward(self, x, support_idx, query_idx):
        pass

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        if not self.training:
            return self.model.forward(x)
        s_logits, _ = self.model.forward(x[0])
        u_logits, _ = self.model.forward(x[1])
        label = torch.arange(self.args.way, dtype=torch.long).repeat(
            self.args.num_tasks * self.args.query  # *(self.train_loader.num_device if args.multi_gpu else 1)
        ).to(self.args.device)
        taco_loss = F.cross_entropy(u_logits, label)
        return s_logits, taco_loss
