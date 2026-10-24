"""Unchanged StudyMIL class extracted from the verified training module."""
import torch
from torch import nn
from torchvision.models import convnext_tiny

class StudyMIL(nn.Module):
    """Trainable ConvNeXt-Tiny encoder with finding-specific attention pooling."""
    def __init__(self, num_targets=12, slots=5, projection=256,
                 encoder_state=None, pretrained=False):
        super().__init__()
        if pretrained:
            raise ValueError("Use the official initialization from CPU preparation")
        self.encoder = convnext_tiny(weights=None)
        if encoder_state is not None:
            self.encoder.load_state_dict(encoder_state, strict=True)
        # Preserve torchvision's final LayerNorm and flatten before its classifier.
        self.encoder.classifier[2] = nn.Identity()
        self.project = nn.Sequential(nn.Linear(768, projection), nn.LayerNorm(projection),
                                     nn.SiLU(), nn.Dropout(0.2))
        self.slot = nn.Embedding(slots, projection)
        self.av = nn.Linear(projection, 128)
        self.au = nn.Linear(projection, 128)
        self.attention = nn.Linear(128, num_targets)
        self.finding_weight = nn.Parameter(torch.empty(num_targets, projection))
        self.finding_bias = nn.Parameter(torch.zeros(num_targets))
        nn.init.xavier_uniform_(self.finding_weight)
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, images, valid, slots, chunk=32):
        assert images.ndim == 5 and images.shape[2] == 3
        b, w, c, h, width = images.shape
        assert valid.shape == slots.shape == (b, w)
        assert valid.dtype == torch.bool and slots.dtype == torch.long
        assert chunk >= 1 and bool(valid.any(1).all()), "Empty study bag"
        x = images.reshape(b * w, c, h, width)
        x = x.float().div(255) if not x.is_floating_point() else x
        indices = valid.reshape(-1).nonzero(as_tuple=True)[0]
        x = ((x[indices] - self.mean) / self.std).contiguous(memory_format=torch.channels_last)
        encoded = torch.cat([self.encoder(part) for part in x.split(chunk)], 0)
        assert encoded.shape == (len(indices), 768)
        features = encoded.new_zeros((b * w, 768)).index_copy(0, indices, encoded)
        tokens = self.project(features.reshape(b, w, 768)) + self.slot(slots)
        scores = self.attention(torch.tanh(self.av(tokens)) * torch.sigmoid(self.au(tokens)))
        scores = scores.transpose(1, 2).float().masked_fill(~valid[:, None, :], -torch.inf)
        attention = scores.softmax(-1).to(tokens.dtype)
        pooled = torch.einsum("bcw,bwd->bcd", attention, tokens)
        return (pooled * self.finding_weight[None]).sum(-1) + self.finding_bias
