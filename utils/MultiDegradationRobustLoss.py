import torch
import torch.nn.functional as F

class MultiDegradationRobustLoss(torch.nn.Module):
  def __init__(self, batch_size, n_views, device, num_gpu, temperature=0.07):
    super(MultiDegradationRobustLoss, self).__init__()
    self.batch_size = batch_size
    self.n_views = n_views
    self.temperature = temperature
    self.device = device
    self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
    self.num_gpu = num_gpu

  def forward(self, features):
    logits, labels = self.info_nce_loss(features)
    return self.criterion(logits, labels)

  def info_nce_loss(self, features):
    labels = torch.cat([torch.arange(self.batch_size//self.num_gpu) for i in range(self.n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(self.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

    logits = logits / self.temperature
    return logits, labels