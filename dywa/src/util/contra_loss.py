import torch
import torch.nn.functional as F

def contrastive_loss(x1, x2, y, margin=1.0):
    """
    x1, x2: embedding vectors (Tensor), shape (batch_size, embedding_dim)
    y: labels (Tensor), shape (batch_size,)
    margin: margin parameter, minimum distance for negative pairs
    """
    # Compute Euclidean distance
    euclidean_distance = F.pairwise_distance(x1, x2, p=2)
    
    # Compute contrastive loss
    loss = torch.mean((y * torch.pow(euclidean_distance, 2)) + 
                      ((1 - y) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)))
    return loss

def triplet_loss(anchor, positive, negative, margin=1.0):
    """
    Compute Triplet Loss

    Args:
    - anchor: anchor samples (Tensor), shape (batch_size, embedding_dim)
    - positive: positive samples (Tensor), shape (batch_size, embedding_dim)
    - negative: negative samples (Tensor), shape (batch_size, embedding_dim)
    - margin: margin hyperparameter, ensures negative pairs are at least this much farther than positive pairs

    Returns:
    - loss: scalar value of Triplet Loss
    """
    # Compute Euclidean distance between anchor and positive
    positive_distance = F.pairwise_distance(anchor, positive, p=2)
    # Compute Euclidean distance between anchor and negative
    negative_distance = F.pairwise_distance(anchor, negative, p=2)
    # Compute Triplet Loss
    loss = torch.mean(torch.relu(positive_distance - negative_distance + margin))
    return loss