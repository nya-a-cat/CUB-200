def train_epoch(
    student_net: nn.Module,
    teacher_net: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_computer: LossComputer,
    data_handler: DataHandler,
    compression_layer: nn.Module,
    device: torch.device,
    config: TrainingConfig
) -> Dict[str, float]:
    """Single training epoch with matching logic from original code."""
    total_cls_loss = 0.0
    total_cons_loss = 0.0
    total_loss = 0.0
    
    for batch_idx, contrastive_batch in enumerate(train_loader):
        # Process batch
        batch = data_handler.process_batch(contrastive_batch, device)
        labels = batch['label']
        
        # Handle unlabeled data
        unlabeled_mask = (labels == -1)
        w = torch.ones(labels.size(0), device=device)
        
        if unlabeled_mask.any():
            # Get pseudo labels and confidence weights
            pseudo_labels, w_unlabeled = get_pseudo_labels(
                teacher_net=teacher_net,
                aug1_images=batch['aug1'][unlabeled_mask],
                aug2_images=batch['aug2'][unlabeled_mask],
                config=config
            )
            
            # Update labels and weights
            labels[unlabeled_mask] = pseudo_labels
            w[unlabeled_mask] = w_unlabeled
        
        # Classification loss
        student_logits = student_net(batch['aug1'])
        cls_loss = loss_computer.compute_classification_loss(
            student_logits=student_logits,
            labels=labels,
            confidence_weights=w
        )
        
        # Consistency loss
        Fs = get_features(student_net, batch['aug1'], config.layer_name)
        Ft = get_features(teacher_net, batch['aug2'], config.layer_name)
        Ft_compressed = compression_layer(Ft)
        
        # Apply inverse transforms
        invaug1_Fs, invaug2_Ft = data_handler.inverse_transform_features(
            Fs, Ft_compressed
        )
        
        cons_loss = loss_computer.compute_consistency_loss(
            invaug2_Ft, invaug1_Fs
        )
        
        # Total loss
        loss = cls_loss + config.consistency_weight * cons_loss
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_cls_loss += cls_loss.item()
        total_cons_loss += cons_loss.item()
        total_loss += loss.item()
        
        # Logging
        if batch_idx % 10 == 0:
            print(
                f"Batch [{batch_idx}], "
                f"ClsLoss: {cls_loss.item():.4f}, "
                f"ConsLoss: {cons_loss.item():.4f}, "
                f"w_mean: {w.mean().item():.4f}, "
                f"TotalLoss: {loss.item():.4f}"
            )
    
    return {
        'train_cls_loss': total_cls_loss / len(train_loader),
        'train_cons_loss': total_cons_loss / len(train_loader),
        'train_total_loss': total_loss / len(train_loader)
    }

def get_pseudo_labels(
    teacher_net: nn.Module,
    aug1_images: torch.Tensor,
    aug2_images: torch.Tensor,
    config: TrainingConfig
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate pseudo labels and confidence weights for unlabeled data."""
    with torch.no_grad():
        # Teacher predictions on both augmentations
        logits_t1 = teacher_net(aug1_images)
        logits_t2 = teacher_net(aug2_images)
        p1 = F.softmax(logits_t1, dim=1)
        p2 = F.softmax(logits_t2, dim=1)
        
        # Calculate confidence weights
        diff = (p1 - p2).pow(2).sum(dim=1).sqrt()
        w_unlabeled = torch.exp(-config.confidence_alpha * diff)
        
        # Generate pseudo labels
        p_avg = 0.5 * (p1 + p2)
        pseudo_labels = p_avg.argmax(dim=1)
        
        return pseudo_labels, w_unlabeled
