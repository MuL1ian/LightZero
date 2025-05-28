#!/usr/bin/env python3
"""
Demonstration of RLHF-style pairwise preference loss

This script shows how the Bradley-Terry model from RLHF is used to train
the reward network to prefer positive examples over negative examples.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def bradley_terry_preference_loss(positive_rewards, negative_rewards):
    """
    Compute RLHF-style pairwise preference loss using Bradley-Terry model.
    
    Args:
        positive_rewards: Rewards for positive (preferred) examples
        negative_rewards: Rewards for negative (not preferred) examples
        
    Returns:
        loss: Pairwise preference loss
    """
    # Bradley-Terry model: P(positive > negative) = sigmoid(r_pos - r_neg)
    # Loss = -log(P(positive > negative)) = -log(sigmoid(r_pos - r_neg))
    # This is equivalent to: softplus(-(r_pos - r_neg))
    
    reward_diff = positive_rewards - negative_rewards
    loss = F.softplus(-reward_diff).mean()
    
    return loss, reward_diff

def demonstrate_rlhf_loss():
    """Demonstrate how the RLHF preference loss works"""
    print("RLHF-Style Pairwise Preference Loss Demonstration")
    print("=" * 55)
    
    # Create example scenarios
    scenarios = [
        ("Perfect Preference", torch.tensor([1.0, 0.8, 0.9]), torch.tensor([0.1, 0.2, 0.0])),
        ("Weak Preference", torch.tensor([0.6, 0.5, 0.7]), torch.tensor([0.4, 0.3, 0.5])),
        ("Wrong Preference", torch.tensor([0.2, 0.1, 0.3]), torch.tensor([0.8, 0.9, 0.7])),
        ("No Preference", torch.tensor([0.5, 0.5, 0.5]), torch.tensor([0.5, 0.5, 0.5])),
    ]
    
    results = []
    
    for name, pos_rewards, neg_rewards in scenarios:
        loss, reward_diff = bradley_terry_preference_loss(pos_rewards, neg_rewards)
        
        # Calculate preference accuracy (how often positive > negative)
        accuracy = (reward_diff > 0).float().mean()
        
        # Calculate average preference strength
        avg_diff = reward_diff.mean()
        
        print(f"\n{name}:")
        print(f"  Positive rewards: {pos_rewards.tolist()}")
        print(f"  Negative rewards: {neg_rewards.tolist()}")
        print(f"  Reward differences: {reward_diff.tolist()}")
        print(f"  Preference loss: {loss.item():.4f}")
        print(f"  Preference accuracy: {accuracy.item():.3f}")
        print(f"  Average preference strength: {avg_diff.item():.3f}")
        
        results.append((name, loss.item(), accuracy.item(), avg_diff.item()))
    
    # Visualize the results
    plt.figure(figsize=(12, 8))
    
    names, losses, accuracies, strengths = zip(*results)
    
    # Plot 1: Loss vs Accuracy
    plt.subplot(2, 2, 1)
    plt.scatter(accuracies, losses, s=100, alpha=0.7)
    for i, name in enumerate(names):
        plt.annotate(name, (accuracies[i], losses[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    plt.xlabel('Preference Accuracy')
    plt.ylabel('RLHF Loss')
    plt.title('Loss vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs Preference Strength
    plt.subplot(2, 2, 2)
    plt.scatter(strengths, losses, s=100, alpha=0.7, color='orange')
    for i, name in enumerate(names):
        plt.annotate(name, (strengths[i], losses[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=8)
    plt.xlabel('Average Preference Strength')
    plt.ylabel('RLHF Loss')
    plt.title('Loss vs Preference Strength')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Loss comparison
    plt.subplot(2, 2, 3)
    plt.bar(range(len(names)), losses, alpha=0.7, color='green')
    plt.xticks(range(len(names)), names, rotation=45)
    plt.ylabel('RLHF Loss')
    plt.title('Loss by Scenario')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy comparison
    plt.subplot(2, 2, 4)
    plt.bar(range(len(names)), accuracies, alpha=0.7, color='red')
    plt.xticks(range(len(names)), names, rotation=45)
    plt.ylabel('Preference Accuracy')
    plt.title('Accuracy by Scenario')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rlhf_preference_loss_demo.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved as 'rlhf_preference_loss_demo.png'")
    
    return results

def demonstrate_training_dynamics():
    """Show how the loss changes during training"""
    print("\n" + "=" * 55)
    print("Training Dynamics Demonstration")
    print("=" * 55)
    
    # Simulate training where reward network learns to prefer positive examples
    epochs = 100
    
    # Initial rewards (random)
    torch.manual_seed(42)
    positive_rewards = torch.randn(10) * 0.1 + 0.5  # Start around 0.5
    negative_rewards = torch.randn(10) * 0.1 + 0.5  # Start around 0.5
    
    losses = []
    accuracies = []
    
    for epoch in range(epochs):
        # Simulate learning: positive rewards increase, negative decrease
        positive_rewards += torch.randn(10) * 0.01 + 0.02  # Slight upward trend
        negative_rewards += torch.randn(10) * 0.01 - 0.02  # Slight downward trend
        
        # Compute loss and accuracy
        loss, reward_diff = bradley_terry_preference_loss(positive_rewards, negative_rewards)
        accuracy = (reward_diff > 0).float().mean()
        
        losses.append(loss.item())
        accuracies.append(accuracy.item())
    
    # Plot training dynamics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='RLHF Loss', color='blue', linewidth=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss')
    plt.title('RLHF Loss During Training')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Preference Accuracy', color='red', linewidth=2)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy')
    plt.title('Preference Accuracy During Training')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('rlhf_training_dynamics.png', dpi=150, bbox_inches='tight')
    print(f"📈 Training dynamics saved as 'rlhf_training_dynamics.png'")
    
    print(f"\nFinal Results:")
    print(f"  Initial loss: {losses[0]:.4f} → Final loss: {losses[-1]:.4f}")
    print(f"  Initial accuracy: {accuracies[0]:.3f} → Final accuracy: {accuracies[-1]:.3f}")

if __name__ == "__main__":
    print("🎯 RLHF Pairwise Preference Loss Demo")
    print("This demonstrates the Bradley-Terry model used in ChatGPT and other RLHF systems")
    print()
    
    # Run demonstrations
    results = demonstrate_rlhf_loss()
    demonstrate_training_dynamics()
    
    print("\n" + "=" * 55)
    print("✅ Demo completed!")
    print("\nKey Insights:")
    print("1. RLHF loss is lowest when positive examples consistently score higher")
    print("2. The loss increases when preferences are weak or reversed")
    print("3. During training, the loss decreases as the model learns preferences")
    print("4. This is the same approach used in ChatGPT and other RLHF systems")
    print("\n🔬 This same loss function is now integrated into the GAG MuZero policy!") 