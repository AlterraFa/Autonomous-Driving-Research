#!/usr/bin/env python3
"""
Visualize the learning rate schedule for WSDSchedule.
"""

import matplotlib.pyplot as plt
import numpy as np


class WSDSchedule:
    def __init__(self, warmup_steps, anneal_steps, T_max, start_lr, ref_lr, final_lr=0.0):
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.anneal_steps = anneal_steps
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps - anneal_steps
        self._step = 0.0

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        elif self._step < self.T_max + self.warmup_steps:
            new_lr = self.ref_lr
        else:
            _step = self._step - (self.T_max + self.warmup_steps)
            progress = float(_step) / float(max(1, self.anneal_steps))
            new_lr = self.ref_lr + progress * (self.final_lr - self.ref_lr)
        return new_lr


def visualize_schedules(configs):
    """
    Visualize multiple LR schedules for comparison.
    
    Args:
        configs: List of dicts with keys:
            - name: str, schedule name
            - warmup: int, warmup epochs
            - anneal: int, anneal epochs
            - epochs: int, total epochs
            - ipe: int, iterations per epoch
            - start_lr: float
            - ref_lr: float
            - final_lr: float
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    for config in configs:
        warmup_steps = config['warmup'] * config['ipe']
        anneal_steps = config['anneal'] * config['ipe']
        T_max = config['epochs'] * config['ipe']
        
        scheduler = WSDSchedule(
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            T_max=T_max,
            start_lr=config['start_lr'],
            ref_lr=config['ref_lr'],
            final_lr=config['final_lr']
        )
        
        lrs = []
        steps = []
        epochs_list = []
        
        for step in range(1, T_max + 1):
            lr = scheduler.step()
            lrs.append(lr)
            steps.append(step)
            epochs_list.append(step / config['ipe'])
        
        # Plot 1: LR vs Steps
        axes[0].plot(epochs_list, lrs, linewidth=2, label=config['name'], marker='o', markersize=3, markevery=int(len(epochs_list)/50))
        
        # Plot 2: LR vs Steps (zoomed to see details)
        axes[1].plot(epochs_list, lrs, linewidth=2, label=config['name'], marker='o', markersize=3, markevery=int(len(epochs_list)/50))
        
        # Calculations for annotations
        stay_end_step = warmup_steps + (T_max - warmup_steps - anneal_steps)
        stay_end_epoch = stay_end_step / config['ipe']
        
        print(f"\n{config['name']}:")
        print(f"  Total steps: {T_max:,} ({config['epochs']} epochs × {config['ipe']} ipe)")
        print(f"  Warmup:  {warmup_steps:,} steps ({config['warmup']} epochs)")
        print(f"  Stay:    {T_max - warmup_steps - anneal_steps:,} steps ({(T_max - warmup_steps - anneal_steps)/config['ipe']:.1f} epochs)")
        print(f"  Anneal:  {anneal_steps:,} steps ({config['anneal']} epochs)")
        print(f"  LR range: {config['start_lr']:.6f} → {config['ref_lr']:.6f} → {config['final_lr']:.6f}")
    
    # Styling Plot 1
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Learning Rate', fontsize=12)
    axes[0].set_title('Learning Rate Schedule (Full View)', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=11, loc='best')
    
    # Styling Plot 2 (zoomed)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Learning Rate', fontsize=12)
    axes[1].set_title('Learning Rate Schedule (Detailed View - Decay Phase)', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=11, loc='best')
    # Set y-limits to zoom on the anneal phase
    all_lrs = []
    for config in configs:
        warmup_steps = config['warmup'] * config['ipe']
        anneal_steps = config['anneal'] * config['ipe']
        T_max = config['epochs'] * config['ipe']
        scheduler = WSDSchedule(
            warmup_steps=warmup_steps,
            anneal_steps=anneal_steps,
            T_max=T_max,
            start_lr=config['start_lr'],
            ref_lr=config['ref_lr'],
            final_lr=config['final_lr']
        )
        for _ in range(T_max):
            all_lrs.append(scheduler.step())
    
    axes[1].set_ylim(min(all_lrs) * 0.8, max(all_lrs) * 1.1)
    
    plt.tight_layout()
    plt.savefig('lr_schedule_visualization.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: lr_schedule_visualization.png")
    plt.show()


if __name__ == '__main__':
    # Your original config (problematic)
    original = {
        'name': 'Original (PROBLEMATIC)',
        'warmup': 0,
        'anneal': 3,
        'epochs': 60,
        'ipe': 200,
        'start_lr': 0.000045,
        'ref_lr': 0.000225,
        'final_lr': 0.0,
    }
    
    # Your new config (fixed)
    fixed = {
        'name': 'Fixed (NEW)',
        'warmup': 10,
        'anneal': 75,
        'epochs': 100,
        'ipe': 200,
        'start_lr': 0.000045,
        'ref_lr': 0.000225,
        'final_lr': 0.0,
    }
    
    visualize_schedules([original, fixed])
