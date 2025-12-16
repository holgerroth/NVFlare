import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure - simplified and more compact
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Simplified color scheme
color_input = '#E8F4F8'
color_processing = '#7FC8E8'
color_transformer = '#4FA3C5'
color_output = '#FFB84D'
color_prediction = '#FF6B6B'

# Helper function to draw boxes
def draw_box(ax, x, y, width, height, text, color, fontsize=12, fontweight='normal'):
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.15", 
                         edgecolor='black', 
                         facecolor=color, 
                         linewidth=2.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, 
           ha='center', va='center', 
           fontsize=fontsize, fontweight=fontweight,
           wrap=True)

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, linewidth=3):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', 
                           color='black', 
                           linewidth=linewidth,
                           mutation_scale=25)
    ax.add_patch(arrow)

# Title
ax.text(5, 13.2, 'Transformer Model Architecture', 
       ha='center', fontsize=20, fontweight='bold')
ax.text(5, 12.7, 'High-Level Overview', 
       ha='center', fontsize=13, style='italic', color='gray')

# 1. Input Layer
y_pos = 11.5
draw_box(ax, 1.5, y_pos, 7, 1.2, 
         'Time Series Input\n10 timesteps × 45 features\n(Signal, Location, Speed, etc.)', 
         color_input, fontsize=11, fontweight='bold')
ax.text(0.5, y_pos + 0.6, '📊', fontsize=28, ha='center', va='center')

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# 2. Embedding + Positional Encoding
y_pos = 9.5
draw_box(ax, 1.5, y_pos, 7, 1, 
         'Input Embedding\n+ Positional Encoding', 
         color_processing, fontsize=11, fontweight='bold')
ax.text(0.5, y_pos + 0.5, '🔄', fontsize=28, ha='center', va='center')

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# 3. Transformer Encoder
y_pos = 7.3
draw_box(ax, 1.5, y_pos, 7, 1.4, 
         'Transformer Encoder\n3 Layers × Multi-Head Attention\nCaptures temporal patterns', 
         color_transformer, fontsize=11, fontweight='bold')
ax.text(0.5, y_pos + 0.7, '🧠', fontsize=28, ha='center', va='center')
ax.text(9.2, y_pos + 0.7, '×3', fontsize=18, fontweight='bold', 
       bbox=dict(boxstyle='circle', facecolor='white', edgecolor='black', linewidth=2))

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# 4. Temporal Aggregation
y_pos = 5.5
draw_box(ax, 1.5, y_pos, 7, 1, 
         'Extract Last Timestep\nRepresentation', 
         color_output, fontsize=11, fontweight='bold')
ax.text(0.5, y_pos + 0.5, '📌', fontsize=28, ha='center', va='center')

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# 5. Output Head
y_pos = 3.7
draw_box(ax, 1.5, y_pos, 7, 1, 
         'Fully Connected Layers\nRegression Head', 
         color_output, fontsize=11, fontweight='bold')
ax.text(0.5, y_pos + 0.5, '⚡', fontsize=28, ha='center', va='center')

draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# 6. Prediction
y_pos = 2.2
draw_box(ax, 1.5, y_pos, 7, 0.9, 
         'Predicted Throughput (Mbps)', 
         color_prediction, fontsize=12, fontweight='bold')
ax.text(0.5, y_pos + 0.45, '🎯', fontsize=28, ha='center', va='center')

# Add key information box at bottom
info_y = 0.7
draw_box(ax, 0.5, info_y - 0.5, 9, 0.9, '', 'white', fontsize=10)
ax.text(5, info_y + 0.15, 'Key: Self-attention learns temporal dependencies to predict future throughput', 
       ha='center', fontsize=10, fontweight='bold')
ax.text(5, info_y - 0.15, 'Input: 10 past observations  →  Output: Next timestep prediction', 
       ha='center', fontsize=9, style='italic', color='#555')

plt.tight_layout()
plt.savefig('transformer_architecture.png', dpi=300, bbox_inches='tight')
print("✅ Simplified figure saved as 'transformer_architecture.png'")
plt.show()
