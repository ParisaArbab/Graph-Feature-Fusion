import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# Sample data: Replace with your 10 lists of training losses
np.random.seed(0)
epochs = list(range(1, 1001))

num_fl = [1, 2, 5, 8, 32]

training_loss = []
val_loss = []

for num in num_fl:
    path = "plots/REDDIT-BINARY/dln256/trained_mp_3/fl_"+str(num)+"/"
    loss_list = [0] * 1000
    val_loss_list = [0] * 1000
    for i in range(10):
        with open(path + "train_loss_fold_" + str(i+1) + ".txt", "r") as f:
            lines = f.readlines()
            for j in range(1000):
                loss_list[j] += math.log10(float(lines[j].strip()))
        with open(path + "/val_loss_fold_" + str(i+1) + ".txt", "r") as f:
            lines = f.readlines()
            for j in range(1000):
                val_loss_list[j] += math.log10(float(lines[j].strip()))
    loss_list = [x / 10 for x in loss_list]
    val_loss_list = [x / 10 for x in val_loss_list]
    training_loss.append(loss_list)
    val_loss.append(val_loss_list)

# Set the seaborn style to "whitegrid"
sns.set(style="whitegrid", font_scale=1.2, rc={"lines.linewidth": 1.5})

# Plot the figure
plt.figure(figsize=(10, 6))
# palette = sns.color_palette("coolwarm", 40)  # High-contrast palette
# palette = sns.color_palette("Set1", 10)  # Colorblind-friendly palette
# palette = sns.color_palette("Blues", 10)
# palette = sns.color_palette("Dark2", 10)
palette = [
    '#FF00FF', '#00FFFF', 
    '#FFFF00', 
    '#0000FF', 
    '#00FF00', 
    # '#FF0000',
    # "#FFA500",
    # "#FF4500",
    "#ADFF2F", 
    "#7FFF00",
    "#00BFFF",    
    "#1E90FF",
    "#FF69B4", 
    "#FF1493", 
    "#DA70D6", 
    "#BA55D3", 
    "#32CD32", 
    "#FFD700" 
]

for idx, loss in enumerate(training_loss):
    sns.lineplot(x=epochs, y=loss, label=f'train_fl={num_fl[idx]}', color=palette[idx])
    v_loss = val_loss[idx]
    sns.lineplot(x=epochs, y=v_loss, label=f'test_fl={num_fl[idx]}', color=palette[idx], linestyle='--')

# Labeling and aesthetics
plt.title('Loss Across Epochs for Our Model with Trained MP', fontsize=16, weight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Loss (log scale)', fontsize=14)
plt.ylim(-0.7, 1.5)
plt.legend(title='Models', fontsize=10, title_fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('loss_trained_n_fl.pdf', format='pdf', bbox_inches='tight')
plt.show()