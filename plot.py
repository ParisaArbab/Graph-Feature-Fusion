from matplotlib import pyplot as plt
import os
import numpy as np

width = []
gcn_acc = []
gcn_std = []
our_acc = []
our_std = []

for i in range(1, 81):
    # if i > 10 and i % 5 != 0:
    #     continue
    path1 = "plots/REDDIT-BINARY/gcn"+str(i)
    if os.path.exists(path1):
        width.append(i)
        acc_list = []
        for j in range(1, 11):
            
            with open(path1 + "/val_acc_fold_" + str(j) + ".txt", "r") as f:
                accs = f.readlines()
                accs = [float(x.strip()) for x in accs]
                acc_list.append(max(accs))
        print(acc_list)
        gcn_acc.append(np.mean(acc_list))
        gcn_std.append(np.std(acc_list))
    path2 = "plots/REDDIT-BINARY/dln"+str(i)
    if os.path.exists(path2):
        acc_list = []
        for j in range(1, 11):
            with open(path2 + "/val_acc_fold_" + str(j) + ".txt", "r") as f:
                accs = f.readlines()
                accs = [float(x.strip()) for x in accs]
                acc_list.append(max(accs))
        our_acc.append(np.mean(acc_list))
        our_std.append(np.std(acc_list))

print(f"number of data points: {len(width)}")
print(f"Standard deviation of GCN: {gcn_std}")
print(f"Standard deviation of Ours: {our_std}")
# plt.figure(figsize=(16, 12))
fig, ax_left = plt.subplots(figsize=(18, 12))
ax_right = ax_left.twinx()
ax_left.plot(width, gcn_acc, label="GCN", color='blue')
ax_left.plot(width, our_acc, label="Ours", color='red')
ax_right.plot(width, gcn_std, label="GCN std", color='blue', linestyle='--')
ax_right.plot(width, our_std, label="Ours std", color='red', linestyle='--')
ax_left.set_xlabel("Width", fontsize=40)
ax_left.set_ylabel("Accuracy", fontsize=40)
ax_right.set_ylabel("Std", fontsize=40)
# ax_left.tick_params(axis='y', labelsize=30)
# ax_right.tick_params(axis='y', labelsize=30)
# ax_left.tick_params(axis='x', labelsize=30)
# ax_left.set_xticks(np.arange(0, 61, 5))
# ax_left.set_xticklabels(np.arange(0, 61, 5), fontsize=30)
# ax_left.set_yticks(np.arange(0, 1.1, 0.1))
# ax_left.set_yticklabels(np.arange(0, 1.1, 0.1), fontsize=30)
# ax_right.set_yticks(np.arange(0, 0.6, 0.1))
# ax_right.set_yticklabels(np.arange(0, 0.6, 0.1), fontsize=30)
ax_left.set_title("REDDIT-BINARY", fontsize=40)
ax_left.legend(loc='upper left', fontsize=30)
ax_right.legend(loc='upper right', fontsize=30)
# Save tyo high resolution pdf
fig.savefig("plots/REDDIT-BINARY/width.pdf", bbox_inches='tight', dpi=300)

