#OV Circuits: All Heads Comparison

fig, axes = plt.subplots(N_LAYERS, N_HEADS,
                         figsize=(4.5 * N_HEADS, 4.5 * N_LAYERS))
fig.suptitle('OV Circuits — All Heads\n'
             '(What computation does each head perform?)',
             fontsize=14, fontweight='bold', y=1.02)

for li in range(N_LAYERS):
    for hi in range(N_HEADS):
        ax = axes[li][hi]
        ov = compute_ov_circuit(model, li, hi)

        im = ax.imshow(ov[:n_show, :n_show], cmap='RdBu_r', aspect='auto')
        title = f'L{li}H{hi}'
        if (li, hi) == best_head:
            title += ' ★ INDUCTION'
            ax.set_title(title, fontsize=11, fontweight='bold',
                        color='#c0392b')
        else:
            ax.set_title(title, fontsize=11)

        ax.set_xticks(range(n_show))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks(range(n_show))
        ax.set_yticklabels(labels, fontsize=8)

plt.tight_layout()
plt.show()