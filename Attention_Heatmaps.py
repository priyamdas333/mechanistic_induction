#@title 4.2 — Attention Heatmaps (All 8 Heads)

fig, axes = plt.subplots(N_LAYERS, N_HEADS,
                         figsize=(4.5 * N_HEADS, 4.5 * N_LAYERS))
fig.suptitle('Attention Patterns — All Heads\n'
             '(Row = query position, Column = key position)',
             fontsize=15, fontweight='bold', y=1.02)

for li in range(N_LAYERS):
    for hi in range(N_HEADS):
        ax = axes[li][hi]
        attn = all_attn[li][hi, 0].cpu().numpy()

        im = ax.imshow(attn, cmap='Blues', vmin=0, vmax=1, aspect='auto')

        title = f'Layer {li}, Head {hi}'
        if (li, hi) == best_head:
            title += ' ★ INDUCTION'
            ax.set_title(title, fontsize=11, fontweight='bold',
                        color='#c0392b')
        else:
            ax.set_title(title, fontsize=11)

        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, fontsize=7, rotation=45)
        ax.set_yticks(range(len(token_labels)))
        ax.set_yticklabels(token_labels, fontsize=7)

        if hi == 0:
            ax.set_ylabel('Query (from)', fontsize=9)
        if li == N_LAYERS - 1:
            ax.set_xlabel('Key (to)', fontsize=9)

plt.colorbar(im, ax=axes, shrink=0.6, label='Attention Weight')
plt.tight_layout()
plt.show()