# Ablation Functions
import numpy as np
import torch
import torch.nn as nn

def evaluate_model(model, dataset, device=DEVICE, n_samples=200):
    """
    Evaluate model performance.
    Returns: (avg_loss, accuracy%, perplexity)
    """
    crit = nn.CrossEntropyLoss()
    total_loss, total_correct, total_tokens = 0, 0, 0

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            x, y = dataset[i]
            x = x.unsqueeze(0).to(device)
            y = y.unsqueeze(0).to(device)

            logits = model(x)
            loss = crit(logits.reshape(-1, model.vocab_size),
                        y.reshape(-1))
            total_loss += loss.item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / min(n_samples, len(dataset))
    accuracy = total_correct / total_tokens * 100
    perplexity = np.exp(avg_loss)
    return avg_loss, accuracy, perplexity


def ablate_head(model, layer_idx, head_idx):
    """
    Create a copy of the model with one head zeroed out.
    This removes the head's contribution completely.
    """
    ablated = copy.deepcopy(model)
    head = ablated.blocks[layer_idx].attn.heads[head_idx]
    for param in head.parameters():
        param.data.zero_()
    return ablated

print("✓ Ablation functions defined")
#-------------------------------------------------------------------------------------


# Create evaluation dataset (different seed from training)
eval_dataset = RepeatedPatternDataset(
    num_samples=200,
    pattern_len=PATTERN_LEN,
    num_repeats=NUM_REPEATS,
    vocab_size=VOCAB_SIZE,
    seed=999  # Different from training seed
)

# Baseline (all heads active)
model.eval()
base_loss, base_acc, base_ppl = evaluate_model(model, eval_dataset)
print("Baseline (all heads active):")
print(f"  Loss: {base_loss:.4f}  |  Accuracy: {base_acc:.1f}%  |  "
      f"Perplexity: {base_ppl:.3f}")
print()

# Ablate each head one at a time
print("Ablation results (one head removed at a time):")
print(f"{'Head':>8}  {'Loss':>8}  {'Accuracy':>10}  {'Perplexity':>12}  "
      f"{'Δ Loss':>8}  {'Note'}")
print("─" * 72)

ablation_results = {}

for li in range(N_LAYERS):
    for hi in range(N_HEADS):
        ablated = ablate_head(model, li, hi)
        loss, acc, ppl = evaluate_model(ablated, eval_dataset)
        delta = loss - base_loss
        is_best = (li, hi) == best_head

        ablation_results[(li, hi)] = {
            'loss': loss, 'accuracy': acc,
            'perplexity': ppl, 'delta_loss': delta,
            'is_induction': is_best
        }

        note = "★ INDUCTION HEAD" if is_best else \
               ("← important" if delta > 0.5 else "")
        print(f"  L{li}H{hi}   {loss:>8.4f}  {acc:>9.1f}%  "
              f"{ppl:>12.3f}  {delta:>+8.4f}  {note}")

# Summary
ind_delta = ablation_results[best_head]['delta_loss']
other_deltas = [r['delta_loss'] for k, r in ablation_results.items()
                if k != best_head]
avg_other = np.mean(other_deltas)

print()
print("CAUSAL EVIDENCE:")
print(f"  Ablating induction head (L{best_head[0]}H{best_head[1]}): "
      f"Δ loss = {ind_delta:+.4f}")
print(f"  Ablating other heads (average):         "
      f"Δ loss = {avg_other:+.4f}")
if avg_other > 0.001:
    print(f"  Ratio: {ind_delta / avg_other:.1f}x more impact")
print()
if ind_delta > 2 * max(avg_other, 0.001):
    print("  → STRONG CAUSAL EVIDENCE: The induction head is specifically")
    print("    responsible for pattern completion.")
else:
    print("  → Induction may be distributed across multiple heads.")