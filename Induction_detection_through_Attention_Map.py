#Once we have the attention matrix, we will like to check induction behaviour pattern
#In our training sequences, we have used a pre-defined pattern length of repeatation(pattern_len)
#Every attention head has an associated attention matrix and we are checking which attention_head 
#is resembling such induction pattern
#att_matrix[i][j]=How much the ith token is connected with the jth token?
#So, will check how much a sequence's starting token is resembling with the previous sequence's starting token
import numpy as np
def compute_induction_score(attn,pattern_len,seq_len):
    scores=[]
    for i in range(pattern_len,seq_len):
        target=i-pattern_len
        if target>=0:
            scores.append(attn[i,target])
    return np.mean(scores) if scores else 0.0

model.eval()
test_x, test_y = train_dataset[0]
test_x_batch = test_x.unsqueeze(0).to(DEVICE)
test_seq = train_dataset.data[0]
token_labels = [train_dataset.idx_to_char[t] for t in test_seq[:-1]]

print(f"Test sequence: {train_dataset.decode(test_seq)}")
print(f"Input tokens:  {train_dataset.decode(test_seq[:-1])}")
print()

# Get attention patterns
with torch.no_grad():
    _, all_attn = model(test_x_batch, return_attention=True)

# Compute induction scores
print("Induction Scores (higher = more induction-like):")
print("-" * 55)

best_score = 0
best_head = (0, 0)
all_scores = {}
seq_len = test_x_batch.shape[1]

for layer_idx in range(N_LAYERS):
    for head_idx in range(N_HEADS):
        attn = all_attn[layer_idx][head_idx, 0].cpu().numpy()
        score = compute_induction_score(attn, PATTERN_LEN, seq_len)
        all_scores[(layer_idx, head_idx)] = score

        if score > best_score:
            best_score = score
            best_head = (layer_idx, head_idx)

        marker = " ← INDUCTION CANDIDATE" if score > 0.3 else ""
        bar = "█" * int(score * 30)
        print(f"  L{layer_idx}H{head_idx}: {score:.4f}  {bar}{marker}")

print()
print(f"★ Best induction head: Layer {best_head[0]}, Head {best_head[1]} "
      f"(score: {best_score:.4f})")

