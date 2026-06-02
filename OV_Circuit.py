#There are 2 circuits associated with each attention_head:
#-QK circuit:WHERE to attend
#-OV circuit:WHAT to copy from the attended position

#OVfull=WU⋅WO⋅WV⋅WTE

#OVfull[i,j]: If the attention head is attending to position j, how much it boosts the 
#logit for predicting i

#For an induction head: attending to token A should boost the logit of the token 
# that follows A in the pattern.
#---------------------------------------------------------------------------------

def compute_ov_circuit(model,layer_idx,head_idx):
    """
    Compute the full OV circuit: W_U · W_O · W_V · W_E^T

    Returns: [vocab_size, vocab_size] matrix
    Entry (i, j) = "if head attends to token j,
                    how much does it boost logit for token i?"

    W_O,W_V,W_U are associated with the particular attention_head
    """
    head=model.blocks[layer_idx].attn.heads[head_idx]

    W_O=head.W_O.weight           #[ d_head, d_model]
    W_V=head.W_V.weight           #[ d_head, d_model]
    W_U=model.head.weight         #[ vocab,  d_model]
    W_E=model.token_embed.weight  #[ vocab,  d_model]

    OV=W_O @ W_V

    OV_full=W_U @ OV @ W_E.T

    return OV_full.detach().cpu().numpy()

# Compute for the best induction head
ov_matrix = compute_ov_circuit(model, 1, 1)
print(f"OV circuit matrix shape: {ov_matrix.shape}")
print(f"Analyzing Layer {best_head[0]}, Head {best_head[1]}")
print()

# For each source token, what destination gets the highest boost?
n_show = min(10, VOCAB_SIZE)
idx_to_char = train_dataset.idx_to_char

print("OV Circuit Interpretation:")
print("If head attends to 'Source' → which 'Dest' token gets boosted?")
print("-" * 55)

for src in range(n_show):
    top_dest = np.argmax(ov_matrix[:, src])
    top_score = ov_matrix[top_dest, src]
    src_char = idx_to_char[src]
    dest_char = idx_to_char[top_dest]
    print(f"  Attend to '{src_char}' → boosts '{dest_char}' "
          f"(score: {top_score:.3f})")

