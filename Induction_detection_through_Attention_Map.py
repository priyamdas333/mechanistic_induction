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

