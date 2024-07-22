batch_size = 32
hidden_size = 30
# Get a batch of random numbers
batch = torch.randint(0, one_hot_Xtr.shape[1], (batch_size,))
Xb = one_hot_Xtr[:, batch, :] # (8, 32, 27)
Yb = Ytr[batch]
Yb = Yb.view(8, -1)
time_steps, batch_size, input_size = Xb.shape
print(time_steps)
print(batch_size)
print(input_size)
Xb.shape, Yb.shape


# Forward pass

loss = 0
for t in range(time_steps):
    if c0 is None: c0 = np.zeros((b,d))
    if h0 is None: h0 = np.zeros((b,d))
    Hin[0] = h0
    Cin[0] =  c0
    
    preact1[t] = Xb[t] @ Fvh + Hin[t] @ Fhh + bias1 # (32, 27) @ (27, 30) + (32, 30) @ (30, 30) + (30)
    preact2[t] = Xb[t] @ i1vh + Hin[t] @ i1hh * bias2 # (32, 27) @ (27, 30) + (32, 30) @ (30, 30) + (30)
    preact3[t] = Xb[t] @ i2vh + Hin[t] @ i2hh + bias3 # (32, 27) @ (27, 30) + (32, 30) @ (30, 30) + (30)
    preact4[t] = Xb[t] @ Ovh + Hin[t] @ Ohh + bias4 # (32, 27) @ (27, 30) + (32, 30) @ (30, 30) + (30)
    
    act1[t] = torch.sigmoid(preact1[t]) # (32, 30)
    act2[t] = torch.sigmoid(preact2[t]) # (32, 30)
    act3[t] = torch.tanh(preact3[t]) # (32, 30)
    act4[t] = torch.sigmoid(preact4[t]) # (32, 30)
    
    Cout[t] = act1[t] * Cin[t] + act2[t] * act3[t] # (32, 30)
    if t < time_steps -1: Cin[t+1] = Cout[t]
    Ctout[t] = torch.tanh(Cout[t]) # (32, 30)
    Hout[t] = Ct[t] * act4[t] # (32, 30)
    if t < time_steps -1: Hin[t+1] = Hout[t] 
    
    
    logits[t] = Hout[t] @ output_matrix # (32, 27)
    counts = logits.exp()
    counts_sum = counts.sum(1, keepdims=True)
    counts_sum_inv = counts_sum**-1 # if I use (1.0 / counts_sum) instead then I can't get backprop to be bit exact...
    probs = counts * counts_sum_inv
    logprobs = probs.log()
    loss += -logprobs[t][torch.arange(32), Yb].mean()

print (loss / time_steps)



# Learning rate
lr = 0.5

# Update parameters using gradients
Fvh -= lr * dFvh
i1vh -= lr * di1vh
i2vh -= lr * di2vh
Ovh -= lr * dOvh

Fhh -= lr * dFhh
i1hh -= lr * di1hh
i2hh -= lr * di2hh
Ohh -= lr * dOhh

bias1 -= lr * dbias1
bias2 -= lr * dbias2
bias3 -= lr * dbias3
bias4 -= lr * dbias4







for t in reversed(range(time_steps)):
    
    # Backpropogate cross entropy
    dlogits[t] = F.softmax(logits[t], 1)
    dlogits[t][torch.arange(batch_size), Yb] -= 1
    dlogits /= n
    
    # Backpropogate dHout
    # t = 7, 6, 5, 4, 3, 2, 1, 0
    if (t < time_steps-1):
        # dHout of a previous time step, must add dHin of the next time step
        # dHout of one time step, becomes the input for the next time step
        # Hout[t] = Hin[t+1]
        dHout[t] = dHin[t+1]
        dHout[t] =  dlogits[t] @ output_matrix.T + dHin[t+1] # (32, 27) @ (27, 30) = (32, 30) dHt on paper derivation
    else: # t = 7
        dHout[t] =  dlogits[t] @ output_matrix.T + dhn

    # Backpropogate output matrix
    doutput_matrix = dlogits[t].T @ dHout[t] # (32, 27)
    
    # Backpropogate dact3 (output gate activations)
    dact3[t] = dHout[t] * Ctout[t] # (32, 27) * (32, 27) = (32, 27)
    
    # Backpropogate dC (current cell state)
    dC[t] = dHout[t] * act4[t] * (1 - torch.tanh(Cout[t])**2) # (32, 27) * (32, 27) * (32, 27) = (32, 27)
    
    # Backpropogate act1 and previous cell state
    if t > 0:
        # Forget gate activations
        # Last cell activations
        dact1[t] = dC[t] * Cout[t-1] # (32, 27) * (32, 27) = (32, 27)
        dC[t-1] = dC[t] * act1[t]
    else:
        dact1[t] = dC[t] * c0
        dc0 = dC[t] * act1[t]
    
    # Backpropogate i1 activations
    dact2[t] = dC[t] * act3[t]
    
    # Backpropogate i2 activations
    dact3[t] = dC[t] * act2[t]
    
    # Backpropogate all preactivations
    dpreact1[t] = dact1[t] * act1[t] * ( 1- act1[t])
    dpreact2[t] = dact2[t] * act2[t] * ( 1- act2[t])
    dpreact3[t] = dact3[t] * (1 - torch.tanh(dpreact3[t])**2)
    dpreact4[t] = dact4[t] * act4[t] * ( 1- act4[t])
    
    # Backpropogate gates
    dFvh = Xb[t].T @ dpreact1[t] # (27, 32) (32, 30) = (27, 30)
    dFhh = Hin[t].T @ dpreact1[t] 
    di1vh = Xb[t].T @ dpreact2[t]
    di1hh = Hin[t].T @ dpreact2[t]
    di2vh = Xb[t].T @ dpreact3[t]
    di2hh = Hin[t].T @ dpreact3[t]
    dOvh = Xb[t].T @ dpreact4[t]
    dOhh = Hin[t].T @ dpreact4[t]
    
    # Backpropogate prevh
    dHin[t] = dpreact1[t] @ Fhh.T + dpreact2[t] @ i1hh.T + dpreact3[t] @ i2hh.T + dpreact4[t] @ Ohh.T
