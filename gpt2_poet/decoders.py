import torch
import numpy as np

def choose_from_top(probs, n=5):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def discretize(y_soft):
    """
    Discretizes a soft categorical signal while retaining the gradient
    """
    shape = y_soft.size()
    k = y_soft.data.max(-1)[1]
    k = k.unsqueeze(-1)
    y_hard = y_soft.data.new(*shape).zero_().scatter_(-1, k, 1.0)
    y_hard = y_hard.view(shape)

    return y_hard - y_soft.detach() + y_soft

def decode(tokenizer, device, cur_ids, outputs, idx, greedy):
    loss, logits = outputs[:2]
    softmax_logits = torch.softmax(logits[0,-1], dim=0) #Take the first(from only one in this case) batch and the last predicted embedding
    if idx < 3:
        n = 20
    else:
        n = 3

    if (greedy):
        next_token_id = choose_greedy_first(softmax_logits.to('cpu').numpy(), n=n) #Greedy BFS
    else:
        next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n) #Randomly(from the topN probability distribution) select the next word

    cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim = 1) #Add the last word to the running sequence
    if next_token_id in tokenizer.encode('<|endoftext|>'):
        poem_finished = True
    else:
        poem_finished = False
    return cur_ids, poem_finished

def choose_greedy_first(probs, n=5): #literally the same thing over and over? confuse
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    choice = np.where(top_prob == np.amax(top_prob))[0] # Get index of highest probability
    token_id = ind[choice][0]
    return int(token_id)
    return torch.argmax(probs, dim=-1)
