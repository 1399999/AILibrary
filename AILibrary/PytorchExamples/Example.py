import torch
import torch.nn.functional as F

## read in all the words
words = open('C:\\AITrainingSets\\Names.txt', 'r').read().splitlines()
# print(words[:8]) # ['emma', 'olivia', 'ava', 'isabella', 'sophia', 'charlotte', 'mia', 'amelia']
# print(len(words)) # 32033

## build the vocabulary of characters and mappings to/from integers
chars = sorted(list(set(''.join(words)))) 
# print(chars) # ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
stoi = {s:i+1 for i,s in enumerate(chars)}
# print(stoi) # {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
# print(itos) # {1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z', 0: '.'}

## build the dataset

# build the dataset
block_size = 3 # context length: how many characters do we take to predict the next one?

def build_dataset(words):  
  X, Y = [], []
  for w in words:
    #print(w)
    context = [0] * block_size
    for ch in w + '.':
      ix = stoi[ch]
      X.append(context)
      Y.append(ix)
      #print(''.join(itos[i] for i in context), '--->', itos[ix])
      context = context[1:] + [ix] # crop and append

  # print(X[:8]) # [[0, 0, 0], [0, 0, 5], [0, 5, 13], [5, 13, 13], [13, 13, 1], [0, 0, 0], [0, 0, 15], [0, 15, 12]]
  # print(Y[:8]) # [5, 13, 13, 1, 0, 15, 12, 9]

  X = torch.tensor(X)
  Y = torch.tensor(Y)

  # print(X.shape, Y.shape) # torch.Size([228146, 3]) torch.Size([228146])
  return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# print(Xtr[:8]) # tensor([[ 0,  0,  0], [ 0,  0, 25], [ 0, 25, 21], [25, 21,  8], [21,  8,  5], [ 8,  5, 14], [ 5, 14,  7], [ 0,  0,  0]])
# print(Ytr[:8]) # tensor([25, 21,  8,  5, 14,  7,  0,  4])
# print(Xdev[:8]) # tensor([[ 0,  0,  0], [ 0,  0,  1], [ 0,  1, 13], [ 1, 13,  1], [13,  1, 25], [ 0,  0,  0], [ 0,  0,  1], [ 0,  1, 25]])
# print(Ydev[:8]) # tensor([ 1, 13,  1, 25,  0,  1, 25, 20])
# print(Xte[:8]) # tensor([[ 0,  0,  0], [ 0,  0, 13], [ 0, 13, 21], [13, 21, 19], [21, 19, 20], [19, 20,  1], [20,  1,  6], [ 1,  6,  1]])
# print(Yte[:8]) # tensor([13, 21, 19, 20,  1,  6,  1,  0])

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C = torch.randn((27, 10), generator=g)
W1 = torch.randn((30, 200), generator=g)
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 27), generator=g)
b2 = torch.randn(27, generator=g)
parameters = [C, W1, b1, W2, b2]

# print(sum(p.nelement() for p in parameters)) # number of parameters in total # 11897

for p in parameters:
  p.requires_grad = True

lre = torch.linspace(-3, 0, 1000)

# print(lre[:8]) # tensor([-3.0000, -2.9970, -2.9940, -2.9910, -2.9880, -2.9850, -2.9820, -2.9790])

lrs = 10**lre
