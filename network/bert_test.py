import torch
# import transformers
# from pathlib import Path
# from tokenizers import ByteLevelBPETokenizer
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F
# import tokenizers


print(F.softmax(torch.tensor([100, -4, 50, 20]).float(), dim=0))
exit()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)

text = "Hello there klabb. lol"
tokens = tokenizer.tokenize(text)
print(tokens)
exit()
input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
# print(input_ids)
with torch.no_grad():
    output = model(input_ids)
last_hidden_states = output[0]
print(last_hidden_states.shape)
exit()
hidden_states = output[2]
word_embedding = torch.stack(hidden_states[:4]).sum(0).squeeze()
n = word_embedding.shape[0]
trimmed = word_embedding.narrow(0, 1, n - 2)
argmax = trimmed.max(0)[1]
# hidden_states = torch.stack(hidden_states)

# all_word_embeddings = tokenizer.get_vocab()

# model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, is_decoder=True)
# with torch.no_grad():
#     output = model(input_ids, encoder_hidden_states=last_hidden_states)
print(word_embedding.shape)
print(trimmed.shape)
print(argmax.shape)
exit()