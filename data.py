import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self,path,block_size = 1024):
        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
 
        self.encoded_data = []
        
        #self.eos_token = self.enc.encode(
        #    " ",
        #    allowed_special = {" "}
        #)[0]
 
        # read the input whole file
        with open(path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        # encode the whole text into token IDs
        full_encoded = self.enc.encode(full_text)
        
        # block_size is 1024
        for i in range(0, len(full_encoded) - self.block_size, self.block_size):
            chunk = full_encoded[i:i + self.block_size + 1] # +1 for the target token
            self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self,idx):
        chunk = self.encoded_data[idx]
        # input is the first 1024 tokens, output is the next 1024 tokens
        x = torch.tensor(chunk[:-1],dtype = torch.long)
        y = torch.tensor(chunk[1:],dtype = torch.long)
        return x,y
    
    def encode(self,text):
        return self.enc.encode(text)
    
    def decode(self,ids):
        return self.enc.decode(ids)