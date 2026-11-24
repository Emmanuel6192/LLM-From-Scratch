import tiktoken
import torch        
import torch.nn as nn
import math
from torch.utils.data import DataLoader,Dataset



#TOKENIZER
class My_tokenizer():
    def __init__ (self):
        super().__init__() 
        self.tokenizer=tiktoken.get_encoding("gpt2")
        
    def encode(self,text):
        tokenized_text=self.tokenizer.encode(text,allowed_special={"<|endoftext|>"})
        return tokenized_text
    def decode(self,tokens):
        de_tokenized_text=self.tokenizer.decode(tokens)
        return de_tokenized_text
    def token_to_tensor(self,tokens):
        tensors = torch.tensor(tokens,dtype=(torch.long))  #Very importnat step, token list must be converted to tensor
        return tensors

   

   #EMBEDDER FOR CREATING EMBEDDINGS
class embedder(nn.Module):
    def __init__(self,config):      #The config is the configuration dictionary. the embedder now requires the dictionary as an arguement
        
        super().__init__()
        self.token=nn.Embedding(config['n_tokens'],config['n_embeddings'])
        self.position=nn.Embedding(config['context_length'],config['n_embeddings'])

    def forward(self,token_ids):
        #token_ids=torch.tensor(token_ids)      #Very importnat step, token list must be converted to tensor
        position_ids = torch.arange(token_ids.shape[-1], device=token_ids.device)
        embedded_tokens= self.token(token_ids)+self.position(position_ids)
        return embedded_tokens
    
    #ATTENTION MECHANISM
class Attention(nn.Module):
    def __init__ (self,config):
        super().__init__()

        self.dropout=nn.Dropout(config['drop_rate'])
        self.key=nn.Linear(config['n_embeddings'],config['n_embeddings'])    #The input layer must be equal to the number of embeddings?? I think, find out
        self.query=nn.Linear(config['n_embeddings'],config['n_embeddings'])# The queries and keys need to have the same dimension because of the matrix multiplication between them                           
        self.values=nn.Linear(config['n_embeddings'],config['n_embeddings'])
        self.outputs=nn.Linear(config['n_embeddings'],config['n_embeddings'])
        
        self.dimensions_per_heads= (config['n_embeddings'])//(config['n_heads'])    # use // so you get an integer when you divide
        self.head_counts=config['n_heads']
        self.register_buffer('mask',torch.triu(torch.ones(config['context_length'],config['context_length'], dtype=torch.bool),diagonal=1))
    def forward (self,inputs):
        queries=self.query(inputs)
        keys=self.key(inputs)
        values=self.values(inputs)

        batch_size, token_count, _   = inputs.shape              #This is called unpacking, it should be written in this format,reversing the  order does assignment
                                                               
        #Creating Multi Head Attention
        new_queries=queries.reshape(batch_size, token_count, self.head_counts,  self.dimensions_per_heads).transpose(1,2)   
        new_keys=keys.reshape(batch_size,   token_count,    self.head_counts,   self.dimensions_per_heads).transpose(1,2)
        new_values=values.reshape(batch_size,   token_count,    self.head_counts,   self.dimensions_per_heads).transpose(1,2)

        
        
        attention_scores=new_queries@new_keys.transpose(-2,-1)          #For keys shaped [batch, tokens, dimensions], 
                                                                #this produces [batch, dimensions, tokens], keeping the batch dimension unchanged
        scaled_scores=(attention_scores)/(math.sqrt(self.dimensions_per_heads)) # normally we divide by the sqrt of the embedding dimension of the keys which would be n_embeddings * 4, but because we have 12 heads of attention, it would be 3072/12
        masked_scores=scaled_scores.masked_fill(self.mask[:token_count,:token_count],float('-inf'))
        attention_weights=torch.softmax(masked_scores,dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context_vectors=attention_weights@new_values
        
        output=context_vectors.transpose(1,2)
        combined_vectors=output.reshape(batch_size, token_count, self.head_counts*self.dimensions_per_heads)
        final_output=self.outputs(combined_vectors)   
        

        return final_output
# FEEDFORWARD MECHANISM
class Feedforward(nn.Module):
    def __init__ (self,config):
        super().__init__()
        self.expand=nn.Linear(config['n_embeddings'],config['n_embeddings']*4)
        self.activation=nn.GELU()
        self.contract=nn.Linear(config['n_embeddings']*4,config['n_embeddings'])
    def forward(self,inputs):
        Layer1=self.expand(inputs)
        Layer2=self.activation(Layer1)
        Output_Layer=self.contract(Layer2)
        return Output_Layer 

     #TRANSFORMER
class Transformer(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.attention=Attention(config)
        self.feed_forward=Feedforward(config)
        self.norm1 = nn.LayerNorm(config['n_embeddings'])
        self.norm2 = nn.LayerNorm(config['n_embeddings'])
        self.dropout=nn.Dropout(config['drop_rate'])
    def forward(self,inputs):
        Layer1=self.norm1(inputs)
        Layer2=self.attention(Layer1)
        Layer3=self.dropout(Layer2)+inputs
        Layer4=self.norm2(Layer3)
        Layer5=self.feed_forward(Layer4)
        Layer6=self.dropout(Layer5)+Layer3
        return Layer6
#The Actual Model



class MiniLLM(nn.Module):
    def __init__(self,config):
        super().__init__()
     
        self.embedder=embedder(config)
        self.dropout=nn.Dropout(config['drop_rate'])
        self.transformer=nn.Sequential(*[Transformer(config) for __ in range(config['n_trans_layers'])]) #*Unpacks the list so each transformer is passed on at a time
        self.layer_norm=nn.LayerNorm(config['n_embeddings'])
        self.outputs=nn.Linear(config['n_embeddings'],config['n_tokens'])
       

       
  
    def forward(self,token_ids):
        out_1=self.embedder(token_ids)
        out_2=self.dropout(out_1)
        out_3=self.transformer(out_2)
        out_4=self.layer_norm(out_3)
        out_5=self.outputs(out_4)
        return out_5



#To Generate Text
tokenizer=My_tokenizer()
class Generate_text(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.max_tokens = config['max_tokens']
        self.context_size = config['context_length']
        self.model = model
        self.top_k=config['top_k_for_generator']
        self.temperature = config['temperature']
        

    def forward(self, token_ids):   # added: three arguments with defaults
        self.model.eval()
    
        for i in range(self.max_tokens):
            context_tokens = token_ids[:, -self.context_size:]

            with torch.no_grad():
                logits = self.model(context_tokens)

            last_token = logits[:, -1, :]

            if self.top_k is not None:               
                top_logits, _ = torch.topk(last_token, self.top_k)
                min_val = top_logits[:, -1:]                                  # changed: -1: keeps the shape [batch, 1]
                last_token = torch.where(
                    last_token < min_val,
                    torch.tensor(float('-inf')).to(last_token.device),
                    last_token
                )
            if self.temperature > 0.0:                 
                last_token = last_token / self.temperature
                probs = torch.softmax(last_token, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:   
                next_token = torch.argmax(last_token, dim=-1, keepdim=True)
           
            token_ids = torch.cat((token_ids, next_token), dim=1)
        return token_ids
    def to_plain_text(self,generated_tokens):
        generated_text = tokenizer.decode(generated_tokens[0].tolist())
        return generated_text








#For Dataset Preparation
class My_Dataset:      #This Dataset Class takes in token IDS some may take text directly
    def __init__(self,config,token_ids):
        super().__init__()
        self.inputs=[]
        self.targets=[]
        context_length=config['context_length']
        stride=config['stride']

        for i in range(0, len(token_ids) - context_length, stride):
    
            input_chunk = token_ids[i:i + context_length]
            target_chunk = token_ids[i + 1:i + context_length + 1]
            self.inputs.append(torch.as_tensor(input_chunk, dtype=torch.long))
            self.targets.append(torch.as_tensor(target_chunk, dtype=torch.long))
  
    def __len__(self):
         return len(self.inputs)
        
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

        
        




        