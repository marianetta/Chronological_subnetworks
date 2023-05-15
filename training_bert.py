#!/usr/bin/env python
# coding: utf-8

# ## dataset

# In[2]:


get_ipython().system('pip install transformers')
get_ipython().system('sudo apt-get install git-lfs')


# In[1]:


get_ipython().system('pip install datasets')


# In[1]:


from datasets import concatenate_datasets, load_dataset

bookcorpus = load_dataset("bookcorpus", split="train")
#wiki = load_dataset("wikipedia", "20220301.en", split="train")
#wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

#assert bookcorpus.features.type == wiki.features.type
#raw_datasets = concatenate_datasets([bookcorpus,wiki])


# In[2]:


get_ipython().system('pip install apache_beam')


# In[3]:


import apache_beam


# In[4]:


wiki = load_dataset("wikipedia", "20220301.en", split="train")
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])  # only keep the 'text' column

assert bookcorpus.features.type == wiki.features.type
raw_datasets = concatenate_datasets([bookcorpus, wiki])


# In[4]:


#raw_datasets[0]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## tokenizer

# In[4]:


from tqdm import tqdm
from transformers import BertTokenizerFast

# repositor id for saving the tokenizer
tokenizer_id="bert-base-uncased-2022-habana"

# create a python generator to dynamically load the data
def batch_iterator(batch_size=10000):
    for i in tqdm(range(0, len(raw_datasets), batch_size)):
        yield raw_datasets[i : i + batch_size]["text"]

# create a tokenizer from existing one to re-use special tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


# In[ ]:


bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
#bert_tokenizer.save_pretrained("tokenizer")


# In[ ]:


bert_tokenizer.save_pretrained("tokenizer")


# ## preprocess

# In[5]:


from transformers import AutoTokenizer
import multiprocessing

# load tokenizer
# tokenizer = AutoTokenizer.from_pr+etrained(f"{user_id}/{tokenizer_id}")
tokenizer = AutoTokenizer.from_pretrained("tokenizer")
num_proc = multiprocessing.cpu_count()
print(f"The max length for the tokenizer is: {tokenizer.model_max_length}")

def group_texts(examples):
    tokenized_inputs = tokenizer(
       examples["text"], return_special_tokens_mask=True, truncation=True, max_length=tokenizer.model_max_length
    )
    return tokenized_inputs

# preprocess dataset
tokenized_datasets = raw_datasets.map(group_texts, batched=True, remove_columns=["text"], num_proc=num_proc, load_from_cache_file = True)
tokenized_datasets.features


# In[6]:


from itertools import chain

# Main data processing function that will concatenate all texts from our dataset and generate chunks of
# max_seq_length.
def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= tokenizer.model_max_length:
        total_length = (total_length // tokenizer.model_max_length) * tokenizer.model_max_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + tokenizer.model_max_length] for i in range(0, total_length, tokenizer.model_max_length)]
        for k, t in concatenated_examples.items()
    }
    return result

tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=num_proc, load_from_cache_file = True)
# shuffle dataset
tokenized_datasets = tokenized_datasets.shuffle(seed=34)

print(f"the dataset contains in total {len(tokenized_datasets)*tokenizer.model_max_length} tokens")
# the dataset contains in total 3417216000 tokens


# In[7]:


#tokenized_datasets


# In[8]:


from transformers import DataCollatorForLanguageModeling
collator = DataCollatorForLanguageModeling(tokenizer = tokenizer)


# ## pre-train

# In[8]:


get_ipython().system('pip install torch')


# In[9]:


import torch
import tqdm


# In[10]:


from transformers import Trainer
from transformers import TrainingArguments


# In[11]:


from transformers import BertConfig


# In[12]:


# bert tiny config
config = BertConfig(
    hidden_size = 128,
    hidden_act = 'gelu',
    initializer_range = 0.02,
    vocab_size=32000,
    hidden_dropout_prob = 0.1,
    num_attention_heads = 2,
    type_vocab_size = 2,
    max_position_embeddings = 512,
    num_hidden_layers = 2,
    intermediate_size = 512,
    attention_probs_dropout_prob = 0.1
)


# In[11]:


#{"hidden_size": 512, "hidden_act": "gelu", "initializer_range": 0.02, "vocab_size": 30522, "hidden_dropout_prob": 0.1, "num_attention_heads": 8, "type_vocab_size": 2, "max_position_embeddings": 512, "num_hidden_layers": 4, "intermediate_size": 2048, "attention_probs_dropout_prob": 0.1}


# In[12]:


#{"hidden_size": 256, "hidden_act": "gelu", "initializer_range": 0.02, "vocab_size": 30522, "hidden_dropout_prob": 0.1, "num_attention_heads": 4, "type_vocab_size": 2, "max_position_embeddings": 512, "num_hidden_layers": 4, "intermediate_size": 1024, "attention_probs_dropout_prob": 0.1}


# In[13]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# and move our model over to the selected device


# In[14]:


torch.cuda.is_available()


# In[15]:


from transformers import BertForMaskedLM
model = BertForMaskedLM(config)


# In[16]:


model.to(device)


# In[ ]:





# In[17]:


training_args = TrainingArguments(
    output_dir = 'checkpoints',
    per_device_train_batch_size = 32,
    learning_rate = 5e-5,
    max_steps = 1000000,
    save_strategy = 'steps',
    save_steps = 100000, 
    seed = 34,
    data_seed = 34
)

trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = tokenized_datasets, 
    tokenizer = tokenizer,
    data_collator = collator
)


# In[ ]:


trainer.train()


# In[ ]:




