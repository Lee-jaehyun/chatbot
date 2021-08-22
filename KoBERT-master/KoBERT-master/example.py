import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from sklearn.model_selection import train_test_split

from transformers import BertModel
from kobert.pytorch_kobert import get_pytorch_kobert_model
from kobert.utils import get_tokenizer
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup




input_ids = torch.LongTensor([[31,51,99], [15,5,0]])
