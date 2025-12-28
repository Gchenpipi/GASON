import torch
import numpy as np
from transformers import AutoTokenizer, T5Model
import pandas as pd
from tqdm import tqdm, trange
from openTSNE import TSNE
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MODEL_NAME = "/ai/dataset/qsedata/GRACE/Pretrained_Models/codet5" # Local model file

POOLING = 'first_last_avg'
# POOLING = 'last_avg'
# POOLING = 'last2avg'

USE_tsne = False
MAX_LENGTH = 512

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model(name):
    tokenizer = AutoTokenizer.from_pretrained(name)  # 替换 T5Tokenizer
    model = T5Model.from_pretrained(name)
    model = model.to(DEVICE)
    return tokenizer, model


def sents_to_vecs(sents, tokenizer, model):
    vecs = []
    with torch.no_grad():
        for sent in tqdm(sents):
        # for sent in tqdm(sents):
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True,  max_length=MAX_LENGTH)
            inputs['input_ids'] = inputs['input_ids'].to(DEVICE)
            inputs['attention_mask'] = inputs['attention_mask'].to(DEVICE)

            decoder_input_ids = torch.full(
                (1, 1),  # 形状: (batch_size, sequence_length)
                tokenizer.pad_token_id,  # T5使用pad_token_id作为起始符
                device=DEVICE
            )

            hidden_states = model(**inputs,decoder_input_ids=decoder_input_ids, return_dict=True, output_hidden_states=True).encoder_hidden_states

            if POOLING == 'first_last_avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[1]).mean(dim=1)
            elif POOLING == 'last_avg':
                output_hidden_state = (hidden_states[-1]).mean(dim=1)
            elif POOLING == 'last2avg':
                output_hidden_state = (hidden_states[-1] + hidden_states[-2]).mean(dim=1)
            else:
                raise Exception("unknown pooling {}".format(POOLING))
            # output_hidden_state [batch_size, hidden_size]
            vec = output_hidden_state.cpu().numpy()[0]
            vecs.append(vec)
    assert len(sents) == len(vecs)
    vecs = np.array(vecs)
    return vecs



def transform_and_normalize(vecs, embedding_train,flag):
    """应用变换，然后标准化
    """
    if flag==0:
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5
    elif flag==1:
        embedding_test = embedding_train.transform(vecs)
        vecs = np.array(embedding_test)
        return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5



def normalize(vecs):
    """标准化
    """
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def main():
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric="euclidean",
        n_jobs=-1,
        random_state=42,
        verbose=True
    )


    print(f"Configs: {MODEL_NAME}-{POOLING}-{USE_tsne}-{MAX_LENGTH}.")
    tokenizer, model = build_model(MODEL_NAME)
    print("Building {} tokenizer and model successfuly.".format(MODEL_NAME))
    df = pd.read_csv("/ai/dataset/qsedata/GRACE/GRACE-main/dataset/Reveal/adddata_to_dataset/new_train_set.csv")
    code_list = df['Code'].tolist()
    print(len(code_list))
    print("Transfer sentences to t5 vectors.")
    vecs_func_body = sents_to_vecs(code_list, tokenizer, model) # [code_list_size, 768]
    if USE_tsne:
        print("Compute train embedding.")
        # kernel, bias = compute_kernel_bias([
        #     vecs_func_body
        # ], n_components=N_COMPONENTS)
        embedding_train = tsne.fit(vecs_func_body)
        vecs_func_body = np.array(embedding_train)
        vecs_func_body = transform_and_normalize(vecs_func_body, embedding_train, 0) # [code_list_size, 2]
    else:
        vecs_func_body = normalize(vecs_func_body)# [code_list_size, 768]
    print(vecs_func_body.shape)
    import pickle
    f = open('./model/Reveal/code_vector_wotsne_512.pkl', 'wb')
    pickle.dump(vecs_func_body, f)
    f.close()
    # f = open('./model/Reveal/embedding_train.pkl', 'wb')
    # pickle.dump(embedding_train, f)
    # f.close()
    # f = open('./model/kernel.pkl', 'wb')
    # pickle.dump(kernel, f)
    # f.close()
    # f = open('./model/bias.pkl', 'wb')
    # pickle.dump(bias, f)
    # f.close()

if __name__ == "__main__":

    main()