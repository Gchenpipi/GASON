# coding:utf-8
import pickle
import faiss
import torch
import numpy as np
import Levenshtein
from tqdm import tqdm
from transformers import AutoTokenizer, T5Model
import pandas as pd
import csv
import json
import os
from t5_tsne import sents_to_vecs, normalize

dim = 768

ALPHA_LIST = [round(x * 0.1, 1) for x in range(11)]  # 0.0 ~ 1.0
TOPK = 5   # 初筛候选数量

def l2_normalize(vec):
    """L2 归一化"""
    return vec / (np.linalg.norm(vec) + 1e-8)

df = pd.read_csv("/ai/dataset/qsedata/GRACE/GRACE-main/dataset/Reveal/adddata_to_dataset/new_train_set.csv")
train_code_list = df['Code'].tolist()
train_ast_list = df['Sim_SBT'].tolist()
train_label_list = df['label'].tolist()

train_cpg_list = [l2_normalize(np.array(json.loads(x), dtype=np.float32)) for x in df['embedding_cpg'].tolist()]

df = pd.read_csv("/ai/dataset/qsedata/GRACE/GRACE-main/dataset/Reveal/adddata_to_dataset/new_test_set.csv")
test_code_list = df['Code'].tolist()
test_ast_list = df['Sim_SBT'].tolist()
test_filename_list = df['Filename'].tolist()

test_cpg_list = [l2_normalize(np.array(json.loads(x), dtype=np.float32)) for x in df['embedding_cpg'].tolist()]

Model_Name = "/ai/dataset/qsedata/GRACE/Pretrained_Models/codet5"
tokenizer = AutoTokenizer.from_pretrained(Model_Name)
model = T5Model.from_pretrained(Model_Name)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)


def sim_jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    ret1 = s1.intersection(s2)
    ret2 = s1.union(s2)
    return 1.0 * len(ret1) / len(ret2) if len(ret2) > 0 else 0.0

def cosine_sim(a, b):
    """标准余弦相似度 [-1,1]"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

def compute_stage1_score(code_vec, cpg_vec, code_vec_train, cpg_vec_train, alpha):
    """第一阶段: 语义 + 结构 (统一尺度 [-1,1])"""
    semantic_score = cosine_sim(code_vec, code_vec_train)
    struct_score   = cosine_sim(cpg_vec, cpg_vec_train)
    return alpha * semantic_score + (1 - alpha) * struct_score

def compute_stage2_score(code_score, ast_score):
    """第二阶段: 词汇 + 语法, 固定7:3"""
    return 0.7 * code_score + 0.3 * ast_score


class Retrieval(object):
    def __init__(self):
        f = open('./model/Reveal/code_vector_wotsne_512.pkl', 'rb')
        self.bert_vec = pickle.load(f)   # 语义向量
        f.close()
        self.vecs = None
        self.ids = None

    def encode_file(self):
        all_ids, all_vecs = [], []
        for i in range(len(train_code_list)):
            all_ids.append(i)
            all_vecs.append(self.bert_vec[i].reshape(1, -1))
        all_vecs = np.concatenate(all_vecs, 0)
        self.vecs = np.array(all_vecs, dtype="float32")
        self.ids = np.array(all_ids, dtype="int64")

    def build_index(self, n_list):
        quant = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quant, dim, min(n_list, self.vecs.shape[0]))
        index.train(self.vecs)
        index.add_with_ids(self.vecs, self.ids)
        self.index = index

    def lower_dimension(self, code):
        body = sents_to_vecs(code, tokenizer, model)
        body = normalize(body)
        return body

    def single_query(self, body_vec, code, ast, cpg_vec, topK, alpha):
        """
        两阶段检索：
        1. 初筛：语义 + 结构相似度 (全量, 统一尺度)
        2. 精排：TopK 内再计算语法+词汇 (固定3:7)
        """
        vec = body_vec.reshape(1, -1).astype('float32')

        stage1_scores = []
        for j in range(len(train_code_list)):
            score1 = compute_stage1_score(
                vec[0], cpg_vec,
                self.vecs[j], train_cpg_list[j],
                alpha
            )
            stage1_scores.append((j, score1))

        stage1_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = stage1_scores[:topK]

        max_score, max_idx = 0, 0
        for j, _ in candidates:
            code_score = sim_jaccard(train_code_list[j].split(), code.split())
            ast_score = Levenshtein.seqratio(str(train_ast_list[j]).split(), str(ast).split())
            final_score = compute_stage2_score(code_score, ast_score)

            if final_score > max_score:
                max_score = final_score
                max_idx = j

        return train_code_list[max_idx], train_ast_list[max_idx], train_label_list[max_idx]


if __name__ == '__main__':
    ccgir = Retrieval()
    print("Sentences to vectors")
    ccgir.encode_file()
    print("加载索引")
    ccgir.build_index(n_list=1)
    ccgir.index.nprobe = 1

    print("编码测试集...")
    code_semantic_vectors = ccgir.lower_dimension(test_code_list)

    output_dir = "gen_example/Reveal/sim_code_addstructure_stage1_topk={}_512".format(TOPK)
    os.makedirs(output_dir, exist_ok=True)

    for alpha in ALPHA_LIST:
        output_path = os.path.join(output_dir, f"sim_code_stageA_{alpha:.1f}.csv")
        print(f"生成文件: {output_path}")

        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(['Filename', 'sim_code', 'sim_label'])

            pbar = tqdm(total=len(test_code_list), desc=f'Alpha={alpha:.1f} - Two-stage retrieval...')
            for i in range(len(test_code_list)):
                sim_code, sim_ast, sim_label = ccgir.single_query(
                    code_semantic_vectors[i].reshape(1,-1),
                    test_code_list[i],
                    test_ast_list[i],
                    test_cpg_list[i],
                    topK=TOPK,
                    alpha=alpha
                )
                writer.writerow([test_filename_list[i], sim_code, sim_label])
                pbar.update(1)
            pbar.close()
