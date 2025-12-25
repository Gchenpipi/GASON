from tree_sitter import Language, Parser  # 解析器库
import tree_sitter_c as tsc
import re
import pandas as pd
import Levenshtein
import os
import sys
import numpy as np
from tqdm import tqdm

def read_c_file(file_path):
    """
    读取.c文件内容并返回字符串
    :param file_path: .c文件的路径
    :return: 文件内容字符串
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except PermissionError:
        print(f"错误：无权限读取文件 {file_path}")
        return None



def process_c_files(directory_path):
    """
    处理目录中的所有.c文件
    :param directory_path: 包含.c文件的目录路径
    """
    parser = Parser()
    c_language = Language("build/my-languages.so", "c")
    parser.set_language(c_language)

    pbar = tqdm(total=len(os.listdir(directory_path)))

    for filename in os.listdir(directory_path):
        if filename.endswith(".c"):
            file_path = os.path.join(directory_path, filename)
            c_code = read_c_file(file_path)

            if c_code:
                tree = parser.parse(bytes(c_code, "utf8"))
                root = tree.root_node  # 注意，root_node 才是可遍历的树节点
                sexp = root.sexp()
                cleaned_sexp = re.sub(r'[:\(\)]', '', sexp)

                # ast.append(sexp)
                save_result(filename, cleaned_sexp, c_code)

        pbar.update(1)





def save_result(filename, Sim_SBT, clean_code, output_csv="./gen_data/chrome/chrome_results.csv"):
    """
    将处理结果保存到CSV文件（支持追加模式）

    :param filename: 原始C文件名
    :param ast_result: AST表示字符串
    :param sbt_result: SBT表示字符串
    :param output_csv: 输出CSV文件名（默认results.csv）
    """
    # 创建数据记录（包含原始文件名和处理结果）
    data = {
        "Filename": [filename],
        "Sim_SBT": [Sim_SBT],
        "Code": [clean_code]
    }

    # 转换为DataFrame
    df = pd.DataFrame(data)

    # 检查文件是否存在以决定是否写入表头
    file_exists = os.path.exists(output_csv)

    # 保存到CSV（追加模式）
    df.to_csv(
        output_csv,
        mode='a',  # 追加模式
        header=not file_exists,  # 新文件写入表头，已存在文件跳过表头
        index=False,  # 不保存行索引
        encoding='utf-8-sig'  # 支持中文字符（Excel兼容）
    )

if __name__ == '__main__':
    process_c_files("./data_preprocess/data/code_after_filtering/chrome")
