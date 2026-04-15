# coding = utf8

import torch
import numpy as np
# import faiss
# 注释掉 faiss 导入
import faiss

# # 替换 faiss 的使用
# class VectorSearchEngine:
#     def __init__(self, vectors, ids):
#         self.vectors = vectors
#         self.ids = ids
#
#     def search(self, query_vectors, k=10):
#         # 使用简单的暴力搜索替代 faiss
#         results = []
#         for query in query_vectors:
#             distances = np.linalg.norm(self.vectors - query, axis=1)
#             top_k_indices = np.argsort(distances)[:k]
#             results.append(self.ids[top_k_indices])
#         return results

class VectorSearchEngine(object):
    def __init__(self, vectors):
        super().__init__()
        if isinstance(vectors, torch.Tensor):
            self.vectors = vectors.detach().cpu().numpy()
        else:
            self.vectors = np.array(vectors)
        self.dim = self.vectors.shape[1]

        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.vectors)

    def search(self, query_vectors, k=10):
        query_vectors = np.asarray(query_vectors)
        topK_distances, topK_indices = self.index.search(query_vectors, k)

        return topK_distances, topK_indices