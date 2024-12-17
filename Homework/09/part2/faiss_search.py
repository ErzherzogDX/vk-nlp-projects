from typing import List, Optional
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from part1.search_engine import Document, SearchResult


class FAISSSearcher:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Инициализация индекса
        """
        self.model = SentenceTransformer(model_name)
        self.documents: List[Document] = []
        self.index: Optional[faiss.Index] = None
        self.dimension: int = 384  # Размерность для 'all-MiniLM-L6-v2'

    def build_index(self, documents: List[Document]) -> None:
        """

        1. Сохранить документы
        2. Получить эмбеддинги через model.encode()
        3. Нормализовать векторы (faiss.normalize_L2)
        4. Создать индекс:
            - Создать quantizer = faiss.IndexFlatIP(dimension)
            - Создать индекс = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
            - Обучить индекс (train)
            - Добавить векторы (add)
        """
        self.documents = documents
        texts = [f"{doc.title} {doc.text}" for doc in documents]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        embeddings = embeddings.astype(np.float32)
        faiss.normalize_L2(embeddings)

        quantizer = faiss.IndexFlatIP(self.dimension)
        n_clusters = 10
        index = faiss.IndexIVFFlat(quantizer, self.dimension, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        self.index = index

    def save(self, path: str) -> None:
        """
        1. Сохранить в pickle:
            - documents
            - индекс (faiss.serialize_index)
        """
        serialized_index = faiss.serialize_index(self.index)
        with open(path, 'wb') as f:
            pickle.dump((self.documents, serialized_index), f)

    def load(self, path: str) -> None:
        """

        1. Загрузить из pickle:
            - documents
            - индекс (faiss.deserialize_index)
        """
        with open(path, 'rb') as f:
            self.documents, serialized_index = pickle.load(f)
        self.index = faiss.deserialize_index(serialized_index)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """

        1. Получить эмбеддинг запроса
        2. Нормализовать вектор
        3. Искать через index.search()
        4. Вернуть найденные документы
        """
        query_emb = self.model.encode([query], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_emb)
        D, I = self.index.search(query_emb, top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            doc = self.documents[idx]
            results.append(
                SearchResult(
                    doc_id=doc.id,
                    score=float(score),
                    title=doc.title,
                    text=doc.text
                )
            )
        return results

    def batch_search(self, queries: List[str], top_k: int = 5) -> List[List[SearchResult]]:
        """
        1. Получить эмбеддинги всех запросов
        2. Нормализовать векторы
        3. Искать через index.search()
        4. Вернуть результаты для каждого запроса
        """
        query_embs = self.model.encode(queries, convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(query_embs)
        D, I = self.index.search(query_embs, top_k)
        all_results = []
        for i in range(len(queries)):
            results = []
            for idx, score in zip(I[i], D[i]):
                doc = self.documents[idx]
                results.append(
                    SearchResult(
                        doc_id=doc.id,
                        score=float(score),
                        title=doc.title,
                        text=doc.text
                    )
                )
            all_results.append(results)
        return all_results
