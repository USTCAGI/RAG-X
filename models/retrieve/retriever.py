from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import ExactMatchFilter, MetadataFilters
from llama_index.core.schema import QueryBundle, TextNode
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.vector_stores.milvus import MilvusVectorStore
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter

def get_all_chunks(query, search_results, chunk_size=200, chunk_overlap=0):
    chunks = []
    markdown_text_splitter = MarkdownTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    for search_result in search_results:
        text = search_result['page_result']
        snippet = search_result['page_snippet']
        if len(text) > 0:
            chunks.extend(markdown_text_splitter.split_text(text))
        if len(snippet) > 0:
            chunks.extend(text_splitter.split_text(snippet))
    return chunks

class Retriever:
    def __init__(self, top_k, top_n, embedding_model_path, reranker_model_path=None, rerank=False, chunk_size=200, chunk_overlap=0, sparse=0, broad_retrieval=False,device="cuda:0" ):
        self.top_k = top_k
        self.top_n = top_n
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.sparse = sparse
        self.embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_path, device=device
        )
        self.rerank = rerank
        if self.rerank:
            self.reranker = SentenceTransformerRerank(top_n=self.top_n, model=reranker_model_path, device=device)
        self.markdown_text_splitter = MarkdownTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        self.broad_retrieval = broad_retrieval
    def retrieve(self, query, interaction_id, search_results):
        ######################## pre-retrieval #########################
        from llama_index.core import VectorStoreIndex
        from llama_index.core.schema import Document, QueryBundle
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.retrievers.bm25 import BM25Retriever
        chunks = []
        if self.broad_retrieval == True:
            documents = []

            for search_result in search_results:
                text = search_result['page_result']
                snippet = search_result['page_snippet']
                if len(text) > 0:
                    documents.append(Document(text=text))
                if len(snippet) > 0:
                    documents.append(Document(text=snippet))

            
            node_parser = SentenceSplitter(chunk_size=200, chunk_overlap=0)
            nodes = node_parser.get_nodes_from_documents(documents)
            similarity_top_k = 50
            if len(nodes) < 50:
                print(f"Not enough nodes for BM25 retrieval. Using all nodes({len(nodes)}).")
                similarity_top_k = len(nodes)
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=similarity_top_k)
            nodes = bm25_retriever.retrieve(query)

            chunks = [node.get_text().strip() for node in nodes]    
        else: 
            # print("broader retrieval is not enabled")
            for search_result in search_results:
                text = search_result['page_result']
                snippet = search_result['page_snippet']
                if len(text) > 0:
                    chunks.extend(self.markdown_text_splitter.split_text(text))
                if len(snippet) > 0:
                    chunks.extend(self.text_splitter.split_text(snippet))



        ######################### retrieval #########################
        
        # node_parser = SentenceSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        nodes = [TextNode(text=chunk) for chunk in chunks]
        dense_nodes = []
        sparse_nodes = []
        if self.top_k - self.sparse > 0:
            index = VectorStoreIndex(nodes, embed_model=self.embedding_model)
            dense_retriever = index.as_retriever(similarity_top_k=self.top_k - self.sparse)
            dense_nodes = dense_retriever.retrieve(query)
        if self.sparse > 0:
            bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=self.sparse)
            sparse_nodes = bm25_retriever.retrieve(query)
        nodes = dense_nodes + sparse_nodes

        ######################### rerank #########################
        
        if self.rerank:
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                query_bundle=QueryBundle(query_str=query)
            )
            top_sentences = [node.get_text().strip() for node in reranked_nodes]
        else:
            top_sentences = [node.get_text().strip() for node in nodes]
        return top_sentences

class Retriever_Milvus:
    def __init__(self, top_k, top_n, collection_name, uri, embedding_model_path, reranker_model_path=None, rerank=False, device="cuda"):
        self.top_k = top_k
        self.top_n = top_n
        self.embedding_model = HuggingFaceEmbedding(
            model_name=embedding_model_path, device=device
        )
        vector_store = MilvusVectorStore(
            collection_name=collection_name,
            uri=uri,
            embedding_field="vector",
            text_key="text",
        )
        self.index = VectorStoreIndex.from_vector_store(vector_store, embed_model=self.embedding_model)
        self.rerank = rerank
        if self.rerank:
            self.reranker = self.reranker = SentenceTransformerRerank(top_n=self.top_k, model=reranker_model_path, device=device)
           
    def retrieve(self, query, interaction_id, search_results):
        metadata_filter = MetadataFilters(
            filters=[ExactMatchFilter(key="interaction_id", value=f"{interaction_id}")]
        )
        retriever = self.index.as_retriever(similarity_top_k=self.top_n, filters=metadata_filter)
        nodes = retriever.retrieve(query)

        ######################### rerank #########################
        if self.rerank:
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                query_bundle=QueryBundle(query_str=query)
            )
            top_sentences = [node.get_text().strip() for node in reranked_nodes]
        else:
            top_sentences = [node.get_text().strip() for node in nodes]
        return top_sentences