from typing import List, Dict, Any
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from init.config import Config
from init.logger import logger
from base.llm import LLMManager
from base.embeddings import EmbeddingManager
from chunking.chunk_processor import ChunkProcessor
from .dynamic_memory import DynamicMemory

class GraphRAGCore:
    """Core GraphRAG implementation."""

    def __init__(self, config: Config, base_dir: str = "./results"):
        """Initialize GraphRAG core."""
        self.config = config
        self.base_dir = base_dir
        self.skip_embedding_init = os.getenv("LICOMEMORY_SKIP_EMBEDDING_INIT", "0").lower() in {"1", "true", "yes", "on"}
        self.llm_manager = LLMManager(
            api_key=config.llm.api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_token,
            temperature=config.llm.temperature,
            base_url=config.llm.base_url,
            enable_concurrent=config.llm.enable_concurrent,
            max_concurrent=config.llm.max_concurrent,
            timeout=config.llm.timeout,
            retry_attempts=config.llm.retry_attempts,
            retry_backoff=config.llm.retry_backoff,
            retry_backoff_max=config.llm.retry_backoff_max,
            fail_on_error=config.llm.fail_on_error,
        )
        if self.skip_embedding_init:
            self.embedding_manager = None
            logger.warning("Embedding manager initialization skipped by LICOMEMORY_SKIP_EMBEDDING_INIT")
        else:
            self.embedding_manager = EmbeddingManager(config.embedding)
        data_type = getattr(config, 'data_type', 'LongmemEval')
        self.chunk_processor = ChunkProcessor(config.chunk, data_type=data_type)

        self.graph = self._create_graph(config)

        if hasattr(self.graph, 'set_extractors'):
            self.graph.set_extractors(self.llm_manager, self.embedding_manager)

        logger.info("GraphRAG Core initialized")

    def _create_graph(self, config: Config):
        graph = DynamicMemory(config, self.base_dir)
        graph.chunk_processor = self.chunk_processor
        return graph

    def _attach_corpus_chunks_to_loaded_graph(self, corpus: List[Dict[str, Any]]) -> int:
        """In load-existing mode, attach corpus chunks into chunk_storage for direct retrieval.

        This keeps the existing graph edges unchanged while expanding retrieval candidates
        with current-sample corpus chunks (including irrelevant sessions).
        """
        if not corpus:
            return 0
        if not hasattr(self.graph, "chunk_storage") or self.graph.chunk_storage is None:
            self.graph.chunk_storage = {}

        chunks = self.chunk_processor.process_corpus(corpus)
        if not chunks:
            return 0

        existing = self.graph.chunk_storage
        added = 0
        for chunk in chunks:
            text = str(chunk.get("text", "") or "").strip()
            if not text:
                continue
            session_id = str(chunk.get("session_id", "") or "")
            session_time = str(chunk.get("session_time", "") or "")
            raw_chunk_id = str(chunk.get("chunk_id", "") or "")
            key = f"corpus::{session_id}::{raw_chunk_id}"
            if key in existing:
                suffix = 1
                while f"{key}::{suffix}" in existing:
                    suffix += 1
                key = f"{key}::{suffix}"
            existing[key] = {
                "text": text,
                "session_id": session_id,
                "session_time": session_time,
            }
            added += 1
        logger.info(
            f"Load-existing corpus attachment added {added} chunks "
            f"(chunk_storage now={len(existing)})"
        )
        return added

    async def insert(self, corpus: List[Dict[str, Any]]) -> None:
        self.graph.time_manager.start_total_graph_building()
        
        add = getattr(self.config.graph, 'add', False)
        force = getattr(self.config.graph, 'force', False)
        graph_path = os.path.join(self.base_dir, f"{self.config.index_name}.pkl")

        # Load-existing mode: do not rebuild graph. Optionally attach corpus chunks
        # into chunk_storage for retrieval-only expansion.
        if not force and not add:
            if os.path.exists(graph_path):
                logger.info(f"Load existing mode: loading graph from {graph_path}")
                self.graph.load_graph(graph_path)
                stats = self.graph.get_graph_stats()
                logger.info(f"Loaded graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            else:
                logger.warning("Load existing mode requested but no graph found; building empty graph")
                await self.graph.build_graph([])
                self.graph.save_graph(graph_path)
                stats = self.graph.get_graph_stats()
                logger.info(f"Graph saved with {stats['num_nodes']} nodes and {stats['num_edges']} edges")

            attach_corpus_chunks = bool(
                getattr(self.config.retriever, "load_only_attach_corpus_chunks", False)
            )
            if attach_corpus_chunks and corpus:
                self._attach_corpus_chunks_to_loaded_graph(corpus)

            self.graph.time_manager.end_total_graph_building()
            logger.info("Document insertion completed")
            return
        
        if add and corpus:
            logger.info("🔄 ADD MODE: Processing sessions sequentially")
            
            if os.path.exists(graph_path):
                logger.info(f"Loading existing graph from {graph_path}")
                self.graph.load_graph(graph_path)
                stats = self.graph.get_graph_stats()
                logger.info(f"Loaded graph with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            else:
                logger.info("No existing graph found, will create new graph")
                await self.graph.build_graph([])
            
            sessions_map = {}
            for doc in corpus:
                session_id = doc.get('session_id', 'unknown')
                if session_id not in sessions_map:
                    sessions_map[session_id] = []
                sessions_map[session_id].append(doc)
            
            logger.info(f"Found {len(sessions_map)} unique sessions to process")
            
            for idx, (session_id, session_docs) in enumerate(sessions_map.items(), 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"Processing session {idx}/{len(sessions_map)}: {session_id}")
                logger.info(f"{'='*80}")
                
                await self.graph.add_single_session(session_docs)
                
                self.graph.save_graph(graph_path)
                stats = self.graph.get_graph_stats()
                logger.info(f"Graph saved with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            
            logger.info("\n" + "="*80)
            logger.info("✅ All sessions processed successfully")
            logger.info("="*80)
            
        else:
            if self.config.retriever.enable_summary:
                logger.info("Generating session summaries...")
                summaries = await self.graph.generate_session_summaries(corpus)
                logger.info(f"Generated {len(summaries)} session summaries")
            else:
                logger.info("Summary generation disabled in config")
                summaries = []
            
            chunks = self.chunk_processor.process_corpus(corpus)
            logger.info(f"Processed {len(chunks)} chunks from {len(corpus)} documents")

            await self.graph.build_graph(chunks)
            
            graph_path = os.path.join(self.base_dir, f"{self.config.index_name}.pkl")
            self.graph.save_graph(graph_path)

            stats = self.graph.get_graph_stats()
            logger.info(f"Graph built with {stats['num_nodes']} nodes and {stats['num_edges']} edges")
            logger.info(f"Graph saved to {graph_path}")
        
        self.graph.time_manager.end_total_graph_building()
        logger.info("Document insertion completed")
