"""
Dataset loading module for Natural Questions and HotpotQA.
"""
from typing import Dict, List, Any, Optional
from datasets import load_dataset
from pathlib import Path
import json

from ..utils.logger import get_logger

logger = get_logger(__name__)


def load_squad(
    num_samples: int = 150,
    cache_dir: Optional[str] = None,
    save_to_disk: bool = True,
    data_dir: str = "src/data/raw"
) -> Dict[str, List[Any]]:
    """
    Load SQuAD v1.1 dataset from pre-downloaded JSON file.
    
    If the JSON file doesn't exist, you need to run:
        python download_squad.py
    
    Args:
        num_samples: Number of samples to load (must match downloaded file)
        cache_dir: Optional cache directory (not used, kept for compatibility)
        save_to_disk: Whether to save dataset as JSON (not used, kept for compatibility)
        data_dir: Directory where dataset JSON is located
        
    Returns:
        Dictionary with 'queries', 'answers', 'contexts', 'documents'
    """
    json_path = Path(data_dir) / f"squad_{num_samples}.json"
    
    if not json_path.exists():
        error_msg = (
            f"SQuAD dataset not found at: {json_path}\n"
            f"Please run: python download_squad.py\n"
            f"This will download {num_samples} samples from Hugging Face."
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    logger.info(f"Loading SQuAD v1.1 from: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
        
        queries = []
        answers = []
        contexts = []
        documents = []
        doc_ids = set()
        
        for sample in samples:
            queries.append(sample['question'])
            answers.append(sample['answer'])
            contexts.append(sample['context'])
            
            doc_id = sample['id']
            if doc_id not in doc_ids:
                documents.append(sample['context'])
                doc_ids.add(doc_id)
        
        logger.info(f"Loaded {len(queries)} questions from SQuAD v1.1")
        logger.info(f"Extracted {len(documents)} unique documents from contexts")
        
        result = {
            'queries': queries,
            'answers': answers,
            'contexts': contexts,
            'documents': documents,
            'metadata': {
                'dataset': 'squad',
                'version': 'v1.1',
                'split': 'validation',
                'num_samples': len(queries),
                'corpus_size': len(documents)
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to load SQuAD: {e}")
        raise


def load_hotpot_qa(
    num_samples: int = 500,
    cache_dir: Optional[str] = None
) -> Dict[str, List[Any]]:
    """
    Load HotpotQA dataset from HuggingFace.
    
    Args:
        num_samples: Number of samples to load
        cache_dir: Optional cache directory for dataset
        
    Returns:
        Dictionary with 'queries', 'answers', 'contexts', 'documents'
    """
    logger.info(f"Loading HotpotQA dataset ({num_samples} samples)...")
    
    try:
        dataset = load_dataset(
            "hotpot_qa",
            "distractor",
            split=f"validation[:{num_samples}]",
            cache_dir=cache_dir
        )
        
        queries = []
        answers = []
        contexts = []
        all_documents = []
        
        for item in dataset:
            queries.append(item['question'])
            
            answers.append(item['answer'])
            
            context_parts = []
            for title, sentences in zip(item['context']['title'], item['context']['sentences']):
                doc_text = ' '.join(sentences)
                context_parts.append(f"{title}: {doc_text}")
                all_documents.append(doc_text)
            
            full_context = ' '.join(context_parts)
            contexts.append(full_context)
        
        logger.info(f"Loaded {len(queries)} questions from HotpotQA")
        
        documents = list(set(all_documents))
        
        return {
            'queries': queries,
            'answers': answers,
            'contexts': contexts,
            'documents': documents,
            'metadata': {
                'dataset': 'hotpot_qa',
                'split': 'validation',
                'num_samples': len(queries)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to load HotpotQA: {e}")
        raise


def load_wikipedia_corpus(
    num_documents: int = 10000,
    cache_dir: Optional[str] = None,
    save_to_disk: bool = True,
    data_dir: str = "src/data/raw"
) -> List[str]:
    """
    Load Wikipedia corpus for retrieval.
    
    Args:
        num_documents: Number of Wikipedia articles to load
        cache_dir: Optional cache directory
        save_to_disk: Whether to save corpus as JSON
        data_dir: Directory to save corpus
        
    Returns:
        List of Wikipedia article texts
    """
    corpus_path = Path(data_dir) / f"wikipedia_{num_documents}.json"
    
    if corpus_path.exists():
        logger.info(f"Loading Wikipedia corpus from local cache: {corpus_path}")
        with open(corpus_path, 'r', encoding='utf-8') as f:
            corpus_data = json.load(f)
        logger.info(f"Loaded {len(corpus_data['documents'])} Wikipedia documents from cache")
        return corpus_data['documents']
    
    logger.info(f"Downloading Wikipedia corpus ({num_documents} documents)...")
    
    try:
        dataset = load_dataset(
            "wikipedia",
            "20220301.en",
            split=f"train[:{num_documents}]",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        
        documents = [item['text'] for item in dataset]
        
        logger.info(f"Loaded {len(documents)} Wikipedia documents")
        
        if save_to_disk:
            corpus_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving Wikipedia corpus to: {corpus_path}")
            corpus_data = {
                'documents': documents,
                'metadata': {
                    'source': 'wikipedia',
                    'version': '20220301.en',
                    'num_documents': len(documents)
                }
            }
            with open(corpus_path, 'w', encoding='utf-8') as f:
                json.dump(corpus_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(documents)} documents to {corpus_path}")
        
        return documents
        
    except Exception as e:
        logger.error(f"Failed to load Wikipedia corpus: {e}")
        raise


def prepare_dataset_for_rag(
    dataset: Dict[str, List[Any]],
    use_contexts_as_corpus: bool = True
) -> Dict[str, Any]:
    """
    Prepare dataset for RAG system.
    
    Args:
        dataset: Dataset dictionary from load_natural_questions or load_hotpot_qa
        use_contexts_as_corpus: If True, use contexts as document corpus
        
    Returns:
        Prepared dataset with document corpus
    """
    logger.info("Preparing dataset for RAG...")
    
    if use_contexts_as_corpus:
        corpus = dataset['documents']
    else:
        corpus = dataset['documents']
    
    prepared = {
        'queries': dataset['queries'],
        'answers': dataset['answers'],
        'contexts': dataset['contexts'],
        'corpus': corpus,
        'metadata': dataset.get('metadata', {})
    }
    
    logger.info(f"Dataset prepared: {len(prepared['queries'])} queries, {len(prepared['corpus'])} documents")
    return prepared


def save_dataset(
    dataset: Dict[str, Any],
    output_path: str
) -> None:
    """
    Save dataset to JSON file.
    
    Args:
        dataset: Dataset dictionary
        output_path: Path to save JSON file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset saved to {output_path}")


def load_saved_dataset(input_path: str) -> Dict[str, Any]:
    """
    Load dataset from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        Loaded dataset
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    logger.info(f"Dataset loaded from {input_path}")
    return dataset
