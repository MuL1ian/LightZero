# evaluate_diffms_style.py
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import selfies as sf
from tqdm import tqdm
from collections import Counter
import argparse
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime
import json
import csv

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Model imports
from lzero.model.muzero_transformer import SelfiesTokenizer
from lzero.model.pretrain.run_pretrain import RealSpectrumSelfiesDataset, load_pretrained_model

# =============================================================================
# DiffMS-style Evaluation Metrics
# =============================================================================

class K_ACC:
    """Top-K accuracy metric - checks if true molecule is in top-K predictions"""
    def __init__(self, k: int):
        self.k = k
        self.correct = 0
        self.total = 0
    
    def update(self, generated_inchis: List[str], true_inchi: str):
        """Update metric with a batch of predictions"""
        if true_inchi in generated_inchis[:self.k]:
            self.correct += 1
        self.total += 1
    
    def compute(self) -> float:
        """Compute final accuracy"""
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def reset(self):
        """Reset metric state"""
        self.correct = 0
        self.total = 0


class K_ACC_Collection:
    """Collection of K_ACC metrics for multiple K values"""
    def __init__(self, k_list: List[int]):
        self.metrics = {}
        for k in k_list:
            self.metrics[f"acc_at_{k}"] = K_ACC(k)
    
    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        """Update all metrics with generated and true molecules"""
        # Convert to InChI and filter valid molecules
        generated_inchis = []
        for mol in generated_mols:
            if mol is not None:
                try:
                    inchi = Chem.MolToInchi(mol)
                    generated_inchis.append(inchi)
                except:
                    pass
        
        inchi_counter = Counter(generated_inchis)
        unique_inchis = [inchi for inchi, _ in inchi_counter.most_common()]
        
        # Get true InChI
        try:
            true_inchi = Chem.MolToInchi(true_mol)
        except:
            true_inchi = "INVALID"
        
        # Update all K_ACC metrics
        for metric in self.metrics.values():
            metric.update(unique_inchis, true_inchi)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()


class K_TanimotoSimilarity:
    """Top-K Tanimoto similarity metric"""
    def __init__(self, k: int):
        self.k = k
        self.similarity_sum = 0.0
        self.total = 0
    
    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        """Update with generated and true molecules"""
        if true_mol is None:
            self.total += 1
            return
        
        try:
            true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
            max_sim = 0.0
            
            for mol in generated_mols[:self.k]:
                if mol is not None:
                    try:
                        gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        sim = DataStructs.TanimotoSimilarity(gen_fp, true_fp)
                        max_sim = max(max_sim, sim)
                    except:
                        pass
            
            self.similarity_sum += max_sim
        except:
            pass
        
        self.total += 1
    
    def compute(self) -> float:
        """Compute average max similarity"""
        if self.total == 0:
            return 0.0
        return self.similarity_sum / self.total
    
    def reset(self):
        """Reset metric state"""
        self.similarity_sum = 0.0
        self.total = 0


class K_CosineSimilarity:
    """Top-K Cosine similarity metric"""
    def __init__(self, k: int):
        self.k = k
        self.similarity_sum = 0.0
        self.total = 0
    
    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        """Update with generated and true molecules"""
        if true_mol is None:
            self.total += 1
            return
        
        try:
            true_fp = AllChem.GetMorganFingerprintAsBitVect(true_mol, 2, nBits=2048)
            max_sim = 0.0
            
            for mol in generated_mols[:self.k]:
                if mol is not None:
                    try:
                        gen_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                        sim = DataStructs.CosineSimilarity(gen_fp, true_fp)
                        max_sim = max(max_sim, sim)
                    except:
                        pass
            
            self.similarity_sum += max_sim
        except:
            pass
        
        self.total += 1
    
    def compute(self) -> float:
        """Compute average max similarity"""
        if self.total == 0:
            return 0.0
        return self.similarity_sum / self.total
    
    def reset(self):
        """Reset metric state"""
        self.similarity_sum = 0.0
        self.total = 0


class K_SimilarityCollection:
    """Collection of similarity metrics for multiple K values"""
    def __init__(self, k_list: List[int]):
        self.metrics = {}
        for k in k_list:
            self.metrics[f"tanimoto_at_{k}"] = K_TanimotoSimilarity(k)
            self.metrics[f"cosine_at_{k}"] = K_CosineSimilarity(k)
    
    def update(self, generated_mols: List[Chem.Mol], true_mol: Chem.Mol):
        """Update all similarity metrics"""
        # Filter out duplicates by InChI
        unique_mols = []
        seen_inchis = set()
        
        for mol in generated_mols:
            if mol is not None:
                try:
                    inchi = Chem.MolToInchi(mol)
                    if inchi not in seen_inchis:
                        unique_mols.append(mol)
                        seen_inchis.add(inchi)
                except:
                    pass
        
        # Update all similarity metrics
        for metric in self.metrics.values():
            metric.update(unique_mols, true_mol)
    
    def compute(self) -> Dict[str, float]:
        """Compute all similarity metrics"""
        return {name: metric.compute() for name, metric in self.metrics.items()}
    
    def reset(self):
        """Reset all metrics"""
        for metric in self.metrics.values():
            metric.reset()


class Validity:
    """Validity metric - measures fraction of valid molecules"""
    def __init__(self):
        self.valid = 0
        self.total = 0
    
    def update(self, generated_mols: List[Chem.Mol]):
        """Update with generated molecules"""
        for mol in generated_mols:
            if mol is not None:
                try:
                    # Check if molecule is valid by trying to compute basic properties
                    Chem.SanitizeMol(mol)
                    self.valid += 1
                except:
                    pass
            self.total += 1
    
    def compute(self) -> float:
        """Compute validity fraction"""
        if self.total == 0:
            return 0.0
        return self.valid / self.total
    
    def reset(self):
        """Reset metric state"""
        self.valid = 0
        self.total = 0


# =============================================================================
# SELFIES Generation Functions
# =============================================================================

def create_dataloader_from_pt(
    data_file: str, 
    batch_size: int, 
    max_len: int, 
    shuffle: bool = False
) -> Tuple[DataLoader, SelfiesTokenizer]:
    """Create dataloader from .pt file"""
    tokenizer = SelfiesTokenizer(max_len=max_len)
    
    dataset = RealSpectrumSelfiesDataset(
        data_file=data_file,
        tokenizer=tokenizer,
        max_len=max_len
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True
    )
    
    return dataloader, tokenizer


def generate_selfies_candidates_batch(
    model, 
    spectrums: torch.Tensor, 
    tokenizer: SelfiesTokenizer, 
    config, 
    device: torch.device, 
    num_candidates: int = 100,
    max_len: Optional[int] = None,
    temperature: float = 1.0
) -> List[List[str]]:
    """
    Generate multiple SELFIES candidates for a batch of spectrums using model.generate()
    
    Args:
        model: Trained model (MassSelfiesED with generate method)
        spectrums: Input spectrum embeddings [batch_size, 4096]
        tokenizer: SELFIES tokenizer
        config: Model configuration
        device: Device
        num_candidates: Number of candidates to generate per sample
        max_len: Maximum generation length
        temperature: Sampling temperature from model.generate()
    
    Returns:
        List of lists, where each inner list contains generated SELFIES for one sample
    """
    # Ensure max_len is a proper integer
    if max_len is None:
        max_len = int(config.max_len)
    else:
        max_len = int(max_len)
    
    model.eval()
    batch_size = spectrums.size(0)
    
    # Store all k generations for each sample
    all_generated_selfies_lists = [[] for _ in range(batch_size)]

    # Generate num_candidates for each sample
    for k_idx in range(num_candidates):
        with torch.no_grad():
            # Use model's native generate method for true autoregressive generation
            try:
                # Convert spectrums to the expected format
                generated_tokens, attention_mask = model.generate(
                    spectrum_input=spectrums,  # [batch_size, 4096]
                    max_len=max_len,
                    temperature=temperature,
                    greedy=(temperature == 0.0),
                    stop_at_eos=True
                )
                # generated_tokens: [batch_size, max_len]
                # attention_mask: [batch_size, max_len]
                
                # Decode generated token sequences for this k-th generation
                current_generated_selfies_list = []
                for i in range(batch_size):
                    tokens = generated_tokens[i].tolist()
                    
                    # Remove SOS token if first token is SOS (first token)
                    if len(tokens) > 0 and tokens[0] == tokenizer.sos_token_id:
                        tokens = tokens[1:]
                    
                    # Remove EOS and PAD tokens
                    if tokenizer.eos_token_id in tokens:
                        eos_idx = tokens.index(tokenizer.eos_token_id)
                        tokens = tokens[:eos_idx]
                    if tokenizer.pad_token_id in tokens:
                        pad_idx = tokens.index(tokenizer.pad_token_id)
                        tokens = tokens[:pad_idx]
                    
                    try:
                        selfies_str = tokenizer.decode_to_selfies(tokens, skip_special_tokens=True)
                        current_generated_selfies_list.append(selfies_str)
                    except Exception as e:
                        print(f"Warning: Failed to decode tokens {tokens}: {e}")
                        current_generated_selfies_list.append("")  # Failed to decode
                
                # Add current k-th generation to total results
                for i in range(batch_size):
                    all_generated_selfies_lists[i].append(current_generated_selfies_list[i])
                    
            except Exception as e:
                print(f"Warning: model.generate failed for candidate {k_idx}: {e}")
                # Fallback: add empty strings for this generation
                for i in range(batch_size):
                    all_generated_selfies_lists[i].append("")

    return all_generated_selfies_lists


def generate_selfies_candidates(
    model, 
    spectrum: torch.Tensor, 
    tokenizer: SelfiesTokenizer, 
    config, 
    device: torch.device, 
    num_candidates: int = 100,
    max_len: Optional[int] = None,
    temperature: float = 1.0
) -> List[str]:
    """
    Generate multiple SELFIES candidates for a single spectrum (backward compatibility)
    
    This function is kept for backward compatibility and calls the batch version internally.
    """
    # Call batch version with single sample
    spectrum_batch = spectrum.unsqueeze(0)  # [1, 4096]
    batch_results = generate_selfies_candidates_batch(
        model=model,
        spectrums=spectrum_batch,
        tokenizer=tokenizer,
        config=config,
        device=device,
        num_candidates=num_candidates,
        max_len=max_len,
        temperature=temperature
    )
    
    return batch_results[0]  # Return results for the single sample


def selfies_to_molecules(selfies_list: List[str]) -> List[Optional[Chem.Mol]]:
    """Convert SELFIES strings to RDKit molecules"""
    molecules = []
    for selfies_str in selfies_list:
        if not selfies_str:
            molecules.append(None)
            continue
        
        try:
            smiles = sf.decoder(selfies_str)
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                Chem.SanitizeMol(mol)
            molecules.append(mol)
        except:
            molecules.append(None)
    
    return molecules


# =============================================================================
# Main Evaluation Functions
# =============================================================================

def evaluate_model_diffms_style(
    model_path: str,
    test_data_path: str,
    device: torch.device,
    num_candidates: int = 100,
    batch_size: int = 32,
    max_samples: int = 0,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.9,
    save_dir: str = "./evaluation_results"
) -> Dict:
    """
    Evaluate model using DiffMS-style metrics
    
    Args:
        model_path: Path to trained model checkpoint
        test_data_path: Path to test dataset
        device: Device to run on
        num_candidates: Number of candidates to generate per sample (like test_samples_to_generate)
        batch_size: Batch size for processing
        max_samples: Maximum samples to evaluate (0 = all)
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p sampling
        save_dir: Directory to save results
    
    Returns:
        Dictionary of evaluation results
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create results directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load model and config
    logger.info(f"Loading model from {model_path}")
    model, config = load_pretrained_model(model_path, device)
    
    # Load test data
    logger.info(f"Loading test data from {test_data_path}")
    test_loader, tokenizer = create_dataloader_from_pt(
        test_data_path, 
        batch_size=batch_size, 
        max_len=config.max_len,
        shuffle=False
    )
    
    # Initialize DiffMS-style metrics
    k_values = list(range(1, min(num_candidates + 1, 101)))  # Top-1 to Top-100 (or num_candidates if smaller)

    test_k_acc = K_ACC_Collection(k_values)
    test_sim_metrics = K_SimilarityCollection(k_values)
    test_validity = Validity()
    
    # Evaluation loop
    logger.info(f"Starting evaluation with {num_candidates} candidates per sample (using batch inference)")
    
    total_samples = 0
    all_results = []
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        if max_samples > 0 and total_samples >= max_samples:
            break
        
        spectrums = batch['spectrum'].to(device)
        target_selfies_list = batch['selfies']
        formulas = batch['formula']
        
        batch_size_actual = spectrums.size(0)
        
        # Check if we need to limit the batch for max_samples
        if max_samples > 0 and total_samples + batch_size_actual > max_samples:
            remaining_samples = max_samples - total_samples
            spectrums = spectrums[:remaining_samples]
            target_selfies_list = target_selfies_list[:remaining_samples]
            formulas = formulas[:remaining_samples]
            batch_size_actual = remaining_samples
        
        # Batch inference for all samples in the batch
        try:
            # Generate candidates for all samples in the batch at once
            batch_generated_selfies = generate_selfies_candidates_batch(
                model=model,
                spectrums=spectrums,
                tokenizer=tokenizer,
                config=config,
                device=device,
                num_candidates=num_candidates,
                temperature=temperature
            )
            
            # Process each sample in the batch
            for i in range(batch_size_actual):
                target_selfies = target_selfies_list[i]
                formula = formulas[i]
                generated_selfies = batch_generated_selfies[i]
                
                # Convert to molecules
                generated_mols = selfies_to_molecules(generated_selfies)
                
                # Get target molecule
                try:
                    target_smiles = sf.decoder(target_selfies)
                    target_mol = Chem.MolFromSmiles(target_smiles)
                    if target_mol is not None:
                        Chem.SanitizeMol(target_mol)
                except:
                    target_mol = None
                
                # Update metrics
                if target_mol is not None:
                    test_k_acc.update(generated_mols, target_mol)
                    test_sim_metrics.update(generated_mols, target_mol)
                
                test_validity.update(generated_mols)
                
                # Store results
                all_results.append({
                    'sample_idx': total_samples,
                    'formula': formula,
                    'target_selfies': target_selfies,
                    'generated_selfies': generated_selfies,
                    'target_mol_valid': target_mol is not None,
                    'num_valid_generated': sum(1 for mol in generated_mols if mol is not None)
                })
                
                total_samples += 1
            
            # Progress logging
            if total_samples % 100 == 0:
                logger.info(f"Processed {total_samples} samples")
                
        except Exception as e:
            logger.error(f"Error processing batch {batch_idx}: {e}")
            # Fallback to single-sample processing for this batch
            logger.info("Falling back to single-sample processing for this batch")
            for i in range(batch_size_actual):
                if max_samples > 0 and total_samples >= max_samples:
                    break
                
                spectrum = spectrums[i]  # [4096]
                target_selfies = target_selfies_list[i]
                formula = formulas[i]
                
                try:
                    generated_selfies = generate_selfies_candidates(
                        model=model,
                        spectrum=spectrum,
                        tokenizer=tokenizer,
                        config=config,
                        device=device,
                        num_candidates=num_candidates,
                        temperature=temperature
                    )
                    
                    # Convert to molecules
                    generated_mols = selfies_to_molecules(generated_selfies)
                    
                    # Get target molecule
                    try:
                        target_smiles = sf.decoder(target_selfies)
                        target_mol = Chem.MolFromSmiles(target_smiles)
                        if target_mol is not None:
                            Chem.SanitizeMol(target_mol)
                    except:
                        target_mol = None
                    
                    # Update metrics
                    if target_mol is not None:
                        test_k_acc.update(generated_mols, target_mol)
                        test_sim_metrics.update(generated_mols, target_mol)
                    
                    test_validity.update(generated_mols)
                    
                    # Store results
                    all_results.append({
                        'sample_idx': total_samples,
                        'formula': formula,
                        'target_selfies': target_selfies,
                        'generated_selfies': generated_selfies,
                        'target_mol_valid': target_mol is not None,
                        'num_valid_generated': sum(1 for mol in generated_mols if mol is not None)
                    })
                    
                    total_samples += 1
                    
                except Exception as e2:
                    logger.error(f"Error processing single sample {total_samples}: {e2}")
                    total_samples += 1
                    continue
    
    # Compute final metrics
    logger.info("Computing final metrics...")
    
    k_acc_results = test_k_acc.compute()
    sim_results = test_sim_metrics.compute()
    validity_result = test_validity.compute()
    
    # Compile results
    results = {
        'evaluation_params': {
            'model_path': model_path,
            'test_data_path': test_data_path,
            'num_candidates': num_candidates,
            'total_samples': total_samples,
            'temperature': temperature,
            'top_k': top_k,
            'top_p': top_p,
            'timestamp': datetime.now().isoformat()
        },
        'metrics': {
            'accuracy': k_acc_results,
            'similarity': sim_results,
            'validity': validity_result
        },
        'detailed_results': all_results
    }
    
    # Log main results
    logger.info("=== EVALUATION RESULTS ===")
    logger.info(f"Total samples evaluated: {total_samples}")
    logger.info(f"Validity: {validity_result:.4f}")
    
    # Log top-K accuracies
    logger.info("Top-K Accuracies:")
    for k in [1, 5, 10, 20, 50, 100]:
        if f"acc_at_{k}" in k_acc_results:
            logger.info(f"  Top-{k}: {k_acc_results[f'acc_at_{k}']:.4f}")
    
    # Log similarities
    logger.info("Top-K Tanimoto Similarities:")
    for k in [1, 5, 10, 20, 50, 100]:
        if f"tanimoto_at_{k}" in sim_results:
            logger.info(f"  Top-{k}: {sim_results[f'tanimoto_at_{k}']:.4f}")
    
    # Save results
    results_file = os.path.join(save_dir, f"diffms_style_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate SELFIES model using DiffMS-style metrics")
    
    # Required arguments
    parser.add_argument("--model_path", type=str, default="/hy-tmp/MassEnv/main/LightZero/lzero/model/pretrain/pretrained_selfies_transformer/final_model.pt", help="Path to trained model checkpoint")
    parser.add_argument("--test_data_path", type=str, default="/hy-tmp/MassEnv/DataLoader/test_spectrum_embeds_msg.pt", help="Path to test dataset")
    
    # Evaluation parameters
    parser.add_argument("--num_candidates", type=int, default=10, help="Number of candidates per sample (like test_samples_to_generate)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for processing")
    parser.add_argument("--max_samples", type=int, default=0, help="Maximum samples to evaluate (0 = all)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 = disabled)")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="./evaluation_results", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    # Run evaluation
    results = evaluate_model_diffms_style(
        model_path=args.model_path,
        test_data_path=args.test_data_path,
        device=device,
        num_candidates=args.num_candidates,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        save_dir=args.save_dir
    )
    
    print("\n=== EVALUATION COMPLETED ===")
    print(f"Results saved to: {args.save_dir}")
    print(f"Total samples: {results['evaluation_params']['total_samples']}")
    print(f"Validity: {results['metrics']['validity']:.4f}")
    print(f"Top-1 Accuracy: {results['metrics']['accuracy'].get('acc_at_1', 0):.4f}")
    print(f"Top-10 Accuracy: {results['metrics']['accuracy'].get('acc_at_10', 0):.4f}")
    print(f"Top-50 Accuracy: {results['metrics']['accuracy'].get('acc_at_50', 0):.4f}")
    print(f"Top-100 Accuracy: {results['metrics']['accuracy'].get('acc_at_100', 0):.4f}")
    print(f"Top-1 cosine similarity: {results['metrics']['similarity'].get('cosine_at_1', 0):.4f}")
    print(f"Top-10 cosine similarity: {results['metrics']['similarity'].get('cosine_at_10', 0):.4f}")
    print(f"Top-50 cosine similarity: {results['metrics']['similarity'].get('cosine_at_50', 0):.4f}")
    print(f"Top-100 cosine similarity: {results['metrics']['similarity'].get('cosine_at_100', 0):.4f}")
    print(f"Top-1 tanimoto similarity: {results['metrics']['similarity'].get('tanimoto_at_1', 0):.4f}")
    print(f"Top-10 tanimoto similarity: {results['metrics']['similarity'].get('tanimoto_at_10', 0):.4f}")
    print(f"Top-50 tanimoto similarity: {results['metrics']['similarity'].get('tanimoto_at_50', 0):.4f}")
    print(f"Top-100 tanimoto similarity: {results['metrics']['similarity'].get('tanimoto_at_100', 0):.4f}")


if __name__ == "__main__":
    main()