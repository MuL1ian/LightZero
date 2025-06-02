import os
import torch
import torch.nn.functional as F
import numpy as np
import selfies as sf
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter, defaultdict
from datetime import datetime
import argparse
from lzero.model.pretrain_transformer import load_pretrained_model, RealSpectrumSelfiesDataset
from lzero.model.selfies_tokenizer import SelfiesTokenizer


def analyze_selfies_lengths(test_data_file, tokenizer):
    print(f"original data file: {test_data_file}")
    
    data = torch.load(test_data_file, weights_only=False, map_location='cpu')
    
    smiles_list = data['smiles']
    selfies_lengths = []
    selfies_token_counts = []
    valid_count = 0
    invalid_count = 0
    
    print(f"total samples: {len(smiles_list):,}")
    
    for i, smiles in enumerate(tqdm(smiles_list, desc="convert smiles to selfies")):
        try:
            selfies_str = sf.encoder(smiles)
            
            selfies_lengths.append(len(selfies_str))
            
            tokens = list(sf.split_selfies(selfies_str))
            selfies_token_counts.append(len(tokens))
            
            valid_count += 1
            
        except Exception as e:
            invalid_count += 1
            if invalid_count <= 5:
                print(f"SMILES convert failed {i}: {smiles} -> {e}")
    
    selfies_lengths = np.array(selfies_lengths)
    selfies_token_counts = np.array(selfies_token_counts)
    
    
    print(f"\n📈 SELFIES length statistics (characters):")
    print(f"  ✅ valid samples: {valid_count:,} ({valid_count/len(smiles_list)*100:.1f}%)")
    print(f"  ❌ invalid samples: {invalid_count:,} ({invalid_count/len(smiles_list)*100:.1f}%)")
    print(f"  📏 min length: {selfies_lengths.min()}")
    print(f"  📏 max length: {selfies_lengths.max()}")
    print(f"  📏 mean length: {selfies_lengths.mean():.1f}")
    print(f"  📏 median length: {np.median(selfies_lengths):.1f}")
    print(f"  📏 std dev: {selfies_lengths.std():.1f}")
    
    print(f"\n📈 SELFIES length statistics (tokens):")
    print(f"  📏 min tokens: {selfies_token_counts.min()}")
    print(f"  📏 max tokens: {selfies_token_counts.max()}")
    print(f"  📏 mean tokens: {selfies_token_counts.mean():.1f}")
    print(f"  📏 median tokens: {np.median(selfies_token_counts):.1f}")
    print(f"  📏 std dev: {selfies_token_counts.std():.1f}")
    
    print(f"\n📊 SELFIES length distribution (characters):")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        print(f"  {p}% percentile: {np.percentile(selfies_lengths, p):.0f}")
    
    print(f"\n📊 SELFIES token distribution:")
    for p in percentiles:
        print(f"  {p}% percentile: {np.percentile(selfies_token_counts, p):.0f}")
    
    length_counter = Counter(selfies_token_counts)
    print(f"\n📊 most common token lengths:")
    for length, count in length_counter.most_common(10):
        print(f"  {length} tokens: {count:,} samples ({count/valid_count*100:.1f}%)")
    
    return {
        'total_samples': len(smiles_list),
        'valid_samples': valid_count,
        'invalid_samples': invalid_count,
        'selfies_char_lengths': selfies_lengths,
        'selfies_token_counts': selfies_token_counts,
        'length_stats': {
            'char_min': selfies_lengths.min(),
            'char_max': selfies_lengths.max(),
            'char_mean': selfies_lengths.mean(),
            'char_median': np.median(selfies_lengths),
            'char_std': selfies_lengths.std(),
            'token_min': selfies_token_counts.min(),
            'token_max': selfies_token_counts.max(),
            'token_mean': selfies_token_counts.mean(),
            'token_median': np.median(selfies_token_counts),
            'token_std': selfies_token_counts.std(),
        }
    }


def generate_selfies_batch_greedy(model, spectrums, tokenizer, config, device, max_len=None, k_predictions=1, temperature=1.0):
    """Batch generation of SELFIES sequences with k predictions and temperature sampling"""
    # Use config.max_len if max_len not specified
    if max_len is None:
        max_len = config.max_len
        
    # Store all k generations for each sample
    all_generated_selfies_lists = [[] for _ in range(spectrums.size(0))]

    for k_idx in range(k_predictions):
        batch_size = spectrums.size(0)
        
        # Initialize generation sequences [batch_size, 1]
        generated_tokens = torch.full((batch_size, 1), tokenizer.sos_token_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for step in range(max_len):
            seq_len = generated_tokens.size(1)
            if seq_len >= config.max_len - 1:
                break
                
            # Create input tensor [batch_size, config.max_len-1]
            input_tensor = torch.full((batch_size, config.max_len - 1), tokenizer.pad_token_id, dtype=torch.long, device=device)
            input_tensor[:, :seq_len] = generated_tokens
            
            # Create attention mask
            attention_mask = (input_tensor != tokenizer.pad_token_id).float()
            
            # Batch forward pass
            logits = model.forward_pretrain(spectrums, input_tensor, attention_mask)
            
            # Get prediction at last position [batch_size, vocab_size]
            last_logits = logits[:, seq_len-1, :]
            
            # Choose sampling strategy based on temperature and k_predictions
            if temperature == 0 or k_predictions == 1:  # Greedy decoding
                next_tokens = torch.argmax(last_logits, dim=-1)
            else:  # Temperature sampling
                scaled_logits = last_logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            
            # Check end conditions
            is_end = (next_tokens == tokenizer.eos_token_id) | (next_tokens == tokenizer.pad_token_id)
            finished = finished | is_end
            
            # If all sequences finished and at least one token generated, stop
            if finished.all() and step > 0:
                break
                
            # Add new tokens
            next_tokens = next_tokens.unsqueeze(1)
            generated_tokens = torch.cat([generated_tokens, next_tokens], dim=1)
            
            # For finished sequences, set new token to pad
            generated_tokens[finished, -1] = tokenizer.pad_token_id
        
        # Decode generated token sequences
        current_generated_selfies_list = []
        for i in range(batch_size):
            tokens = generated_tokens[i].tolist()
            tokens = tokens[1:]  # Remove SOS token
            
            # Truncate at EOS or PAD token
            if tokenizer.eos_token_id in tokens:
                eos_idx = tokens.index(tokenizer.eos_token_id)
                tokens = tokens[:eos_idx]
            elif tokenizer.pad_token_id in tokens:
                pad_idx = tokens.index(tokenizer.pad_token_id)
                tokens = tokens[:pad_idx]
                
            try:
                selfies_str = tokenizer.decode_to_selfies(tokens, skip_special_tokens=True)
                current_generated_selfies_list.append(selfies_str)
            except Exception as e:
                current_generated_selfies_list.append("")
        
        # Add current k-th generation to total results
        for i in range(batch_size):
            all_generated_selfies_lists[i].append(current_generated_selfies_list[i])

    # Return k predictions for each sample
    return all_generated_selfies_lists


def evaluate_selfies_exact_match(model, config, test_data_file, device, batch_size=32, k_predictions=1, temperature=1.0):
    print(f"\n🎯 Evaluating SELFIES exact match accuracy...")
    print(f"📊 Generation settings: k_predictions={k_predictions}, temperature={temperature}")
    
    tokenizer = SelfiesTokenizer(max_len=config.max_len)
    
    test_dataset = RealSpectrumSelfiesDataset(
        data_file=test_data_file,
        tokenizer=tokenizer,
        max_len=config.max_len
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    model.eval()
    
    total_samples = 0
    exact_matches = 0
    partial_matches = 0
    generation_failures = 0
    
    results = []
    match_by_length = defaultdict(lambda: {'total': 0, 'exact': 0})
    
    print(f"Starting evaluation of {len(test_dataset):,} test samples...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="evaluating")):
            spectrums = batch['spectrum'].to(device)
            target_selfies_list = batch['selfies']
            formulas = batch['formula']
            
            batch_size_actual = spectrums.size(0)
            
            try:
                generated_selfies_multi_k = generate_selfies_batch_greedy(
                    model, spectrums, tokenizer, config, device, 
                    max_len=config.max_len, 
                    k_predictions=k_predictions, 
                    temperature=temperature
                )
                
                for i in range(batch_size_actual):
                    target_selfies = target_selfies_list[i]
                    generated_selfies_candidates = generated_selfies_multi_k[i]
                    
                    is_exact_match = False
                    is_partial_match = False
                    best_candidate_selfies = ""
                    
                    for gen_selfies in generated_selfies_candidates:
                        if gen_selfies == target_selfies:
                            is_exact_match = True
                            best_candidate_selfies = gen_selfies
                            break
                        elif gen_selfies and not best_candidate_selfies:
                            best_candidate_selfies = gen_selfies

                    if not is_exact_match and best_candidate_selfies:
                        min_len = min(len(best_candidate_selfies), len(target_selfies))
                        if min_len > 0:
                            common_prefix = 0
                            for j in range(min_len):
                                if best_candidate_selfies[j] == target_selfies[j]:
                                    common_prefix += 1
                                else:
                                    break
                            is_partial_match = (common_prefix / min_len) > 0.5
                    
                    if is_exact_match:
                        exact_matches += 1
                    elif is_partial_match:
                        partial_matches += 1
                    
                    target_length = len(list(sf.split_selfies(target_selfies)))
                    match_by_length[target_length]['total'] += 1
                    if is_exact_match:
                        match_by_length[target_length]['exact'] += 1
                    
                    results.append({
                        'sample_idx': total_samples,
                        'target_selfies': target_selfies,
                        'generated_selfies': generated_selfies_candidates,
                        'exact_match': is_exact_match,
                        'partial_match': is_partial_match,
                        'formula': formulas[i],
                        'target_length': target_length,
                    })
                    
                    total_samples += 1
                
            except Exception as e:
                print(f"Batch generation failed, falling back to individual generation: {e}")
                for i in range(batch_size_actual):
                    spectrum = spectrums[i:i+1]
                    target_selfies = target_selfies_list[i]
                    formula = formulas[i]
                    
                    try:
                        generated_selfies_candidates_single = generate_selfies_batch_greedy(
                            model, spectrum, tokenizer, config, device, 
                            max_len=config.max_len, 
                            k_predictions=k_predictions, 
                            temperature=temperature
                        )[0]
                        
                        is_exact_match = False
                        is_partial_match = False
                        best_candidate_selfies = ""
                        
                        for gen_selfies in generated_selfies_candidates_single:
                            if gen_selfies == target_selfies:
                                is_exact_match = True
                                best_candidate_selfies = gen_selfies
                                break
                            elif gen_selfies and not best_candidate_selfies:
                                best_candidate_selfies = gen_selfies

                        if not is_exact_match and best_candidate_selfies:
                            min_len = min(len(best_candidate_selfies), len(target_selfies))
                            if min_len > 0:
                                common_prefix = 0
                                for j in range(min_len):
                                    if best_candidate_selfies[j] == target_selfies[j]:
                                        common_prefix += 1
                                    else:
                                        break
                                is_partial_match = (common_prefix / min_len) > 0.5
                        
                        if is_exact_match:
                            exact_matches += 1
                        elif is_partial_match:
                            partial_matches += 1
                            
                        target_length = len(list(sf.split_selfies(target_selfies)))
                        match_by_length[target_length]['total'] += 1
                        if is_exact_match:
                            match_by_length[target_length]['exact'] += 1
                        
                        results.append({
                            'sample_idx': total_samples,
                            'target_selfies': target_selfies,
                            'generated_selfies': generated_selfies_candidates_single,
                            'exact_match': is_exact_match,
                            'partial_match': is_partial_match,
                            'formula': formula,
                            'target_length': target_length,
                        })
                        
                    except Exception as e2:
                        generation_failures += 1
                        if generation_failures <= 5:
                            print(f"Individual generation failed {total_samples}: {e2}")
                        
                        results.append({
                            'sample_idx': total_samples,
                            'target_selfies': target_selfies,
                            'generated_selfies': None,
                            'exact_match': False,
                            'partial_match': False,
                            'formula': formula,
                            'target_length': len(list(sf.split_selfies(target_selfies))),
                            'error': str(e2)
                        })
                    
                    total_samples += 1
            
            if total_samples % 1000 == 0:
                current_exact_acc = exact_matches / total_samples * 100
                current_partial_acc = partial_matches / total_samples * 100
                print(f"  Processed {total_samples:,} samples - Exact: {current_exact_acc:.2f}%, Partial: {current_partial_acc:.2f}%")
    
    exact_accuracy = exact_matches / total_samples * 100
    partial_accuracy = partial_matches / total_samples * 100
    combined_accuracy = (exact_matches + partial_matches) / total_samples * 100
    generation_success_rate = (total_samples - generation_failures) / total_samples * 100
    
    print(f"\n🎯 Final Evaluation Results:")
    print(f"  📊 Total samples: {total_samples:,}")
    print(f"  ✅ Exact matches: {exact_matches:,} ({exact_accuracy:.2f}%)")
    print(f"  🔸 Partial matches: {partial_matches:,} ({partial_accuracy:.2f}%)")
    print(f"  🔸 Combined (exact + partial): {exact_matches + partial_matches:,} ({combined_accuracy:.2f}%)")
    print(f"  ❌ Generation failures: {generation_failures:,} ({generation_failures/total_samples*100:.2f}%)")
    print(f"  🚀 Generation success rate: {generation_success_rate:.2f}%")
    
    print(f"\n📊 Accuracy analysis by SELFIES length:")
    sorted_lengths = sorted(match_by_length.keys())
    for length in sorted_lengths:
        stats = match_by_length[length]
        if stats['total'] >= 5:
            acc = stats['exact'] / stats['total'] * 100
            print(f"  {length} tokens: {stats['exact']}/{stats['total']} = {acc:.1f}%")
    
    return {
        'total_samples': total_samples,
        'exact_matches': exact_matches,
        'partial_matches': partial_matches,
        'generation_failures': generation_failures,
        'exact_accuracy': exact_accuracy,
        'partial_accuracy': partial_accuracy,
        'combined_accuracy': combined_accuracy,
        'generation_success_rate': generation_success_rate,
        'results': results,
        'match_by_length': dict(match_by_length),
        'k_predictions': k_predictions,
        'temperature': temperature
    }


def plot_length_distribution(stats, save_path=None):
    try:
        plt.figure(figsize=(12, 8))
        plt.hist(stats['selfies_token_counts'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('SELFIES token distribution')
        plt.xlabel('token number')
        plt.ylabel('frequency')
        plt.axvline(stats['length_stats']['token_mean'], color='red', linestyle='--',
                   label=f'mean: {stats["length_stats"]["token_mean"]:.1f}')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 length distribution plot saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Plotting failed: {e}")


def save_evaluation_results(eval_results, stats, model_path, test_data_file, save_dir, k_predictions, temperature):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"evaluation_results_{timestamp}.txt")
    
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("SELFIES exact match accuracy evaluation report\n")
        f.write("=" * 50 + "\n")
        f.write(f"time: {datetime.now()}\n")
        f.write(f"model: {model_path}\n")
        f.write(f"test data: {test_data_file}\n")
        f.write(f"k_predictions: {k_predictions}\n")
        f.write(f"temperature: {temperature}\n\n")
        
        f.write("SELFIES length statistics:\n")
        f.write(f"  total samples: {stats['total_samples']:,}\n")
        f.write(f"  valid samples: {stats['valid_samples']:,}\n")
        f.write(f"  char length - mean: {stats['length_stats']['char_mean']:.1f}, std: {stats['length_stats']['char_std']:.1f}\n")
        f.write(f"  token number - mean: {stats['length_stats']['token_mean']:.1f}, std: {stats['length_stats']['token_std']:.1f}\n\n")
        
        f.write("model evaluation results:\n")
        f.write(f"  total test samples: {eval_results['total_samples']:,}\n")
        f.write(f"  exact match: {eval_results['exact_matches']:,} ({eval_results['exact_accuracy']:.2f}%)\n")
        f.write(f"  partial match: {eval_results['partial_matches']:,} ({eval_results['partial_accuracy']:.2f}%)\n")
        f.write(f"  combined accuracy: {eval_results['combined_accuracy']:.2f}%\n")
        f.write(f"  generation failed: {eval_results['generation_failures']:,}\n")
        f.write(f"  generation success rate: {eval_results['generation_success_rate']:.2f}%\n\n")
        
        f.write("by length analysis:\n")
        for length, match_stats in sorted(eval_results['match_by_length'].items()):
            if match_stats['total'] >= 5:
                acc = match_stats['exact'] / match_stats['total'] * 100
                f.write(f"  {length} tokens: {match_stats['exact']}/{match_stats['total']} = {acc:.1f}%\n")
    
    print(f"📄 evaluation results saved to: {results_file}")
    return results_file


def main():
    parser = argparse.ArgumentParser(description="Evaluate best model SELFIES exact match accuracy on test set")
    
    parser.add_argument("--model_path", type=str, 
                       default="./pretrained_selfies_transformer/best_model.pt",
                       help="Best model checkpoint path")
    parser.add_argument("--test_data_file", type=str,
                       default="/hy-tmp/MCTS/MassEnv/DataLoader/test_spectrum_embeds_msg.pt",
                       help="Test data file path")
    
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device (auto/cuda/cpu)")
    parser.add_argument("--max_generation_length", type=int, default=50,
                       help="Maximum generation length")
    
    parser.add_argument("--k_predictions", type=int, default=1,
                       help="Number of SELFIES sequences to generate per spectrum (for diversity evaluation). If > 1, sampling will be used.")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature for generating multiple predictions. Higher values increase randomness. Only applies when k_predictions > 1. Set to 0 for argmax (greedy) sampling.")
    
    parser.add_argument("--save_results", action="store_true", default=True,
                       help="Save evaluation results")
    parser.add_argument("--results_dir", type=str, default="./evaluation_results",
                       help="Results save directory")
    parser.add_argument("--plot_distribution", action="store_true", default=True,
                       help="Plot length distribution")
    
    args = parser.parse_args()
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"🖥️ Using device: {device}")
    
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return
    
    if not os.path.exists(args.test_data_file):
        print(f"❌ Test data file not found: {args.test_data_file}")
        return
    
    if args.save_results:
        os.makedirs(args.results_dir, exist_ok=True)
    
    print(f"📥 Loading model: {args.model_path}")
    try:
        model, config = load_pretrained_model(args.model_path, device)
        print(f"✅ Model loaded successfully")
        print(f"Model config: vocab_size={config.vocab_size}, max_len={config.max_len}")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    tokenizer = SelfiesTokenizer(max_len=config.max_len)
    
    print(f"\n{'='*60}")
    print(f"Step 1: Analyze test set SELFIES length distribution")
    print(f"{'='*60}")
    
    stats = analyze_selfies_lengths(args.test_data_file, tokenizer)
    
    if args.plot_distribution:
        plot_path = os.path.join(args.results_dir, "selfies_length_distribution.png") if args.save_results else None
        plot_length_distribution(stats, plot_path)
    
    print(f"\n{'='*60}")
    print(f"Step 2: Evaluate model SELFIES exact match accuracy")
    print(f"{'='*60}")
    
    eval_results = evaluate_selfies_exact_match(
        model, config, args.test_data_file, device, args.batch_size,
        k_predictions=args.k_predictions, temperature=args.temperature
    )
    
    if args.save_results:
        save_evaluation_results(
            eval_results, stats, args.model_path, args.test_data_file, args.results_dir,
            args.k_predictions, args.temperature
        )
    
    print(f"\n🎉 Evaluation completed!")
    print(f"📊 Final results:")
    print(f"   - SELFIES exact match accuracy = {eval_results['exact_accuracy']:.2f}%")
    print(f"   - SELFIES partial match accuracy = {eval_results['partial_accuracy']:.2f}%")
    print(f"   - Combined accuracy = {eval_results['combined_accuracy']:.2f}%")


if __name__ == "__main__":
    main() 