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
from lzero.model.selfies_tokenizer import SelfiesTokenizer
from lzero.model.pretrain.run_pretrain import RealSpectrumSelfiesDataset, load_pretrained_model

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
            
            # Batch forward pass - handle both old (logits only) and new (logits, values) format
            output = model.forward_pretrain(spectrums, input_tensor, attention_mask, return_value=False)
            if isinstance(output, tuple):
                logits = output[0]  # Extract logits from (logits, values) tuple
            else:
                logits = output  # Use directly if only logits returned
            
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
    chemical_equivalents = 0
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
                    is_chemically_equivalent = False
                    is_partial_match = False
                    best_candidate_selfies = ""
                    best_similarity_score = 0.0
                    
                    for gen_selfies in generated_selfies_candidates:
                        exact_match, chem_equiv, partial_match, similarity = check_selfies_correctness(target_selfies, gen_selfies)
                        
                        if exact_match:
                            is_exact_match = True
                            is_chemically_equivalent = True
                            is_partial_match = True
                            best_candidate_selfies = gen_selfies
                            best_similarity_score = 1.0
                            break
                        elif chem_equiv and not is_chemically_equivalent:
                            is_chemically_equivalent = True
                            best_candidate_selfies = gen_selfies
                            best_similarity_score = similarity
                        elif similarity > best_similarity_score:
                            is_partial_match = partial_match
                            best_candidate_selfies = gen_selfies
                            best_similarity_score = similarity

                    if is_exact_match:
                        exact_matches += 1
                    elif is_chemically_equivalent:
                        chemical_equivalents += 1
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
                        is_chemically_equivalent = False
                        is_partial_match = False
                        best_candidate_selfies = ""
                        best_similarity_score = 0.0
                        
                        for gen_selfies in generated_selfies_candidates_single:
                            exact_match, chem_equiv, partial_match, similarity = check_selfies_correctness(target_selfies, gen_selfies)
                            
                            if exact_match:
                                is_exact_match = True
                                is_chemically_equivalent = True
                                is_partial_match = True
                                best_candidate_selfies = gen_selfies
                                best_similarity_score = 1.0
                                break
                            elif chem_equiv and not is_chemically_equivalent:
                                is_chemically_equivalent = True
                                best_candidate_selfies = gen_selfies
                                best_similarity_score = similarity
                            elif similarity > best_similarity_score:
                                is_partial_match = partial_match
                                best_candidate_selfies = gen_selfies
                                best_similarity_score = similarity

                        if is_exact_match:
                            exact_matches += 1
                        elif is_chemically_equivalent:
                            chemical_equivalents += 1
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
    chemical_accuracy = chemical_equivalents / total_samples * 100
    partial_accuracy = partial_matches / total_samples * 100
    combined_accuracy = (exact_matches + chemical_equivalents + partial_matches) / total_samples * 100
    generation_success_rate = (total_samples - generation_failures) / total_samples * 100
    
    print(f"\n🎯 Final Evaluation Results:")
    print(f"  📊 Total samples: {total_samples:,}")
    print(f"  ✅ Exact matches: {exact_matches:,} ({exact_accuracy:.2f}%)")
    print(f"  🧪 Chemical equivalents: {chemical_equivalents:,} ({chemical_accuracy:.2f}%)")
    print(f"  🔸 Partial matches: {partial_matches:,} ({partial_accuracy:.2f}%)")
    print(f"  🔸 Combined (exact + chemical + partial): {exact_matches + chemical_equivalents + partial_matches:,} ({combined_accuracy:.2f}%)")
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
        'chemical_equivalents': chemical_equivalents,
        'partial_matches': partial_matches,
        'generation_failures': generation_failures,
        'exact_accuracy': exact_accuracy,
        'chemical_accuracy': chemical_accuracy,
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


def save_evaluation_results(eval_results, stats, model_path, test_data_file, save_dir, k_predictions, temperature, badcase_examples=None, error_counts=None, value_eval_results=None):
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
        f.write(f"  chemical equivalents: {eval_results['chemical_equivalents']:,} ({eval_results['chemical_accuracy']:.2f}%)\n")
        f.write(f"  partial match: {eval_results['partial_matches']:,} ({eval_results['partial_accuracy']:.2f}%)\n")
        f.write(f"  combined accuracy: {eval_results['combined_accuracy']:.2f}%\n")
        f.write(f"  generation failed: {eval_results['generation_failures']:,}\n")
        f.write(f"  generation success rate: {eval_results['generation_success_rate']:.2f}%\n\n")
        
        f.write("by length analysis:\n")
        for length, match_stats in sorted(eval_results['match_by_length'].items()):
            if match_stats['total'] >= 5:
                acc = match_stats['exact'] / match_stats['total'] * 100
                f.write(f"  {length} tokens: {match_stats['exact']}/{match_stats['total']} = {acc:.1f}%\n")
        
        if value_eval_results:
            f.write("\nValue Head evaluation results:\n")
            f.write(f"  total test samples: {value_eval_results['total_samples']:,}\n")
            f.write(f"  total tokens: {value_eval_results['total_tokens']:,}\n")
            f.write(f"  value accuracy: {value_eval_results['value_accuracy']:.2f}%\n")
            f.write(f"  precision: {value_eval_results['precision']:.3f}\n")
            f.write(f"  recall: {value_eval_results['recall']:.3f}\n")
            f.write(f"  f1 score: {value_eval_results['f1_score']:.3f}\n")
            f.write(f"  positive samples: {value_eval_results['positive_samples']:,}\n")
            f.write(f"  negative samples: {value_eval_results['negative_samples']:,}\n")
        
        if error_counts:
            f.write(f"\nBadcase error type statistics:\n")
            total_errors = sum(error_counts.values())
            sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
            
            for error_type, count in sorted_errors:
                if count > 0:
                    percentage = count / total_errors * 100 if total_errors > 0 else 0
                    f.write(f"  {error_type}: {count} ({percentage:.1f}%)\n")
    
    print(f"📄 evaluation results saved to: {results_file}")
    return results_file


def analyze_badcase(target_selfies, generated_selfies, formula):
    error_types = []
    
    if not generated_selfies or generated_selfies == "":
        return ["generation_failure"], "生成失败"
    
    try:
        sf.decoder(generated_selfies)
        is_valid_selfies = True
    except:
        is_valid_selfies = False
        error_types.append("invalid_selfies")
    
    target_tokens = list(sf.split_selfies(target_selfies))
    try:
        generated_tokens = list(sf.split_selfies(generated_selfies))
    except:
        generated_tokens = []
    
    target_len = len(target_tokens)
    generated_len = len(generated_tokens)
    
    length_ratio = generated_len / target_len if target_len > 0 else 0
    
    if length_ratio < 0.5:
        error_types.append("too_short")
    elif length_ratio > 2.0:
        error_types.append("too_long")
    elif abs(length_ratio - 1.0) > 0.3:
        error_types.append("length_mismatch")
    
    if generated_tokens and target_tokens:
        common_prefix = 0
        min_len = min(len(target_tokens), len(generated_tokens))
        for i in range(min_len):
            if target_tokens[i] == generated_tokens[i]:
                common_prefix += 1
            else:
                break
        
        prefix_ratio = common_prefix / min_len if min_len > 0 else 0
        
        if prefix_ratio > 0.8:
            error_types.append("suffix_error")
        elif prefix_ratio > 0.5:
            error_types.append("partial_match")
        elif prefix_ratio < 0.2:
            error_types.append("completely_wrong")
    
    # check consecutive repeats
    if generated_tokens:
        consecutive_repeats = 0
        for i in range(1, len(generated_tokens)):
            if generated_tokens[i] == generated_tokens[i-1]:
                consecutive_repeats += 1
        
        if consecutive_repeats > len(generated_tokens) * 0.3:
            error_types.append("repetitive")
    
    # check truncated
    if generated_selfies.endswith('[') or generated_selfies.count('[') != generated_selfies.count(']'):
        error_types.append("truncated")
    
    # if no other error types, but mismatch, classify as general error
    if not error_types and generated_selfies != target_selfies:
        error_types.append("general_mismatch")
    
    # generate error description
    error_desc = f"长度: {generated_len}/{target_len} ({length_ratio:.2f}x)"
    if not is_valid_selfies:
        error_desc += ", 无效SELFIES"
    
    return error_types, error_desc


def collect_badcase_examples(results, max_examples_per_type=5):
    """collect examples for each error type"""
    badcase_examples = {
        "generation_failure": [],
        "invalid_selfies": [],
        "too_short": [],
        "too_long": [],
        "length_mismatch": [],
        "suffix_error": [],
        "partial_match": [],
        "completely_wrong": [],
        "repetitive": [],
        "truncated": [],
        "general_mismatch": []
    }
    
    error_counts = {key: 0 for key in badcase_examples.keys()}
    
    for result in results:
        if result['exact_match']:
            continue
            
        target_selfies = result['target_selfies']
        generated_selfies_list = result['generated_selfies']
        formula = result['formula']
        
        # take the first generated result for analysis
        generated_selfies = generated_selfies_list[0] if generated_selfies_list else ""
        
        error_types, error_desc = analyze_badcase(target_selfies, generated_selfies, formula)
        
        for error_type in error_types:
            error_counts[error_type] += 1
            
            if len(badcase_examples[error_type]) < max_examples_per_type:
                badcase_examples[error_type].append({
                    'sample_idx': result['sample_idx'],
                    'formula': formula,
                    'target_selfies': target_selfies,
                    'generated_selfies': generated_selfies,
                    'error_desc': error_desc,
                    'target_length': len(list(sf.split_selfies(target_selfies))),
                    'generated_length': len(list(sf.split_selfies(generated_selfies))) if generated_selfies else 0
                })
    
    return badcase_examples, error_counts


def print_badcase_analysis(badcase_examples, error_counts, total_errors):
    """打印badcase分析结果"""
    print(f"\n🔍 Badcase 分析报告")
    print(f"{'='*60}")
    
    # sort by error count
    sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
    
    for error_type, count in sorted_errors:
        if count == 0:
            continue
            
        percentage = count / total_errors * 100 if total_errors > 0 else 0
        print(f"\n📊 {error_type.upper()}: {count} 个 ({percentage:.1f}%)")
        
        error_type_names = {
            "generation_failure": "生成失败",
            "invalid_selfies": "无效SELFIES语法",
            "too_short": "生成过短",
            "too_long": "生成过长", 
            "length_mismatch": "长度不匹配",
            "suffix_error": "前缀正确但后缀错误",
            "partial_match": "部分匹配",
            "completely_wrong": "完全错误",
            "repetitive": "重复生成",
            "truncated": "截断错误",
            "general_mismatch": "一般不匹配"
        }
        
        print(f"   类型: {error_type_names.get(error_type, error_type)}")
        
        examples = badcase_examples[error_type]
        for i, example in enumerate(examples[:5], 1):
            print(f"\n   样例 {i}:")
            print(f"     分子式: {example['formula']}")
            print(f"     真实值: {example['target_selfies'][:100]}{'...' if len(example['target_selfies']) > 100 else ''}")
            print(f"     生成值: {example['generated_selfies'][:100]}{'...' if len(example['generated_selfies']) > 100 else ''}")
            print(f"     错误信息: {example['error_desc']}")
        
        if len(examples) > 5:
            print(f"   ... 还有 {len(examples) - 5} 个样例")


def save_badcase_analysis(badcase_examples, error_counts, save_dir, timestamp):
    """save badcase analysis to file"""
    badcase_file = os.path.join(save_dir, f"badcase_analysis_{timestamp}.txt")
    
    with open(badcase_file, 'w', encoding='utf-8') as f:
        f.write("BADCASE 分析报告\n")
        f.write("=" * 50 + "\n")
        f.write(f"生成时间: {datetime.now()}\n\n")
        
        total_errors = sum(error_counts.values())
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        for error_type, count in sorted_errors:
            if count == 0:
                continue
                
            percentage = count / total_errors * 100 if total_errors > 0 else 0
            f.write(f"\n{error_type.upper()}: {count} 个 ({percentage:.1f}%)\n")
            f.write("-" * 30 + "\n")
            
            examples = badcase_examples[error_type]
            for i, example in enumerate(examples, 1):
                f.write(f"\n样例 {i}:\n")
                f.write(f"  样本ID: {example['sample_idx']}\n")
                f.write(f"  分子式: {example['formula']}\n")
                f.write(f"  真实SELFIES: {example['target_selfies']}\n")
                f.write(f"  生成SELFIES: {example['generated_selfies']}\n")
                f.write(f"  错误描述: {example['error_desc']}\n")
                f.write(f"  长度对比: {example['generated_length']}/{example['target_length']}\n")
    
    print(f"📄 Badcase分析保存到: {badcase_file}")
    return badcase_file


def check_selfies_correctness(target_selfies, generated_selfies):
    """
    检查生成的SELFIES的正确性
    返回: (is_exact_match, is_chemically_equivalent, is_partial_match, similarity_score)
    """
    if not generated_selfies or generated_selfies.strip() == "":
        return False, False, False, 0.0
    
    # 1. 精确匹配（字符串完全相同）
    if generated_selfies.strip() == target_selfies.strip():
        return True, True, True, 1.0
    
    # 2. 化学等价性检查（转换为SMILES后比较）
    is_chemically_equivalent = False
    try:
        target_smiles = sf.decoder(target_selfies)
        generated_smiles = sf.decoder(generated_selfies)
        
        # 标准化SMILES进行比较
        from rdkit import Chem
        target_mol = Chem.MolFromSmiles(target_smiles)
        generated_mol = Chem.MolFromSmiles(generated_smiles)
        
        if target_mol is not None and generated_mol is not None:
            target_canonical = Chem.MolToSmiles(target_mol, canonical=True)
            generated_canonical = Chem.MolToSmiles(generated_mol, canonical=True)
            is_chemically_equivalent = (target_canonical == generated_canonical)
    except:
        # 如果转换失败，说明生成的SELFIES无效
        is_chemically_equivalent = False
    
    # 3. Token级别的相似度计算
    try:
        target_tokens = list(sf.split_selfies(target_selfies))
        generated_tokens = list(sf.split_selfies(generated_selfies))
    except:
        target_tokens = []
        generated_tokens = []
    
    if not target_tokens or not generated_tokens:
        return False, is_chemically_equivalent, False, 0.0
    
    # 计算token级别的相似度
    # 方法1: 最长公共子序列 (LCS)
    def lcs_length(seq1, seq2):
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    lcs_len = lcs_length(target_tokens, generated_tokens)
    max_len = max(len(target_tokens), len(generated_tokens))
    lcs_similarity = lcs_len / max_len if max_len > 0 else 0.0
    
    # 方法2: 前缀匹配
    common_prefix = 0
    min_len = min(len(target_tokens), len(generated_tokens))
    for i in range(min_len):
        if target_tokens[i] == generated_tokens[i]:
            common_prefix += 1
        else:
            break
    
    prefix_similarity = common_prefix / min_len if min_len > 0 else 0.0
    
    # 方法3: Jaccard相似度（token集合的交集/并集）
    target_set = set(target_tokens)
    generated_set = set(generated_tokens)
    intersection = len(target_set & generated_set)
    union = len(target_set | generated_set)
    jaccard_similarity = intersection / union if union > 0 else 0.0
    
    # 综合相似度分数
    similarity_score = (lcs_similarity * 0.4 + prefix_similarity * 0.4 + jaccard_similarity * 0.2)
    
    # 判断是否为部分匹配
    is_partial_match = similarity_score > 0.5 or prefix_similarity > 0.6
    
    return False, is_chemically_equivalent, is_partial_match, similarity_score


def generate_value_labels_for_gt_sequence(model, spectrum, input_ids, attention_mask, tokenizer, device):
    """
    为真实序列生成value标签：基于模型预测的累积正确性
    """
    batch_size, seq_len = input_ids.shape
    
    # 获取模型的action预测
    with torch.no_grad():
        output = model.forward_pretrain(spectrum, input_ids, attention_mask, return_value=False)
        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output
    
    # 生成value标签
    value_labels = torch.zeros(batch_size, seq_len, device=device)
    
    for b in range(batch_size):
        # 对于每个序列，从右往左计算累积正确性
        cumulative_correct = True
        for t in range(seq_len - 1, -1, -1):  # 从最后一个位置往前
            if t < seq_len - 1:  # 不是最后一个位置
                # 检查当前位置的预测是否正确
                predicted_token = torch.argmax(logits[b, t, :])
                actual_next_token = input_ids[b, t + 1]
                
                # 如果预测错误，则累积正确性变为False
                if predicted_token != actual_next_token:
                    cumulative_correct = False
            
            # 设置value标签
            value_labels[b, t] = 1.0 if cumulative_correct else 0.0
    
    return value_labels


def evaluate_value_accuracy(model, config, test_data_file, device, batch_size=32, num_samples=1000):
    """
    评估value head的准确率
    """
    print(f"\n🎯 正在评估Value准确率...")
    
    tokenizer = SelfiesTokenizer(max_len=config.max_len)
    
    test_dataset = RealSpectrumSelfiesDataset(
        data_file=test_data_file,
        tokenizer=tokenizer,
        max_len=config.max_len
    )
    
    # 限制评估样本数量以加快速度
    if num_samples > 0 and num_samples < len(test_dataset):
        # 随机采样指定数量的样本
        indices = torch.randperm(len(test_dataset))[:num_samples]
        test_subset = torch.utils.data.Subset(test_dataset, indices)
    else:
        test_subset = test_dataset
    
    test_loader = torch.utils.data.DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    model.eval()
    
    total_tokens = 0
    correct_predictions = 0
    total_samples = 0
    value_predictions_all = []
    value_labels_all = []
    
    print(f"开始评估 {len(test_subset):,} 个测试样本的value准确率...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="评估value准确率")):
            spectrums = batch['spectrum'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            batch_size_actual = spectrums.size(0)
            
            try:
                # 1. 生成真实value标签（基于模型预测的累积正确性）
                value_labels = generate_value_labels_for_gt_sequence(
                    model, spectrums, input_ids, attention_mask, tokenizer, device
                )
                
                # 2. 获取模型的value预测
                _, value_predictions = model.forward_pretrain(
                    spectrums, input_ids, attention_mask, return_value=True
                )
                
                # 3. 计算准确率
                value_preds_sigmoid = torch.sigmoid(value_predictions.squeeze(-1))  # (B, T)
                value_pred_binary = (value_preds_sigmoid > 0.5).float()
                
                # 创建mask，忽略pad tokens
                value_mask = (input_ids != tokenizer.pad_token_id).float()
                
                # 计算正确预测的token数量
                correct = (value_pred_binary == value_labels) * value_mask
                correct_predictions += correct.sum().item()
                total_tokens += value_mask.sum().item()
                total_samples += batch_size_actual
                
                # 收集预测和标签用于详细分析
                masked_predictions = value_preds_sigmoid * value_mask
                masked_labels = value_labels * value_mask
                
                for b in range(batch_size_actual):
                    seq_mask = value_mask[b]
                    seq_len = int(seq_mask.sum().item())
                    if seq_len > 0:
                        value_predictions_all.extend(masked_predictions[b, :seq_len].cpu().numpy())
                        value_labels_all.extend(masked_labels[b, :seq_len].cpu().numpy())
                
            except Exception as e:
                print(f"批次 {batch_idx} 处理失败: {e}")
                continue
            
            # 定期打印进度
            if (batch_idx + 1) % 10 == 0:
                current_acc = correct_predictions / total_tokens * 100 if total_tokens > 0 else 0
                print(f"  处理了 {total_samples:,} 个样本 - 当前Value准确率: {current_acc:.2f}%")
    
    # 计算最终结果
    value_accuracy = correct_predictions / total_tokens * 100 if total_tokens > 0 else 0
    
    # 详细分析
    value_predictions_all = np.array(value_predictions_all)
    value_labels_all = np.array(value_labels_all)
    
    # 计算各种统计指标
    positive_samples = np.sum(value_labels_all == 1)
    negative_samples = np.sum(value_labels_all == 0)
    
    # True/False Positives/Negatives
    predictions_binary = (value_predictions_all > 0.5).astype(int)
    tp = np.sum((predictions_binary == 1) & (value_labels_all == 1))
    tn = np.sum((predictions_binary == 0) & (value_labels_all == 0))
    fp = np.sum((predictions_binary == 1) & (value_labels_all == 0))
    fn = np.sum((predictions_binary == 0) & (value_labels_all == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 平均预测值
    avg_prediction = np.mean(value_predictions_all)
    avg_positive_prediction = np.mean(value_predictions_all[value_labels_all == 1]) if positive_samples > 0 else 0
    avg_negative_prediction = np.mean(value_predictions_all[value_labels_all == 0]) if negative_samples > 0 else 0
    
    print(f"\n🎯 Value Head评估结果:")
    print(f"  📊 总样本数: {total_samples:,}")
    print(f"  📊 总tokens数: {total_tokens:,}")
    print(f"  ✅ Value准确率: {value_accuracy:.2f}%")
    print(f"  📈 正样本数: {positive_samples:,} ({positive_samples/len(value_labels_all)*100:.1f}%)")
    print(f"  📉 负样本数: {negative_samples:,} ({negative_samples/len(value_labels_all)*100:.1f}%)")
    print(f"  🎯 精确率: {precision:.3f}")
    print(f"  🎯 召回率: {recall:.3f}")
    print(f"  🎯 F1分数: {f1_score:.3f}")
    print(f"  📊 平均预测值: {avg_prediction:.3f}")
    print(f"  📊 正样本平均预测: {avg_positive_prediction:.3f}")
    print(f"  📊 负样本平均预测: {avg_negative_prediction:.3f}")
    
    return {
        'total_samples': total_samples,
        'total_tokens': total_tokens,
        'correct_predictions': correct_predictions,
        'value_accuracy': value_accuracy,
        'positive_samples': positive_samples,
        'negative_samples': negative_samples,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'avg_prediction': avg_prediction,
        'avg_positive_prediction': avg_positive_prediction,
        'avg_negative_prediction': avg_negative_prediction,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate best model SELFIES exact match accuracy on test set")
    
    parser.add_argument("--model_path", type=str, 
                       default="./pretrained_selfies_transformer/best_model.pt",
                       help="Best model checkpoint path")
    parser.add_argument("--test_data_file", type=str,
                       default="/hy-tmp/MassEnv/DataLoader/test_spectrum_embeds_msg.pt",
                       help="Test data file path")
    
    parser.add_argument("--batch_size", type=int, default=512,
                       help="Batch size")
    parser.add_argument("--device", type=torch.device, default="cuda",
                       help="Device (auto/cuda/cpu)")
    parser.add_argument("--max_generation_length", type=int, default=120,
                       help="Maximum generation length")
    
    parser.add_argument("--k_predictions", type=int, default=1,
                       help="Number of SELFIES sequences to generate per spectrum (for diversity evaluation). If > 1, sampling will be used.")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature for generating multiple predictions. Higher values increase randomness. Only applies when k_predictions > 1. Set to 0 for argmax (greedy) sampling.")
    
    # Value评估相关参数
    parser.add_argument("--evaluate_value", action="store_true", default=True,
                       help="是否评估value head的准确率")
    parser.add_argument("--value_eval_samples", type=int, default=1000,
                       help="用于value评估的样本数量 (0表示使用全部测试集)")
    
    parser.add_argument("--save_results", action="store_true", default=True,
                       help="Save evaluation results")
    parser.add_argument("--results_dir", type=str, default="./evaluation_results",
                       help="Results save directory")
    parser.add_argument("--plot_distribution", action="store_true", default=True,
                       help="Plot length distribution")
    parser.add_argument("--analyze_badcase", action="store_true", default=True,
                       help="Perform detailed badcase analysis")
    parser.add_argument("--badcase_examples_per_type", type=int, default=5,
                       help="Number of examples to collect per error type")
    
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Auto-detected device: {device}")
    else:
        device = torch.device(args.device)
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
    
    # 新增：Value评估
    value_eval_results = None
    if args.evaluate_value:
        print(f"\n{'='*60}")
        print(f"Step 3: Evaluate Value Head accuracy")
        print(f"{'='*60}")
        
        value_eval_results = evaluate_value_accuracy(
            model, config, args.test_data_file, device, 
            batch_size=args.batch_size, num_samples=args.value_eval_samples
        )
    
    print(f"📊 Final results:")
    print(f"   - SELFIES exact match accuracy = {eval_results['exact_accuracy']:.2f}%")
    print(f"   - SELFIES chemical equivalents = {eval_results['chemical_accuracy']:.2f}%")
    print(f"   - SELFIES partial match accuracy = {eval_results['partial_accuracy']:.2f}%")
    print(f"   - Combined accuracy = {eval_results['combined_accuracy']:.2f}%")
    
    if value_eval_results:
        print(f"   - Value Head accuracy = {value_eval_results['value_accuracy']:.2f}%")
        print(f"   - Value Head F1 score = {value_eval_results['f1_score']:.3f}")

    # 进行badcase分析
    if args.analyze_badcase:
        badcase_examples, error_counts = collect_badcase_examples(eval_results['results'], args.badcase_examples_per_type)
        print_badcase_analysis(badcase_examples, error_counts, eval_results['total_samples'] - eval_results['exact_matches'])
    else:
        badcase_examples, error_counts = None, None
    
    if args.save_results:
        save_evaluation_results(
            eval_results, stats, args.model_path, args.test_data_file, args.results_dir,
            args.k_predictions, args.temperature, badcase_examples, error_counts, value_eval_results
        )
        if args.analyze_badcase and badcase_examples:
            save_badcase_analysis(badcase_examples, error_counts, args.results_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))

    print(f"\n🎉 Evaluation completed!")


if __name__ == "__main__":
    print(torch.cuda.is_available())
    main() 