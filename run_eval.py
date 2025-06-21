import os
import re
import json
import torch
import random
import time
from datetime import datetime, timedelta
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Ensure HF_TOKEN is set in your environment
login(token=os.environ.get("HF_TOKEN"), add_to_git_credential=True)

# Constants (updated to match new reward function)
FORMAT_REGEX_PATTERN = r"<think>((?:(?!<\/think>).)*)<\/think>\s*<answer>((?:(?!<\/answer>).)*)<\/answer>"
FORMAT_REGEX = re.compile(FORMAT_REGEX_PATTERN, re.DOTALL)

EQUATION_EXTRACTION_PATTERNS = [
    r"<answer>((?:(?!<\/answer>).)*)<\/answer>",
    r"<\/think>\s*([^<]*(?:\([^)]+\)[^<]*)*=\s*\d+(?:\.\d+)?)",
    r"<\/think>\s*([^<]*(?:\([^)]+\)[^<]*)*)",
]
EQUATION_REGEXES = [re.compile(pattern, re.DOTALL) for pattern in EQUATION_EXTRACTION_PATTERNS]

ALLOWED_EQUATION_CHARS_PATTERN = r'^[\d+\-*/().\s]+$'
ALLOWED_EQUATION_CHARS_REGEX = re.compile(ALLOWED_EQUATION_CHARS_PATTERN)

FLOAT_COMPARISON_TOLERANCE = 1e-5

def generate_r1_prompt(example, tokenizer):
    """Generate the same prompt format as used in training"""
    numbers_list = example["nums"]
    target_val = example["target"]
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a math expert. You will first reason carefully about the problem, then provide the user with the answer. Your entire response MUST start directly with the <think> tag and follow the format: <think>Your reasoning here.</think><answer>Your equation here.</answer>"
        },
        {
            "role": "user",
            "content": f"Given the numbers {numbers_list} and the target number {target_val}, please provide a solution to reach the target number using the four basic arithmetic operations: addition, subtraction, multiplication, and division (+, -, *, /). You can use each number only once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
        },
    ]
    return tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True)

def evaluate_response(completion_text, target_str, current_available_nums_str):
    """Evaluate a single response using the updated reward function logic"""
    try:
        current_reward = 0.0  # Default to zero
        has_correct_format = False
        equation_content = None
        evaluation_details = {
            "has_correct_format": False,
            "equation_found": False,
            "numbers_used_correctly": False,
            "answer_correct": False,
            "evaluation_error": None
        }
        
        completion_text_stripped = completion_text.strip()
        format_match = FORMAT_REGEX.search(completion_text_stripped)
        
        # 1. First, check for the correct format
        if format_match and len(format_match.groups()) == 2:
            has_correct_format = True
            equation_content = format_match.group(2).strip()
            evaluation_details["has_correct_format"] = True
        else:
            # Fallback to extract equation from imperfect format
            for regex in EQUATION_REGEXES:
                match = regex.search(completion_text_stripped)
                if match and match.group(1):
                    equation_content = match.group(1).strip()
                    break
        
        if not equation_content:
            evaluation_details["evaluation_error"] = "No equation found"
            return current_reward, evaluation_details

        evaluation_details["equation_found"] = True
        
        # 2. Extract and clean the mathematical expression
        expression_part = equation_content.split('=')[0].strip()
        if not expression_part or not ALLOWED_EQUATION_CHARS_REGEX.match(expression_part.replace(' ', '')):
            evaluation_details["evaluation_error"] = "Expression part is empty or contains forbidden characters"
            return current_reward, evaluation_details

        # 3. Verify number usage
        try:
            expected_numbers_int = sorted([int(n) for n in current_available_nums_str])
            temp_expr = expression_part.replace('(', ' ').replace(')', ' ')
            used_numbers_str_list = [item for item in re.split(r'[+\-*/\s]', temp_expr) if item.isdigit()]
            used_numbers_int = sorted([int(n) for n in used_numbers_str_list])

            if used_numbers_int != expected_numbers_int:
                evaluation_details["evaluation_error"] = f"Number usage mismatch. Used: {used_numbers_int}, Expected: {expected_numbers_int}"
                return current_reward, evaluation_details  # Penalty for using wrong numbers
                
        except (ValueError, IndexError) as e:
            evaluation_details["evaluation_error"] = f"Error parsing numbers: {e}"
            return current_reward, evaluation_details

        evaluation_details["numbers_used_correctly"] = True

        # 4. Evaluate the expression and assign reward ONLY if correct
        try:
            target_val_float = float(target_str)
            result = eval(expression_part, {"__builtins__": {}}, {})
            
            if abs(float(result) - target_val_float) < FLOAT_COMPARISON_TOLERANCE:
                evaluation_details["answer_correct"] = True
                # The answer is correct. Now check format for bonus.
                if has_correct_format:
                    current_reward = 1.0  # Perfect: Correct answer AND format
                else:
                    current_reward = 0.8  # Good: Correct answer, wrong format
            # If the answer is wrong, the reward remains 0.0, regardless of format.
            
        except (SyntaxError, ZeroDivisionError, NameError, TypeError) as e:
            # Any evaluation error results in zero reward
            evaluation_details["evaluation_error"] = f"Error evaluating expression: {e}"
            current_reward = 0.0
        
        return current_reward, evaluation_details

    except Exception as e:
        evaluation_details["evaluation_error"] = f"Unexpected error: {e}"
        return 0.0, evaluation_details

def load_models_and_tokenizer(base_model_name, finetuned_model_path):
    """Load base model, fine-tuned model, and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    
    print("Loading fine-tuned model...")
    finetuned_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=True,
    )
    finetuned_model = PeftModel.from_pretrained(finetuned_model, finetuned_model_path)
    print(f"Fine-tuned model type: {type(finetuned_model)}")
    
    return base_model, finetuned_model, tokenizer

def generate_response_batch(model, tokenizer, prompts, max_new_tokens=1024, batch_size=4):
    """Generate responses from model in batches"""
    all_responses = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch with padding
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract generated parts for each sample in batch
        batch_responses = []
        for j, output in enumerate(outputs):
            # Get the length of the input for this sample
            input_length = inputs['input_ids'][j].ne(tokenizer.pad_token_id).sum().item()
            generated_tokens = output[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            batch_responses.append(response)
        
        all_responses.extend(batch_responses)
        
        # Clear GPU cache after each batch
        torch.cuda.empty_cache()
    
    return all_responses

def evaluate_batch_responses(batch_samples, batch_prompts, base_responses, finetuned_responses, sample_indices_batch, results):
    """Evaluate responses for a batch and update results"""
    for i, sample in enumerate(batch_samples):
        # Evaluate responses
        base_reward, base_details = evaluate_response(base_responses[i], str(sample["target"]), sample["nums"])
        finetuned_reward, finetuned_details = evaluate_response(finetuned_responses[i], str(sample["target"]), sample["nums"])
        
        # Store results
        sample_result = {
            "sample_index": sample_indices_batch[i],
            "numbers": sample["nums"],
            "target": sample["target"],
            "prompt": batch_prompts[i],
            "base_model": {
                "response": base_responses[i],
                "reward": base_reward,
                "evaluation_details": base_details
            },
            "finetuned_model": {
                "response": finetuned_responses[i],
                "reward": finetuned_reward,
                "evaluation_details": finetuned_details
            }
        }
        
        results["samples"].append(sample_result)
        
        # Update summary statistics
        results["summary"]["base_model"]["total_reward"] += base_reward
        results["summary"]["finetuned_model"]["total_reward"] += finetuned_reward
        
        # Update perfect answers (reward = 1.0)
        if base_reward == 1.0:
            results["summary"]["base_model"]["perfect_answers"] += 1
        if finetuned_reward == 1.0:
            results["summary"]["finetuned_model"]["perfect_answers"] += 1
        
        # Update good answers (reward >= 0.8, which includes both 0.8 and 1.0)
        if base_reward >= 0.8:
            results["summary"]["base_model"]["good_answers"] += 1
        if finetuned_reward >= 0.8:
            results["summary"]["finetuned_model"]["good_answers"] += 1
            
        if base_details["has_correct_format"]:
            results["summary"]["base_model"]["correct_format_count"] += 1
        if finetuned_details["has_correct_format"]:
            results["summary"]["finetuned_model"]["correct_format_count"] += 1
            
        if base_details["answer_correct"]:
            results["summary"]["base_model"]["correct_answer_count"] += 1
        if finetuned_details["answer_correct"]:
            results["summary"]["finetuned_model"]["correct_answer_count"] += 1

def run_evaluation():
    """Main evaluation function"""
    # Configuration
    BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
    FINETUNED_MODEL_PATH = "qwen3-0.6b-grpo-countdown-tasks-v1"  # Path to your fine-tuned model
    DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
    NUM_SAMPLES = 100
    BATCH_SIZE = 32  # Adjust based on your GPU memory
    
    print("Starting evaluation...")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Fine-tuned model: {FINETUNED_MODEL_PATH}")
    print(f"Number of samples: {NUM_SAMPLES}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(DATASET_ID, split="train")
    
    # Select random samples
    total_samples = len(dataset)
    sample_indices = random.sample(range(total_samples), NUM_SAMPLES)
    eval_samples = dataset.select(sample_indices)
    
    print(f"Selected {NUM_SAMPLES} random samples from {total_samples} total samples")
    
    # Load models
    base_model, finetuned_model, tokenizer = load_models_and_tokenizer(BASE_MODEL_NAME, FINETUNED_MODEL_PATH)
    
    # Initialize results structure
    results = {
        "metadata": {
            "base_model": BASE_MODEL_NAME,
            "finetuned_model": FINETUNED_MODEL_PATH,
            "dataset": DATASET_ID,
            "num_samples": NUM_SAMPLES,
            "batch_size": BATCH_SIZE,
            "timestamp": datetime.now().isoformat(),
            "sample_indices": sample_indices
        },
        "samples": [],
        "summary": {
            "base_model": {
                "total_reward": 0.0,
                "perfect_answers": 0,
                "good_answers": 0,  # Added for reward >= 0.8
                "correct_format_count": 0,
                "correct_answer_count": 0
            },
            "finetuned_model": {
                "total_reward": 0.0,
                "perfect_answers": 0,
                "good_answers": 0,  # Added for reward >= 0.8
                "correct_format_count": 0,
                "correct_answer_count": 0
            }
        }
    }
    
    # Process samples in batches
    print("="*60)
    print("STARTING BATCH-WISE EVALUATION")
    print("="*60)
    
    eval_start_time = time.time()
    total_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in range(0, NUM_SAMPLES, BATCH_SIZE):
        batch_start_time = time.time()
        batch_num = batch_idx // BATCH_SIZE + 1
        
        # Get current batch
        batch_end = min(batch_idx + BATCH_SIZE, NUM_SAMPLES)
        batch_samples = eval_samples.select(range(batch_idx, batch_end))
        batch_sample_indices = sample_indices[batch_idx:batch_end]
        
        print(f"\nBatch {batch_num}/{total_batches}: Processing samples {batch_idx+1}-{batch_end}")
        
        # Prepare prompts for this batch
        batch_prompts = []
        for sample in batch_samples:
            prompt = generate_r1_prompt(sample, tokenizer)
            batch_prompts.append(prompt)
        
        # Generate responses from base model
        print(f"  Generating base model responses...")
        base_gen_start = time.time()
        base_responses = generate_response_batch(base_model, tokenizer, batch_prompts, batch_size=BATCH_SIZE)
        base_gen_time = time.time() - base_gen_start
        
        # Generate responses from finetuned model
        print(f"  Generating finetuned model responses...")
        ft_gen_start = time.time()
        finetuned_responses = generate_response_batch(finetuned_model, tokenizer, batch_prompts, batch_size=BATCH_SIZE)
        ft_gen_time = time.time() - ft_gen_start
        
        # Evaluate responses for this batch
        print(f"  Evaluating responses...")
        eval_start = time.time()
        evaluate_batch_responses(batch_samples, batch_prompts, base_responses, finetuned_responses, batch_sample_indices, results)
        eval_time = time.time() - eval_start
        
        # Calculate progress and timing
        batch_time = time.time() - batch_start_time
        elapsed_time = time.time() - eval_start_time
        avg_batch_time = elapsed_time / batch_num
        remaining_batches = total_batches - batch_num
        eta = remaining_batches * avg_batch_time
        
        # Current running statistics
        samples_processed = len(results["samples"])
        current_base_avg = results["summary"]["base_model"]["total_reward"] / samples_processed if samples_processed > 0 else 0
        current_ft_avg = results["summary"]["finetuned_model"]["total_reward"] / samples_processed if samples_processed > 0 else 0
        current_base_perfect = results["summary"]["base_model"]["perfect_answers"] / samples_processed if samples_processed > 0 else 0
        current_ft_perfect = results["summary"]["finetuned_model"]["perfect_answers"] / samples_processed if samples_processed > 0 else 0
        current_base_good = results["summary"]["base_model"]["good_answers"] / samples_processed if samples_processed > 0 else 0
        current_ft_good = results["summary"]["finetuned_model"]["good_answers"] / samples_processed if samples_processed > 0 else 0
        
        print(f"  Batch completed in {batch_time:.1f}s (Gen: {base_gen_time:.1f}s + {ft_gen_time:.1f}s, Eval: {eval_time:.1f}s)")
        print(f"  Running averages - Base: {current_base_avg:.3f} (Perfect: {current_base_perfect:.1%}, Good: {current_base_good:.1%})")
        print(f"                     FT: {current_ft_avg:.3f} (Perfect: {current_ft_perfect:.1%}, Good: {current_ft_good:.1%})")
        print(f"  Progress: {batch_num}/{total_batches} | Elapsed: {elapsed_time:.1f}s | ETA: {eta:.1f}s")
    
    total_eval_time = time.time() - eval_start_time
    print(f"\nTotal evaluation time: {total_eval_time:.1f}s")
    print("="*60)
    
    # Finalize results
    results["metadata"]["timing"] = {
        "total_time": total_eval_time,
        "average_time_per_batch": total_eval_time / total_batches
    }
    
    results["summary"]["base_model"]["average_reward"] = results["summary"]["base_model"]["total_reward"] / NUM_SAMPLES
    results["summary"]["finetuned_model"]["average_reward"] = results["summary"]["finetuned_model"]["total_reward"] / NUM_SAMPLES
    
    results["summary"]["base_model"]["perfect_answer_rate"] = results["summary"]["base_model"]["perfect_answers"] / NUM_SAMPLES
    results["summary"]["finetuned_model"]["perfect_answer_rate"] = results["summary"]["finetuned_model"]["perfect_answers"] / NUM_SAMPLES
    
    results["summary"]["base_model"]["good_answer_rate"] = results["summary"]["base_model"]["good_answers"] / NUM_SAMPLES
    results["summary"]["finetuned_model"]["good_answer_rate"] = results["summary"]["finetuned_model"]["good_answers"] / NUM_SAMPLES
    
    results["summary"]["base_model"]["correct_format_rate"] = results["summary"]["base_model"]["correct_format_count"] / NUM_SAMPLES
    results["summary"]["finetuned_model"]["correct_format_rate"] = results["summary"]["finetuned_model"]["correct_format_count"] / NUM_SAMPLES
    
    results["summary"]["base_model"]["correct_answer_rate"] = results["summary"]["base_model"]["correct_answer_count"] / NUM_SAMPLES
    results["summary"]["finetuned_model"]["correct_answer_rate"] = results["summary"]["finetuned_model"]["correct_answer_count"] / NUM_SAMPLES
    
    # Save results
    output_filename = f"countdown_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nEvaluation completed! Results saved to {output_filename}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Base Model ({BASE_MODEL_NAME}):")
    print(f"  Average Reward: {results['summary']['base_model']['average_reward']:.3f}")
    print(f"  Perfect Answers (1.0): {results['summary']['base_model']['perfect_answers']}/{NUM_SAMPLES} ({results['summary']['base_model']['perfect_answer_rate']:.1%})")
    print(f"  Good Answers (≥0.8): {results['summary']['base_model']['good_answers']}/{NUM_SAMPLES} ({results['summary']['base_model']['good_answer_rate']:.1%})")
    print(f"  Correct Format: {results['summary']['base_model']['correct_format_count']}/{NUM_SAMPLES} ({results['summary']['base_model']['correct_format_rate']:.1%})")
    print(f"  Correct Answer: {results['summary']['base_model']['correct_answer_count']}/{NUM_SAMPLES} ({results['summary']['base_model']['correct_answer_rate']:.1%})")
    
    print(f"\nFine-tuned Model:")
    print(f"  Average Reward: {results['summary']['finetuned_model']['average_reward']:.3f}")
    print(f"  Perfect Answers (1.0): {results['summary']['finetuned_model']['perfect_answers']}/{NUM_SAMPLES} ({results['summary']['finetuned_model']['perfect_answer_rate']:.1%})")
    print(f"  Good Answers (≥0.8): {results['summary']['finetuned_model']['good_answers']}/{NUM_SAMPLES} ({results['summary']['finetuned_model']['good_answer_rate']:.1%})")
    print(f"  Correct Format: {results['summary']['finetuned_model']['correct_format_count']}/{NUM_SAMPLES} ({results['summary']['finetuned_model']['correct_format_rate']:.1%})")
    print(f"  Correct Answer: {results['summary']['finetuned_model']['correct_answer_count']}/{NUM_SAMPLES} ({results['summary']['finetuned_model']['correct_answer_rate']:.1%})")
    
    improvement_reward = results['summary']['finetuned_model']['average_reward'] - results['summary']['base_model']['average_reward']
    improvement_perfect = results['summary']['finetuned_model']['perfect_answer_rate'] - results['summary']['base_model']['perfect_answer_rate']
    improvement_good = results['summary']['finetuned_model']['good_answer_rate'] - results['summary']['base_model']['good_answer_rate']
    
    print(f"\nImprovement:")
    print(f"  Reward: {improvement_reward:+.3f}")
    print(f"  Perfect Answer Rate: {improvement_perfect:+.1%}")
    print(f"  Good Answer Rate: {improvement_good:+.1%}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()