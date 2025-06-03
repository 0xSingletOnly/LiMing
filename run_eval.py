import os
import re
import json
import torch
import random
from datetime import datetime
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# Ensure HF_TOKEN is set in your environment
login(token=os.environ.get("HF_TOKEN"), add_to_git_credential=True)

# Constants (same as in training script)
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
    """Evaluate a single response using the same logic as the reward function"""
    try:
        current_reward = 0.0
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
        if format_match and len(format_match.groups()) == 2:
            has_correct_format = True
            equation_content = format_match.group(2).strip()
            evaluation_details["has_correct_format"] = True
        else:
            for regex in EQUATION_REGEXES:
                match = regex.search(completion_text_stripped)
                if match:
                    raw_extracted_content = match.group(1)
                    if raw_extracted_content:
                        equation_content = raw_extracted_content.strip()
                        break
        
        if not equation_content:
            evaluation_details["evaluation_error"] = "No equation found"
            return current_reward, evaluation_details

        evaluation_details["equation_found"] = True
        
        equation_lines = equation_content.split('\n')
        math_expression = None
        for line in equation_lines:
            line = line.strip()
            if not line:
                continue
            if re.search(r'[\d+\-*/()]', line):
                math_expression = line
                break
        
        if not math_expression:
            math_expression = equation_content
        
        expression_part = ""
        if '=' in math_expression:
            expression_part = math_expression.split('=')[0].strip()
        else:
            expression_part = math_expression.strip()
        
        expression_part = re.sub(r'[.]+$', '', expression_part).strip()
        
        if not expression_part:
            evaluation_details["evaluation_error"] = "Expression part is empty after cleaning"
            return current_reward, evaluation_details

        if not ALLOWED_EQUATION_CHARS_REGEX.match(expression_part.replace(' ', '')):
            evaluation_details["evaluation_error"] = "Expression contains forbidden characters"
            return current_reward, evaluation_details

        try:
            expected_numbers_int = sorted([int(n) for n in current_available_nums_str])
            
            temp_expr = expression_part
            for op in ['+', '-', '*', '/', '(', ')']:
                temp_expr = temp_expr.replace(op, ' ')
            
            used_numbers_str_list = []
            for item in temp_expr.split(' '):
                item_s = item.strip()
                if item_s.isdigit():
                     used_numbers_str_list.append(item_s)
            
            used_numbers_int = sorted([int(n) for n in used_numbers_str_list])

        except ValueError as e:
            evaluation_details["evaluation_error"] = f"Error parsing numbers: {e}"
            return current_reward, evaluation_details

        if used_numbers_int != expected_numbers_int:
            evaluation_details["evaluation_error"] = f"Number usage mismatch. Used: {used_numbers_int}, Expected: {expected_numbers_int}"
            return current_reward, evaluation_details

        evaluation_details["numbers_used_correctly"] = True

        try:
            target_val_float = float(target_str)
            result = eval(expression_part, {"__builtins__": {}}, {})
            
            if abs(float(result) - target_val_float) < FLOAT_COMPARISON_TOLERANCE:
                evaluation_details["answer_correct"] = True
                if has_correct_format:
                    current_reward = 1.0
                else:
                    current_reward = 0.6
            else:
                if has_correct_format:
                    current_reward = 0.3
                else:
                    current_reward = 0.0
        
        except ZeroDivisionError:
            evaluation_details["evaluation_error"] = "ZeroDivisionError"
            current_reward = 0.1 if has_correct_format else 0.0
        except SyntaxError:
            evaluation_details["evaluation_error"] = "SyntaxError"
            current_reward = 0.1 if has_correct_format else 0.0
        except Exception as eval_e:
            evaluation_details["evaluation_error"] = f"Error evaluating expression: {eval_e}"
            current_reward = 0.1 if has_correct_format else 0.0
        
        return current_reward, evaluation_details

    except Exception as e:
        evaluation_details["evaluation_error"] = f"Unexpected error: {e}"
        return 0.0, evaluation_details

def load_models_and_tokenizer(base_model_name, finetuned_model_path):
    """Load base model, fine-tuned model, and tokenizer"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
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
    
    return base_model, finetuned_model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    """Generate response from model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
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
    
    # Extract only the generated part (exclude the input prompt)
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response

def run_evaluation():
    """Main evaluation function"""
    # Configuration
    BASE_MODEL_NAME = "Qwen/Qwen3-0.6B"
    FINETUNED_MODEL_PATH = "qwen3-0.6b-grpo-countdown-tasks"  # Path to your fine-tuned model
    DATASET_ID = "Jiayi-Pan/Countdown-Tasks-3to4"
    NUM_SAMPLES = 50
    
    print("Starting evaluation...")
    print(f"Base model: {BASE_MODEL_NAME}")
    print(f"Fine-tuned model: {FINETUNED_MODEL_PATH}")
    print(f"Number of samples: {NUM_SAMPLES}")
    
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
    
    # Run evaluation
    results = {
        "metadata": {
            "base_model": BASE_MODEL_NAME,
            "finetuned_model": FINETUNED_MODEL_PATH,
            "dataset": DATASET_ID,
            "num_samples": NUM_SAMPLES,
            "timestamp": datetime.now().isoformat(),
            "sample_indices": sample_indices
        },
        "samples": [],
        "summary": {
            "base_model": {
                "total_reward": 0.0,
                "perfect_answers": 0,
                "correct_format_count": 0,
                "correct_answer_count": 0
            },
            "finetuned_model": {
                "total_reward": 0.0,
                "perfect_answers": 0,
                "correct_format_count": 0,
                "correct_answer_count": 0
            }
        }
    }
    
    print("Running evaluation on samples...")
    for i, sample in enumerate(eval_samples):
        print(f"Processing sample {i+1}/{NUM_SAMPLES}...")
        
        # Generate prompt
        prompt = generate_r1_prompt(sample, tokenizer)
        
        # Generate responses
        print("  Generating base model response...")
        base_response = generate_response(base_model, tokenizer, prompt)
        
        print("  Generating fine-tuned model response...")
        finetuned_response = generate_response(finetuned_model, tokenizer, prompt)
        
        # Evaluate responses
        base_reward, base_details = evaluate_response(base_response, str(sample["target"]), sample["nums"])
        finetuned_reward, finetuned_details = evaluate_response(finetuned_response, str(sample["target"]), sample["nums"])
        
        # Store results
        sample_result = {
            "sample_index": sample_indices[i],
            "numbers": sample["nums"],
            "target": sample["target"],
            "prompt": prompt,
            "base_model": {
                "response": base_response,
                "reward": base_reward,
                "evaluation_details": base_details
            },
            "finetuned_model": {
                "response": finetuned_response,
                "reward": finetuned_reward,
                "evaluation_details": finetuned_details
            }
        }
        
        results["samples"].append(sample_result)
        
        # Update summary statistics
        results["summary"]["base_model"]["total_reward"] += base_reward
        results["summary"]["finetuned_model"]["total_reward"] += finetuned_reward
        
        if base_reward == 1.0:
            results["summary"]["base_model"]["perfect_answers"] += 1
        if finetuned_reward == 1.0:
            results["summary"]["finetuned_model"]["perfect_answers"] += 1
            
        if base_details["has_correct_format"]:
            results["summary"]["base_model"]["correct_format_count"] += 1
        if finetuned_details["has_correct_format"]:
            results["summary"]["finetuned_model"]["correct_format_count"] += 1
            
        if base_details["answer_correct"]:
            results["summary"]["base_model"]["correct_answer_count"] += 1
        if finetuned_details["answer_correct"]:
            results["summary"]["finetuned_model"]["correct_answer_count"] += 1
    
    # Calculate averages
    results["summary"]["base_model"]["average_reward"] = results["summary"]["base_model"]["total_reward"] / NUM_SAMPLES
    results["summary"]["finetuned_model"]["average_reward"] = results["summary"]["finetuned_model"]["total_reward"] / NUM_SAMPLES
    
    results["summary"]["base_model"]["perfect_answer_rate"] = results["summary"]["base_model"]["perfect_answers"] / NUM_SAMPLES
    results["summary"]["finetuned_model"]["perfect_answer_rate"] = results["summary"]["finetuned_model"]["perfect_answers"] / NUM_SAMPLES
    
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
    print(f"  Perfect Answers: {results['summary']['base_model']['perfect_answers']}/{NUM_SAMPLES} ({results['summary']['base_model']['perfect_answer_rate']:.1%})")
    print(f"  Correct Format: {results['summary']['base_model']['correct_format_count']}/{NUM_SAMPLES} ({results['summary']['base_model']['correct_format_rate']:.1%})")
    print(f"  Correct Answer: {results['summary']['base_model']['correct_answer_count']}/{NUM_SAMPLES} ({results['summary']['base_model']['correct_answer_rate']:.1%})")
    
    print(f"\nFine-tuned Model:")
    print(f"  Average Reward: {results['summary']['finetuned_model']['average_reward']:.3f}")
    print(f"  Perfect Answers: {results['summary']['finetuned_model']['perfect_answers']}/{NUM_SAMPLES} ({results['summary']['finetuned_model']['perfect_answer_rate']:.1%})")
    print(f"  Correct Format: {results['summary']['finetuned_model']['correct_format_count']}/{NUM_SAMPLES} ({results['summary']['finetuned_model']['correct_format_rate']:.1%})")
    print(f"  Correct Answer: {results['summary']['finetuned_model']['correct_answer_count']}/{NUM_SAMPLES} ({results['summary']['finetuned_model']['correct_answer_rate']:.1%})")
    
    improvement_reward = results['summary']['finetuned_model']['average_reward'] - results['summary']['base_model']['average_reward']
    improvement_perfect = results['summary']['finetuned_model']['perfect_answer_rate'] - results['summary']['base_model']['perfect_answer_rate']
    
    print(f"\nImprovement:")
    print(f"  Reward: {improvement_reward:+.3f}")
    print(f"  Perfect Answer Rate: {improvement_perfect:+.1%}")
    print("="*60)

if __name__ == "__main__":
    run_evaluation()