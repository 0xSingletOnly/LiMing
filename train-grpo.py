import os
import re
import torch # Added for torch.bfloat16, though string "bfloat16" often works

from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig
from peft import LoraConfig

# Ensure HF_TOKEN is set in your environment
login(token=os.environ.get("HF_TOKEN"), add_to_git_credential=True)

# MODEL SETUP
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DATASET SETUP
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset = load_dataset(dataset_id, split="train")
dataset = dataset.shuffle().select(range(1000))

def generate_r1_prompt(example):
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
    return {
        "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True),
        "target": target_val,  
        "nums": numbers_list
    }

dataset = dataset.map(generate_r1_prompt)

train_test_split = dataset.train_test_split(test_size=0.1, seed=42) # Added seed
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# TRAINING SETUP
# --- Constants ---
# Regex for the full <think>...</think><answer>...</answer> format
FORMAT_REGEX_PATTERN = r"<think>((?:(?!<\/think>).)*)<\/think>\s*<answer>((?:(?!<\/answer>).)*)<\/answer>"
FORMAT_REGEX = re.compile(FORMAT_REGEX_PATTERN, re.DOTALL)

# More flexible regex to extract equations - looks for patterns after </think> or within <answer> tags
EQUATION_EXTRACTION_PATTERNS = [
    r"<answer>((?:(?!<\/answer>).)*)<\/answer>",  # Standard <answer> tag
    r"<\/think>\s*([^<]*(?:\([^)]+\)[^<]*)*=\s*\d+(?:\.\d+)?)",  # After </think>, equation with =
    r"<\/think>\s*([^<]*(?:\([^)]+\)[^<]*)*)",  # After </think>, any mathematical expression
]
EQUATION_REGEXES = [re.compile(pattern, re.DOTALL) for pattern in EQUATION_EXTRACTION_PATTERNS]

# Regex for allowed characters in an equation (evaluable part, no '=')
ALLOWED_EQUATION_CHARS_PATTERN = r'^[\d+\-*/().\s]+$' # Removed '='
ALLOWED_EQUATION_CHARS_REGEX = re.compile(ALLOWED_EQUATION_CHARS_PATTERN)

# Tolerance for float comparisons
FLOAT_COMPARISON_TOLERANCE = 1e-5


def reward_func(
    completions: list[str],
    target: list[str],  # List of target strings
    nums: list[list[str]],  # List of (list of available number strings)
    **kwargs
) -> list[float]:
    rewards = []
    for completion_text, target_str, current_available_nums_str in zip(completions, target, nums):
        try:
            current_reward = 0.0
            has_correct_format = False
            equation_content = None
            
            completion_text_stripped = completion_text.strip()
            format_match = FORMAT_REGEX.search(completion_text_stripped)
            if format_match and len(format_match.groups()) == 2:
                has_correct_format = True
                equation_content = format_match.group(2).strip()
                print(f"Debug: Perfect format detected, equation content: '{equation_content}' for target {target_str}")
            else:
                for regex in EQUATION_REGEXES:
                    match = regex.search(completion_text_stripped)
                    if match:
                        raw_extracted_content = match.group(1)
                        if raw_extracted_content: # Ensure group(1) actually captured something
                            equation_content = raw_extracted_content.strip()
                            print(f"Debug: Imperfect format, extracted: '{equation_content}' for target {target_str}")
                            break
            
            if not equation_content:
                print(f"Debug: No equation found in: {completion_text_stripped[:50] + '...' + completion_text_stripped[-50:]}... for target {target_str}")
                rewards.append(current_reward)
                continue

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
                math_expression = equation_content # Use the full (stripped) content if no specific line found
            
            expression_part = ""
            if '=' in math_expression:
                expression_part = math_expression.split('=')[0].strip()
            else:
                expression_part = math_expression.strip()
            
            expression_part = re.sub(r'[.]+$', '', expression_part).strip() # Remove trailing periods
            
            if not expression_part:
                print(f"Debug: Expression part is empty after cleaning: '{math_expression}' for target {target_str}")
                rewards.append(current_reward)
                continue

            print(f"Debug: Expression to evaluate: '{expression_part}' for target {target_str}")

            if not ALLOWED_EQUATION_CHARS_REGEX.match(expression_part.replace(' ', '')):
                print(f"Debug: Expression contains forbidden characters: '{expression_part}' for target {target_str}")
                rewards.append(current_reward)
                continue

            try:
                expected_numbers_int = sorted([int(n) for n in current_available_nums_str])
                
                # Extract numbers from expression_part
                temp_expr = expression_part
                for op in ['+', '-', '*', '/', '(', ')']: # Replace ops with space for splitting
                    temp_expr = temp_expr.replace(op, ' ')
                
                used_numbers_str_list = []
                for item in temp_expr.split(' '):
                    item_s = item.strip()
                    if item_s.isdigit():
                         used_numbers_str_list.append(item_s)
                
                used_numbers_int = sorted([int(n) for n in used_numbers_str_list])

            except ValueError as e:
                print(f"Debug: Error parsing numbers ({e}) for expr '{expression_part}', available '{current_available_nums_str}' for target {target_str}")
                rewards.append(current_reward)
                continue

            if used_numbers_int != expected_numbers_int:
                print(f"Debug: Number usage mismatch. Used: {used_numbers_int}, Expected: {expected_numbers_int} (from {current_available_nums_str}) in '{expression_part}' for target {target_str}")
                rewards.append(current_reward)
                continue

            try:
                target_val_float = float(target_str)
                result = eval(expression_part, {"__builtins__": {}}, {}) 
                
                print(f"Debug: Eval: '{expression_part}' = {result}, Target: {target_val_float}, Format OK: {has_correct_format}")

                if abs(float(result) - target_val_float) < FLOAT_COMPARISON_TOLERANCE:
                    if has_correct_format:
                        current_reward = 1.0
                        print(f"Debug: PERFECT! Correct format + correct answer = reward of 1.0 for target {target_str}")
                    else:
                        current_reward = 0.6
                        print(f"Debug: GOOD! Wrong format but correct answer = reward of 0.6 for target {target_str}")
                else:
                    if has_correct_format:
                        current_reward = 0.3
                        print(f"Debug: PARTIAL! Correct format but wrong answer ({result} vs {target_val_float}) = reward of 0.3 for target {target_str}")
                    else:
                        current_reward = 0.0
                        print(f"Debug: BAD! Wrong format + wrong answer = reward of 0.0 for target {target_str}")
            
            except ZeroDivisionError:
                print(f"Debug: ZeroDivisionError for '{expression_part}' for target {target_str}")
                current_reward = 0.1 if has_correct_format else 0.0
            except SyntaxError:
                print(f"Debug: SyntaxError for '{expression_part}' for target {target_str}")
                current_reward = 0.1 if has_correct_format else 0.0
            except Exception as eval_e:
                print(f"Debug: Error evaluating expression '{expression_part}': {eval_e} for target {target_str}")
                current_reward = 0.1 if has_correct_format else 0.0
            
            rewards.append(current_reward)

        except Exception as e:
            print(f"Debug: Unexpected error in reward_func: {e} for completion {completion_text_stripped[:50] + '...' + completion_text_stripped[-50:]}")
            rewards.append(0.0)
    return rewards


model_config = ModelConfig(
    model_name_or_path=model_name,
    torch_dtype="bfloat16", # string is fine for ModelConfig
    use_peft=True,          # GRPOTrainer will handle PEFT model setup
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    fan_in_fan_out=False,
    modules_to_save=None,
)

training_args = GRPOConfig(
    output_dir="qwen3-0.6b-grpo-countdown-tasks",
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    logging_steps=5,
    report_to="tensorboard",
    max_steps=500,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2, 
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    bf16=True,
    max_prompt_length=256,    
    max_completion_length=1024, 
    num_generations=2,
    beta=0.001,
)

trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
)

# Start training
print("Starting training...")
trainer.train()
print("Training finished. Saving model...")
trainer.save_model(training_args.output_dir)
print(f"Model saved to {training_args.output_dir}")