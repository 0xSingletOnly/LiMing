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
model_name = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DATASET SETUP
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset = load_dataset(dataset_id, split="train")
dataset = dataset.select(range(10000))

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
        {
            "role": "assistant",
            "content": "Let me solve this step by step.\n<think>"
        },
    ]
    return {
        "prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, add_generation_prompt=True),
        "target": target_val,  
        "nums": numbers_list
    }

dataset = dataset.map(generate_r1_prompt)

train_test_split = dataset.shuffle().train_test_split(test_size=0.1) 
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
    target: list[str],
    nums: list[list[str]],
    **kwargs
) -> list[float]:
    rewards = []
    for completion_text, target_str, current_available_nums_str in zip(completions, target, nums):
        try:
            current_reward = 0.0 # Default to zero
            has_correct_format = False
            equation_content = None

            completion_text = "<think>" + completion_text 
            completion_text_stripped = completion_text.strip()
            format_match = FORMAT_REGEX.search(completion_text_stripped)
            
            # 1. First, check for the correct format
            if format_match and len(format_match.groups()) == 2:
                has_correct_format = True
                equation_content = format_match.group(2).strip()
            else:
                # Fallback to extract equation from imperfect format
                for regex in EQUATION_REGEXES:
                    match = regex.search(completion_text_stripped)
                    if match and match.group(1):
                        equation_content = match.group(1).strip()
                        break
            
            if not equation_content:
                rewards.append(0.0)
                continue

            # 2. Extract and clean the mathematical expression
            expression_part = equation_content.split('=')[0].strip()
            if not expression_part or not ALLOWED_EQUATION_CHARS_REGEX.match(expression_part.replace(' ', '')):
                rewards.append(0.0)
                continue

            # 3. Verify number usage
            try:
                expected_numbers_int = sorted([int(n) for n in current_available_nums_str])
                temp_expr = expression_part.replace('(', ' ').replace(')', ' ')
                used_numbers_str_list = [item for item in re.split(r'[+\-*/\s]', temp_expr) if item.isdigit()]
                used_numbers_int = sorted([int(n) for n in used_numbers_str_list])

                if used_numbers_int != expected_numbers_int:
                    rewards.append(0.0) # Penalty for using wrong numbers
                    continue
            except (ValueError, IndexError):
                rewards.append(0.0)
                continue

            # 4. Evaluate the expression and assign reward ONLY if correct
            try:
                target_val_float = float(target_str)
                result = eval(expression_part, {"__builtins__": {}}, {})
                
                if abs(float(result) - target_val_float) < FLOAT_COMPARISON_TOLERANCE:
                    # The answer is correct. Now check format for bonus.
                    if has_correct_format:
                        current_reward = 1.0 # Perfect: Correct answer AND format
                    else:
                        current_reward = 0.8 # Good: Correct answer, wrong format
                # If the answer is wrong, the reward remains 0.0, regardless of format.
                
            except (SyntaxError, ZeroDivisionError, NameError, TypeError):
                # Any evaluation error results in zero reward
                current_reward = 0.0
            
            rewards.append(current_reward)

        except Exception as e:
            print(f"Debug: Unexpected error in reward_func: {e}")
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
    output_dir="qwen3-1.7b-grpo-countdown-tasks-v1",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    logging_steps=5,
    report_to="tensorboard",
    max_steps=1125, # 9000 (train_samples) / 24 (global_batch_size) * 1 (num_epochs)
    per_device_train_batch_size=8,
    gradient_accumulation_steps=1, 
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}, 
    bf16=True,
    max_prompt_length=256,    
    max_completion_length=1024, 
    num_generations=4,
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