import os
import re

from huggingface_hub import login

from datasets import load_dataset

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer, ModelConfig
from peft import LoraConfig


login(token=os.environ.get("HF_TOKEN"), add_to_git_credential=True) 

# MODEL SETUP
model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# DATASET SETUP
dataset_id = "Jiayi-Pan/Countdown-Tasks-3to4"
dataset = load_dataset(dataset_id, split="train")
dataset = dataset.shuffle().select(range(1000))

def generate_r1_prompt(numbers, target):
    r1_prefix = [
        {
            "role": "system",
            "content": "You are a math expert. You will first reason carefully about the problem, then provide the user with the answer."
        },
        {
            "role": "user",
            "content": f"Given the numbers {numbers} and the target number {target}, please provide a solution to reach the target number using the four basic arithmetic operations: addition, subtraction, multiplication, and division (+, -, *, /). You can use each number only once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>."
        },
    ]
    return {"prompt": tokenizer.apply_chat_template(r1_prefix, tokenize=False, continue_final_message=False), "target": target}

dataset = dataset.map(
    lambda x: generate_r1_prompt(x["nums"], x["target"]),
)

train_test_split = dataset.train_test_split(test_size=0.1)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# TRAINING SETUP
# --- Constants ---
# Regex for the full <think>...</think><answer>...</answer> format
# It ensures <think> and <answer> are direct children and not nested within each other incorrectly.
# The part ((?:(?!<\/think>).)*) captures content within <think> non-greedily.
# The part ((?:(?!<\/answer>).)*) captures content within <answer> non-greedily.
FORMAT_REGEX_PATTERN = r"^<think>((?:(?!<\/think>).)*)<\/think>\s*<answer>((?:(?!<\/answer>).)*)<\/answer>$"
FORMAT_REGEX = re.compile(FORMAT_REGEX_PATTERN, re.DOTALL)

# Regex to extract content from <answer> tag
ANSWER_REGEX_PATTERN = r"<answer>((?:(?!<\/answer>).)*)<\/answer>"
ANSWER_REGEX = re.compile(ANSWER_REGEX_PATTERN, re.DOTALL) # re.DOTALL allows . to match newlines

# Regex for allowed characters in an equation
ALLOWED_EQUATION_CHARS_PATTERN = r'^[\d+\-*/().\s]+$'
ALLOWED_EQUATION_CHARS_REGEX = re.compile(ALLOWED_EQUATION_CHARS_PATTERN)

# Tolerance for float comparisons
FLOAT_COMPARISON_TOLERANCE = 1e-5

# Optional: Setup a logger if you want to see errors instead of just getting 0.0
# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO) # Or logging.DEBUG for more verbosity


def format_reward_func(completions: list[str], **kwargs) -> list[float]:
    """
    Checks if completions strictly follow the <think>...</think>\n<answer>...</answer> format.
    The model is expected to generate the full string including the opening <think> tag.

    Args:
        completions (list[str]): Generated outputs from the assistant, each expected to
                                 start with "<think>" and follow the full format.
                                 Example: "<think>I will solve it.</think>\n<answer>42</answer>"
        **kwargs: Additional keyword arguments (ignored by this function).

    Returns:
        list[float]: Reward scores (1.0 for correct format, 0.0 otherwise).
    """
    rewards = []
    for completion_text in completions:
        try:
            completion_text = completion_text.strip()  # Clean up whitespace
            # The completion_text itself is expected to be the full string
            match = FORMAT_REGEX.search(completion_text)

            if match and len(match.groups()) == 2:
                # Both <think> and <answer> content captured
                rewards.append(1.0)
            else:
                print(f"Debug: Format mismatch for: {completion_text}")
                rewards.append(0.0)
        except Exception as e:
            print(f"Debug: Error processing completion for format check: {completion_text}, Error: {e}")
            rewards.append(0.0)
    return rewards


def equation_reward_func(
    completions: list[str],
    target: list[str],
    nums: list[list[str]],
    **kwargs
) -> list[float]:
    rewards = []
    for completion_text, target_str, available_nums_str in zip(completions, target, nums):
        try:
            current_reward = 0.0 

            answer_match = ANSWER_REGEX.search(completion_text)
            if not answer_match:
                rewards.append(current_reward)
                continue

            full_answer_content = answer_match.group(1).strip()
            if not full_answer_content:
                rewards.append(current_reward)
                continue

            # Try to split "expression = result"
            parts = full_answer_content.rsplit('=', 1)
            expression_part_str = parts[0].strip()
            
            # It's possible the model only gives the expression, or gives something else.
            # If no '=', the whole thing is the expression.
            if len(parts) == 1: # No '=' found or it's at the very beginning
                expression_part_str = full_answer_content 
            elif not expression_part_str: # Handles cases like "= 138"
                rewards.append(current_reward) # Invalid format if expression is empty
                continue


            if not expression_part_str: # Handle empty expression after split
                rewards.append(current_reward)
                continue

            # 1. Check for allowed characters in the EXPRESSION PART
            if not ALLOWED_EQUATION_CHARS_REGEX.match(expression_part_str):
                print(f"Debug: Expression '{expression_part_str}' contains forbidden characters.")
                rewards.append(current_reward)
                continue

            # 2. Check number usage (using expression_part_str)
            try:
                expected_numbers_int = sorted([int(n) for n in available_nums_str])
            except ValueError:
                rewards.append(current_reward)
                continue

            used_numbers_str = re.findall(r'\d+', expression_part_str) # Check numbers in expression only
            try:
                used_numbers_int = sorted([int(n) for n in used_numbers_str])
            except ValueError:
                rewards.append(current_reward)
                continue

            if used_numbers_int != expected_numbers_int:
                print(f"Debug: Number usage mismatch. Used: {used_numbers_int}, Expected: {expected_numbers_int} in '{expression_part_str}'")
                rewards.append(current_reward)
                continue

            # 3. Evaluate the EXPRESSION PART and check correctness against target_str
            try:
                target_val = float(target_str)
                eval_globals = {"__builtins__": {}} 
                eval_locals = {}
                result = eval(expression_part_str, eval_globals, eval_locals) # Evaluate only the expression

                if abs(float(result) - target_val) < FLOAT_COMPARISON_TOLERANCE:
                    current_reward = 1.0
                # else:
                    print(f"Debug: Equation result mismatch. Eq: '{expression_part_str}' -> {result}, Target: {target_val}")
            except SyntaxError:
                print(f"Debug: Syntax error in expression: {expression_part_str}")
                pass 
            except TypeError:
                pass 
            except ZeroDivisionError:
                pass 
            except Exception as eval_e:
                print(f"Debug: Unexpected error evaluating expression '{expression_part_str}': {eval_e}")
                pass 

            rewards.append(current_reward)

        except Exception as e:
            rewards.append(0.0)

    return rewards


model_config = ModelConfig(
    model_name_or_path=model_name,
    torch_dtype="bfloat16",
    use_peft=True,
    load_in_4bit=True
)

peft_config = LoraConfig(
    r=16,                       # LoRA rank - higher means more capacity but more parameters
    lora_alpha=32,              # LoRA alpha - scaling factor
    lora_dropout=0.05,          # Dropout probability for LoRA layers
    bias="none",                # Don't train bias parameters to save memory
    task_type="CAUSAL_LM",      # Task type for causal language modeling
    target_modules=[
        # Attention layers
        "q_proj", 
        "k_proj", 
        "v_proj", 
        "o_proj",
        # MLP/FFN layers
        "gate_proj",  
        "up_proj", 
        "down_proj"
    ],
    # QLoRA specific settings
    fan_in_fan_out=False,       # Set to True for specific architectures that need this
    modules_to_save=None,       # Specific modules to fully fine-tune if needed
)

training_args = GRPOConfig(
    output_dir="qwen0.6-grpo-countdown-tasks",
    learning_rate=5e-7,
    lr_scheduler_type="cosine",
    logging_steps=10,
    report_to="tensorboard",
    max_steps=100,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=True,
    # GRPO specific arguments
    max_prompt_length=256,
    max_completion_length=1024,
    num_generations=2,
    beta=0.001,
)

trainer = GRPOTrainer(
    model=model_config.model_name_or_path,
    reward_funcs=[format_reward_func, equation_reward_func],
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=peft_config,
)

# Start training
print("Starting training...")
trainer.train()
trainer.save_model(training_args.output_dir)