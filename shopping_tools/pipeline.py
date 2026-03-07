import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
BASE_DIR = "/kaggle/input/datasets/prithvikarthik/shopping-tools/"
DATA_PATH = BASE_DIR + "shopping_tool_balanced.csv"
BASELINE_PATH = BASE_DIR + "shopping_tools.json"
PERTURBATIONS_PATH = BASE_DIR + "shopping_tool_perturbations.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Model Once ---
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()

# --- Load Data ---
df_balanced = pd.read_csv(DATA_PATH)
with open(BASELINE_PATH) as f:
    baseline_tools = json.load(f)
with open(PERTURBATIONS_PATH) as f:
    perturbations = json.load(f)

perturbation_types = [
    "description_mismatch",
    "paraphrase",
    # "vague_description",
    # "description_overload",
    # "noise_injection",
    # "negation_injection",
    # "prefix_noise_injection",
    # "interleaved_noise_injection",
]


def run_eval(tool_desc_dict):
    """
    Runs inference using generation (like the reference code).
    Returns accuracy and mean confidence.
    """

    tool_list = list(tool_desc_dict.keys())
    tool_text = "\n".join([f"{t}: {tool_desc_dict[t]}" for t in tool_list])

    correct_count = 0
    confidences = []
    all_errors = []

    system_prompt = f"""
You are a tool selection agent.

Select the SINGLE best tool for the user query.

Available tools:
{tool_text}

Instructions:
- Return ONLY the tool name.
- Do NOT output explanations.

Valid answers:
{", ".join(tool_list)}
"""

    for _, row in df_balanced.iterrows():
        query = row["Query"]
        gold_tool = row["Tool"]

        prompt = f"""{system_prompt}

User query:
{query}

Answer:
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # -------------------------
        # Extract generated tokens
        # -------------------------
        generated_tokens = outputs.sequences[0][inputs["input_ids"].shape[1] :]

        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        decoded = decoded.split("\n")[0].strip()

        # -------------------------
        # Map generated text to tool
        # -------------------------
        pred_tool = None
        for t in tool_list:
            if t.lower() in decoded.lower():
                pred_tool = t
                break

        # -------------------------
        # Compute confidence
        # -------------------------
        token_probs = []

        for i, token_id in enumerate(generated_tokens):
            probs = torch.softmax(outputs.scores[i], dim=-1)
            prob = probs[0, token_id].item()
            token_probs.append(prob)

        confidence = np.mean(token_probs) if token_probs else 0
        confidences.append(confidence)

        # -------------------------
        # Accuracy check
        # -------------------------
        if pred_tool == gold_tool:
            correct_count += 1
        else:
            all_errors.append(
                {
                    "query": query,
                    "gold_tool": gold_tool,
                    "predicted_tool": pred_tool,
                    "model_output": decoded,
                    "confidence": confidence,
                }
            )

    # save errors
    if len(all_errors) > 0:
        pd.DataFrame(all_errors).to_csv("shopping_tool_errors.csv", index=False)

    accuracy = correct_count / len(df_balanced)
    mean_conf = np.mean(confidences)

    return accuracy, mean_conf


# --- Execution Pipeline ---
results = []

# 1. Baseline
print("\nEvaluating Baseline...")
acc, conf = run_eval(baseline_tools)
results.append({"type": "baseline", "accuracy": acc, "confidence": conf})

# 2. Perturbations
for p_type in perturbation_types:
    if p_type not in perturbations:
        continue

    print(f"Evaluating Perturbation: {p_type}...")
    type_accs, type_confs = [], []

    # We aggregate across variants within the type
    variants = perturbations[p_type]
    for variant in tqdm(variants):
        acc, conf = run_eval(variant)
        type_accs.append(acc)
        type_confs.append(conf)

    results.append(
        {
            "type": p_type,
            "accuracy": np.mean(type_accs),
            "confidence": np.mean(type_confs),
        }
    )

# --- Final Aggregation & Display ---
summary_df = pd.DataFrame(results)
summary_df["acc_delta"] = summary_df["accuracy"] - summary_df.iloc[0]["accuracy"]
summary_df["conf_delta"] = summary_df["confidence"] - summary_df.iloc[0]["confidence"]

print("\n" + "=" * 30)
print("FINAL PERTURBATION STUDY")
print("=" * 30)
print(summary_df[["type", "accuracy", "confidence", "acc_delta", "conf_delta"]])

summary_df.to_csv("shopping_tool_perturbation_summary.csv", index=False)
