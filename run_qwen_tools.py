import json
import pandas as pd
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import argparse

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
DES_PATH = "exp_descriptions.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# ARGUMENTS
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", required=True)
parser.add_argument("--output", default="results.csv")
parser.add_argument("--save_probe", action="store_true")
args = parser.parse_args()

# -------------------------
# LOAD MODEL
# -------------------------
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    output_hidden_states=True
)

model.eval()

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(args.data)

with open(DES_PATH) as f:
    tool_desc = json.load(f)

tool_list = list(tool_desc.keys())

tool_text = "\n".join(
    [f"{t}: {tool_desc[t]}" for t in tool_list]
)

tool_options = "\n".join(tool_list)

# -------------------------
# SYSTEM PROMPT
# -------------------------
system_prompt = f"""
You are a tool selection agent.

Select the SINGLE best tool for the user query.

Available tools:
{tool_text}

Instructions:
- Return ONLY the tool name.
- Do NOT output explanations.

Valid answers:
{tool_options}
"""

# -------------------------
# INFERENCE
# -------------------------
results = []

probe_features = []
probe_labels = []

print("Running inference...")

for _, row in tqdm(df.iterrows(), total=len(df)):

    query = row["Query"]

    prompt = f"""{system_prompt}

User query:
{query}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # forward pass for hidden states
    with torch.no_grad():
        outputs = model(**inputs)

    # mean pooled hidden state
    hidden = outputs.hidden_states[-1].mean(dim=1).cpu().numpy()[0]

    # generate prediction
    gen = model.generate(
        **inputs,
        max_new_tokens=4,
        do_sample=False,
        temperature=0
    )

    generated = gen[0][inputs["input_ids"].shape[1]:]

    decoded = tokenizer.decode(
        generated,
        skip_special_tokens=True
    ).strip()

    decoded = decoded.split("\n")[0].strip()

    pred_tool = None

    for t in tool_list:
        if t.lower() in decoded.lower():
            pred_tool = t
            break

    results.append({
        "query": query,
        "gold_tool": row["Tool"],
        "pred_tool": pred_tool,
        "model_output": decoded
    })

    # collect probe dataset
    if args.save_probe and pred_tool is not None:
        probe_features.append(hidden)
        probe_labels.append(pred_tool)

# -------------------------
# SAVE RESULTS
# -------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(args.output, index=False)

accuracy = (results_df["gold_tool"] == results_df["pred_tool"]).mean()

print("Saved:", args.output)
print("Accuracy:", accuracy)

# -------------------------
# SAVE PROBE DATASET
# -------------------------
if args.save_probe:

    X = np.array(probe_features)
    y = np.array(probe_labels)

    np.save("X_probe.npy", X)
    np.save("y_probe.npy", y)

    print("Saved probe dataset:")
    print("X_probe.npy", X.shape)
    print("y_probe.npy", y.shape)