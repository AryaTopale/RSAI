import json
import os
import time

from dotenv import load_dotenv
from groq import Groq

# Load environment variables (ensure secrets.env is present in the working directory)
load_dotenv("secrets.env")
# -------------------------
# Config
# -------------------------
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = "llama-3.1-8b-instant"
TARGET_PER_TYPE = 80
SLEEP_BETWEEN_CALLS = 2
INPUT_FILE = "shopping_tools/shopping_tool_descriptions.json"
OUTPUT_FILE = "shopping_tools/shopping_tool_perturbations.json"
PERTURBATION_TYPES = [
    "description_mismatch",
    "negation_injection",
    "paraphrase",
    "noise_injection",
    "vague_description",
    "description_overload",
]
# -------------------------
# Load Original Tools
# -------------------------
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"Input file {INPUT_FILE} not found.")
with open(INPUT_FILE, "r") as f:
    SHOPPING_TOOLS = json.load(f)
# -------------------------
# Groq Client
# -------------------------
if not API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment. Please check secrets.env.")
client = Groq(api_key=API_KEY)
# -------------------------
# Prompt builder
# -------------------------
PROMPT_TEMPLATE = """
You are helping generate controlled perturbations of tool descriptions for an agent interpretability
       experiment.

 Goal:
 Create multiple perturbed versions of tool descriptions so we can study how an AI agent's tool-calling
       confidence changes under different types of description corruption.

 Tools and Original Descriptions:
 {tools_json}

 Generate BETWEEN 15 AND 20 variations for EACH of the following perturbation types.

 1. description_mismatch
 Swap descriptions between tools while keeping tool names fixed.

 2. negation_injection
 Rewrite descriptions by introducing negations or double-negations while preserving the overall meaning.

 3. paraphrase
 Rewrite descriptions using different wording but with the same meaning.

 4. noise_injection
 Add irrelevant or distracting sentences while keeping the core meaning intact.

 5. vague_description
 Rewrite descriptions to be more generic and less specific.

 6. description_overload
 Expand descriptions to be longer and more verbose with additional details.

 Important constraints:
 - Keep tool names exactly the same as the keys in the input JSON.
 - Maintain grammatical English.
 - Avoid repeating earlier phrasing patterns.
 - Return STRICT JSON only.

 Required format:
 {{
 "description_mismatch": [{{...}}, {{...}}],
 "negation_injection": [{{...}}, {{...}}],
 "paraphrase": [{{...}}, {{...}}],
 "noise_injection": [{{...}}, {{...}}],
 "vague_description": [{{...}}, {{...}}],
 "description_overload": [{{...}}, {{...}}]
 }}

 Each list must contain 10-12 variants.
 """


def clean_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.split("`")[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


# Initialize dataset
dataset = {t: [] for t in PERTURBATION_TYPES}
seen = {t: set() for t in PERTURBATION_TYPES}


def enough():
    return all(len(dataset[t]) >= TARGET_PER_TYPE for t in PERTURBATION_TYPES)


def add_variants(data):
    for t in PERTURBATION_TYPES:
        if t not in data:
            continue
        for variant in data[t]:
            # Ensure all tools are present in the variant
            if not all(k in variant for k in SHOPPING_TOOLS):
                continue

            key = json.dumps(variant, sort_keys=True)
            if key not in seen[t]:
                seen[t].add(key)
                dataset[t].append(variant)


iteration = 1
tools_json_str = json.dumps(SHOPPING_TOOLS, indent=2)

while not enough():
    print(f"\nGenerating batch {iteration}")
    try:
        #   Mix in time to ensure randomness for the model if it supports it via prompt
        prompt = (
            PROMPT_TEMPLATE.format(tools_json=tools_json_str)
            + f"\n\nRandom seed: {iteration + time.time()}"
        )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )

        raw_text = response.choices[0].message.content or ""
        cleaned = clean_json(raw_text)
        data = json.loads(cleaned)

        add_variants(data)

        for t in PERTURBATION_TYPES:
            print(f"{t}: {len(dataset[t])}")

        iteration += 1
        time.sleep(SLEEP_BETWEEN_CALLS)

    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)

#   Finalize and save
final_dataset = {t: dataset[t][:TARGET_PER_TYPE] for t in PERTURBATION_TYPES}

with open(OUTPUT_FILE, "w") as f:
    json.dump(final_dataset, f, indent=2)

print(f"\nDataset saved to {OUTPUT_FILE}")
