# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# WORKDIR /workspace

# ENV PYTHONUNBUFFERED=1

# # Install required packages
# RUN pip install --no-cache-dir \
#     transformers==4.40.2 \
#     accelerate \
#     pandas \
#     tqdm \
#     numpy \
#     scikit-learn \
#     sentencepiece

# # Copy project files
# COPY . .

# # Default command: generate probe dataset from train split
# CMD ["python", "run_qwen_tools.py", "--data", "train.csv", "--output", "train_results.csv", "--save_probe"]
# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# WORKDIR /workspace

# ENV PYTHONUNBUFFERED=1

# RUN pip install --no-cache-dir \
#     numpy \
#     pandas \
#     scikit-learn \
#     tqdm

# COPY . .

# CMD ["python", "train_probe.py"]
# FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

# WORKDIR /workspace

# ENV PYTHONUNBUFFERED=1

# RUN pip install --no-cache-dir \
#     transformers==4.40.2 \
#     accelerate \
#     pandas \
#     tqdm \
#     numpy \
#     scikit-learn \
#     matplotlib \
#     sentencepiece

# COPY . .

# CMD ["python", "layerwise_probe.py"]

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /workspace

RUN apt-get update && apt-get install -y git

COPY . /workspace

RUN pip install --no-cache-dir \
    transformers \
    accelerate \
    pandas \
    scikit-learn \
    matplotlib \
    tqdm

CMD ["python", "layerwise_probe_eval.py"]