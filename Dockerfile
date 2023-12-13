# Base image
FROM runpod/base:0.4.2-cuda11.8.0

ENV HF_HUB_ENABLE_HF_TRANSFER=0

RUN apt-get update && apt-get install -y aria2
RUN aria2c -x8 "https://civitai.com/api/download/models/169921?type=Model&format=SafeTensor&size=pruned&fp=fp16" -o /base_model.safetensors
#COPY /base_model.safetensors /
# Install Python dependencies (Worker Template)
COPY builder/requirements.txt /requirements.txt
RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade -r /requirements.txt --no-cache-dir && \
    rm /requirements.txt

# Cache Models
COPY builder/cache_models.py /cache_models.py
RUN python3.11 /cache_models.py && \
    rm /cache_models.py

# Add src files (Worker Template)
ADD src .

CMD python3.11 -u /rp_handler.py

