FROM python:3.9-slim

# 1) Install system-level dependencies, including a C++ compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    gcc \
    g++ \
    make \
    # Optionally, for faster builds:
    # ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2) Copy your requirements file
COPY requirements.txt /app/

# 3) Install torch first (optional, or reorder your requirements so torch is installed first)
RUN pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1

# 4) Install PyG wheel (optional, if you want a pinned version or a CPU-specific wheel)
# RUN pip install torch-geometric==2.6.1 -f https://data.pyg.org/whl/torch-2.5.1+cpu.html

# 5) Then install everything else from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 6) Copy the rest of your code
COPY . /app/

EXPOSE 8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
