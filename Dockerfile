FROM python:3.12-slim

WORKDIR /app

# System deps:
#   gcc/g++/make  — Prophet / Stan compilation
#   libglib2.0-0 + others — kaleido headless browser for PDF chart rendering
RUN apt-get update && apt-get install -y \
    gcc g++ make \
    libglib2.0-0 libnss3 libnspr4 \
    libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libgbm1 libpango-1.0-0 libcairo2 \
    libasound2 libx11-6 libx11-xcb1 libxcb1 \
    libxext6 libxi6 libxtst6 \
    libxss1 xdg-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
