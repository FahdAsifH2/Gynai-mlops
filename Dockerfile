FROM python:3.11

WORKDIR /app

# Install heavy packages first (cached layer)
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Install rest of packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python","main.py"]