FROM python:3.9-slim

WORKDIR /app

# Create .streamlit directory and copy secrets
RUN mkdir -p /app/.streamlit
COPY .streamlit/secrets.toml /app/.streamlit/secrets.toml

# Copy requirements and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire repository so that the "src" folder is preserved
COPY . /app/

# Set PYTHONPATH so Python can locate the src package
ENV PYTHONPATH=/app

# Run the Streamlit app (change the server port if needed)
CMD ["streamlit", "run", "src/app.py"]
