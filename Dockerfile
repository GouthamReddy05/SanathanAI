# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose the port Spaces uses
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
