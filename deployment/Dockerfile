# dPython image as a base image
FROM python:3.11-slim 

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements_app.txt into the container
COPY requirements_app.txt .

# Install the dependencies from the requirements file
RUN pip install --no-cache-dir -r requirements_app.txt

# Copy the entire Streamlit app into the container
COPY . .

# Expose the port that Streamlit will run on
EXPOSE 8501

# Run the Streamlit app when the container starts
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"]
