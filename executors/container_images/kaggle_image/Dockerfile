FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime


# Create a non-root user with matching UID/GID to host user (pass via --build-arg HOST_UID=$(id -u))
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN groupadd -g ${HOST_GID} appuser && \
    useradd -r -u ${HOST_UID} -g appuser appuser

# Copy the executor application code
COPY main.py kernel_executor.py instruction.txt output_cleaning.py /app/
WORKDIR /app

# Set up environment variables
ENV PORT=8432
ENV CODE_FOLDER="/data"
ENV PYTHONWARNINGS="ignore::FutureWarning"

# Expose the port
EXPOSE 8432

# Install dependencies once during build (not at runtime)
COPY requirements.kaggle.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create submission directory and set ownership
RUN mkdir -p /submission && chown -R appuser:appuser /submission /app
RUN chmod -R 777 /submission /app

# Switch to non-root user
USER appuser

# Run the application
CMD ["python", "main.py"]