# Use the official slim Python image — small, secure, current.
FROM python:3.12-slim

# Don't let pip cache wheels (smaller image), don't buffer stdout (live logs).
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System libs:
#   build-essential        — for any pip wheels that compile from source
#   openjdk-21-jre-headless — required by PySpark (Spark runs on the JVM).
#                            Java 17 was dropped from Debian Trixie; Spark 3.5+ supports Java 21.
#   procps                 — `ps` is used by some Spark scripts
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        openjdk-21-jre-headless \
        procps \
    && rm -rf /var/lib/apt/lists/*

# Spark needs JAVA_HOME pointing at the JRE.
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 \
    PATH="/usr/lib/jvm/java-21-openjdk-amd64/bin:${PATH}"

WORKDIR /workspace

# Install Python deps first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# JupyterLab listens on 8888; Spark's web UI uses 4040 (and 4041, 4042... for extra contexts).
EXPOSE 8888 4040

# Start JupyterLab. --ip=0.0.0.0 lets the host browser reach it via the port mapping.
# --no-browser skips trying to open a browser inside the container.
# --ServerApp.token='' disables the auth token for local dev convenience.
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password=", \
     "--ServerApp.root_dir=/workspace"]
