FROM ubuntu:noble
ENV VIRTUAL_ENV=/opt/venv
ENV PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install --assume-yes python3-full \
                                    python3-venv \
                                    python3-pip \
                                    cython3 \
    && apt-get clean \
    && apt-get install -y build-essential \
    && adduser --system --no-create-home nonroot

WORKDIR /www
COPY . /www
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --no-cache-dir -r /www/requirements.txt

USER nonroot
EXPOSE 5000
CMD exec gunicorn \
    --bind :5000 \
    --workers 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --threads 3 src.app:model_app \
    --timeout 0 \
    --preload