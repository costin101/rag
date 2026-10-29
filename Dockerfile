# Session container image: the stub generator only (rule 7). python:3.14-slim, linux manifest
# list resolved 2026-10-29; the digest is the pin, so bump it deliberately.
FROM python:3.14-slim@sha256:cd89c30eda26ee911a4611b18680086e0ea42e6f5f8cbe653d07fa7958c9cec8
ENV PYTHONDONTWRITEBYTECODE=1
RUN useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin session
COPY container/entrypoint.py /app/entrypoint.py
USER 65532
ENTRYPOINT ["python", "-I", "/app/entrypoint.py"]
