FROM python:3.14-slim
RUN useradd --uid 65532 --no-create-home --shell /usr/sbin/nologin spike
COPY entrypoint.py /app/entrypoint.py
USER 65532
ENTRYPOINT ["python", "-I", "/app/entrypoint.py"]
