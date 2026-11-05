# Build from the road_vlm_samples folder, so reward.py (the caption rules) is in the build context:
#   docker build -f label_studio/road_captions/Dockerfile -t road-captions .
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY label_studio/road_captions/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY reward.py /app/reward.py
COPY label_studio/road_captions /app/road_captions
RUN useradd --system --uid 10001 app \
 && mkdir -p /var/log/road-captions && chown app /var/log/road-captions
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/road-api/health')"
CMD ["uvicorn", "road_captions.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
