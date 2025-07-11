FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN mkdir -p /app/logs

RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]