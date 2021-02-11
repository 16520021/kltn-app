FROM python:3.7-slim

WORKDIR /app

COPY . .

RUN apt update
RUN apt install -y make automake gcc g++ subversion python3-dev
RUN apt-get install -y cmake
RUN pip install -r requirements.txt --upgrade

CMD ["python", "app.py"]