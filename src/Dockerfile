FROM python:3

RUN apt-get update -y && \
  apt-get install -y python3-pip python3-dev

RUN apt-get update -y && \
    apt-get install -y libgl1 libglib2.0-0

WORKDIR /app

COPY . .

RUN pip install opencv-python

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
CMD ["python", "engine.py"]