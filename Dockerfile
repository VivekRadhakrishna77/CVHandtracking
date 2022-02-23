# Dockerfile    Image   Container

# Define Dockerfile

FROM python:3.8

ADD handtracking.py .

RUN apt-get update && apt-get install -y python3-opencv
RUN pip install opencv_contrib_python numpy mediapipe

CMD [ "python", "./handtracking.py" ]