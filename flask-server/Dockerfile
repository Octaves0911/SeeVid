FROM python:3.8

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3

ENV PATH="VIRTUAL_ENV/bin:$PATH"


WORKDIR /app
ADD . /app

RUN python -m pip install -r requirements.txt

#RUN pip install opencv-python

EXPOSE 5000
CMD ["python","app.py"]
