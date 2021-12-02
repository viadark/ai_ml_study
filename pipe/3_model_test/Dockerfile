FROM frolvlad/alpine-python-machinelearning
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN python -m pip install --upgrade pip
RUN pip install requests
ENTRYPOINT ["python", "test_model.py"]