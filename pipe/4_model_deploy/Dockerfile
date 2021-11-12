FROM frolvlad/alpine-python-machinelearning
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN pip install google-cloud
RUN pip install google-cloud-aiplatform
RUN pip install google-cloud-storage
ENTRYPOINT ["python", "test_model.py"]