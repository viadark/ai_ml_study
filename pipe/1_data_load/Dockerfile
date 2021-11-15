FROM python:3
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY . /code/
RUN python -m pip install --upgrade pip
RUN pip install pandas
ENTRYPOINT ["python", "load_data.py"]