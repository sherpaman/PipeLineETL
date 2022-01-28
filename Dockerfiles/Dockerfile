FROM python:3.8
ARG db_url
ENV DB_URL=$db_url
VOLUME /data
WORKDIR my_workdir
COPY .. .
# COPY ./app /app
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir "uvicorn[standard]"
# RUN pip install numpy~=1.21.2
RUN apt-get update
# RUN apt-get -y install libblas-dev libgeos-dev proj-bin libproj-dev
RUN python setup.py install