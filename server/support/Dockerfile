FROM tiangolo/uwsgi-nginx-flask:python3.7

COPY ./support /support
COPY ./deep_api /deep_api

RUN pip install --upgrade pip
RUN pip install -r /support/requirements.txt
RUN sh /support/scripts/download_googlenet.sh
