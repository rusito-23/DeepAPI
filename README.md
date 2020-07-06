# Deep Dream API

Flask application to run the Deep Dream Algorithm through an API.

## Root Folder

The root folder is `server`.
All commands stated below should be executed in this folder. 

## Requirements

The pip requirements can be found in the [support files folder](server/support) and installed using:

```
pip install -r support/requirements.txt
```

Another important requirement is to download the [pretrained GoogLeNet weigths for PyTorch](https://download.pytorch.org/models/googlenet-1378be20.pth). This can be done by running the following command, which download and prepares the weights:

```
./support/scripts/download_googlenet.sh
```

## Run

### uwsgi

```
uwsgi --ini support/uwsgi.ini 
```

### development

```
python deep_api
```
