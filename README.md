# Deep Dream API

Flask application to run the Deep Dream Algorithm through an API.

![Trippy Bunny](demo/trippy.jpeg)

## Requirements

```
pip install -r requirements.txt
```

Another important requirement is to download the [pretrained GoogLeNet weigths for PyTorch](https://download.pytorch.org/models/googlenet-1378be20.pth). This can be done by running the following command, which download and prepares the weights:

```
./support/scripts/download_googlenet.sh
```

## Run

### uwsgi

```
uwsgi --ini uwsgi.ini 
```

### development

```
python deep_api
```
