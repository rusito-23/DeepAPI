# Deep Dream API

Flask application to run the Deep Dream Algorithm through an API.

| Original | Painting | Trippy |
| --- | --- | --- |
| ![Bunny](demo/bunny.jpg) | ![Painting Bunny](demo/painting.jpeg) | ![Trippy Bunny](demo/trippy.jpeg) |

## Requirements

- [Dev Requirements](./support/requirements/dev.txt) can be installed using: `pip install -r ./support/requirements/dev.txt`.
- [Heroku Requirements](./support/requirements/heroku.txt) are prepared to install the CPU-only PyTorch version and will work in Linux only.

## Run

### uwsgi

```
uwsgi --ini uwsgi.ini 
```

### development

```
python deep_api
```
