# Deep Dream API

## Intro

This is a Deep Dream implementation using PyTorch with pretrained GoogLeNet weights. The algorithm is very simple and has been written to be fast. The speed depends on the selected style, using the **trippy** style as base, these are the current speeds:

| MBP (i7 - 16GB) | Heroku |
| --- | --- |
| ~4s | ~20s |

Here are some of the available styles:

| Original | Painting |
| --- | --- |
| ![Bunny](demo/bunny.jpg) | ![Painting Bunny](demo/painting.jpeg) |
| <center>**Mixed**</center> | <center>**Trippy**</center> |
| ![Mixed Bunny](demo/texture.jpeg) | ![Trippy Bunny](demo/trippy.jpeg) |
| <center>**Syd Barret**</center> | <center>**Whatever**</center> |
| ![Syd Barret Bunny](demo/barret.jpeg) | ![Whatever Bunny](demo/whatever.jpeg) |


## API

The only available call for now is: **http://deep-api-23.herokuapp.com/deep/dream/<style>**
where *<style>* is one of:

- painting
- texture
- trippy
- barret
- whatever

No API Key is required.

## Deployment

This API is hosted in [heroku](http://heroku.com). Here is a snippet to quickly try the API using the **trippy** style:

```
curl \
    -X POST \
    -F 'image=@/path/to/source/image.jpeg' \
    http://deep-api-23.herokuapp.com/deep/dream/trippy \
    --output /path/to/output/image.jpeg
```

The Heroku app is linked to this repo to perform continous deployment over the master branch.

## Requirements

- [Dev Requirements](./support/requirements/dev.txt) can be installed using: `pip install -r ./support/requirements/dev.txt`.
- [Heroku Requirements](./support/requirements/heroku.txt) are prepared to install the CPU-only PyTorch version and will work in Linux only.

## Development Run

### uwsgi

```
uwsgi --ini uwsgi.ini 
```

### development

```
python deep_api
```
