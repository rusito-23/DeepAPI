#!/bin/sh


## -- Test all styles -- ##

styles=(
    trippy
    texture
    mixed
)


for style in ${styles[@]}; do
    curl \
        -X POST \
        -F 'image=@bunny.jpg' \
        http://localhost:5000/deep/dream/$style \
        --output $style.jpeg
done
