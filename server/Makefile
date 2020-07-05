
TAG=stable
NAME=deep_api
APP=${NAME}:${TAG}

DOCKER_FILE=./support/Dockerfile

LOCAL_PORT=56733
CONTAINER_PORT=80


build:
	docker build -t ${APP} -f ${DOCKER_FILE} .

run: build
	docker run \
	    --detach \
	    --publish ${LOCAL_PORT}:${CONTAINER_PORT} \
	    --name=${APP} \
	    --volume ${PWD}:/app ${APP}} 
