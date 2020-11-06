#!/usr/bin/env sh

export $(egrep -v '^#' .env | xargs)

mlflow models serve -m "models:/MNIST_TINY/Staging"
