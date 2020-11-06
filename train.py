import mlflow
import os
import uuid
import argparse
import mlflow.fastai
import fastai.vision as vis


def parse_args():
    parser = argparse.ArgumentParser(description="Fastai example")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="learning rate to update step size at each step (default: 0.01)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of epochs (default: 5). Note it takes about 1 min per epoch",
    )
    return parser.parse_args()

def main():
    # Parse command=line arguments
    args = parse_args()

    # Setup MLFlow Tracking
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    expr_name = str(uuid.uuid1())
    s3_bucket = os.environ.get("AWS_S3_BUCKET")  # replace this value
    mlflow.create_experiment(expr_name, s3_bucket)
    mlflow.set_experiment(expr_name)

    # Experiment Variables
    print("MLFlow Tracking Server URI: " + mlflow.get_tracking_uri())
    print("Artifact URI: " + mlflow.get_artifact_uri())  # should print out a s3 bucket path

    # Download and untar the MNIST data set
    path = vis.untar_data(vis.URLs.MNIST_TINY)

    # Prepare, transform, and normalize the data
    data = vis.ImageDataBunch.from_folder(path, ds_tfms=(vis.rand_pad(2, 28), []), bs=64)
    data.normalize(vis.imagenet_stats)

    # Train and fit the Learner model
    learn = vis.cnn_learner(data, vis.models.resnet18, metrics=vis.accuracy)

    # Enable auto logging
    mlflow.fastai.autolog()

    # Start MLflow session
    #with mlflow.start_run():
        # Train and fit with default or supplied command line arguments
    learn.fit(args.epochs, args.lr)

if __name__ == "__main__":
    main()