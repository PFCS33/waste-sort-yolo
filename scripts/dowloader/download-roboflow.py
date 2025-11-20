from roboflow import Roboflow
rf = Roboflow(api_key="u6dBxE6mzDkitzf9IOgk")
project = rf.workspace("material-identification").project("garbage-classification-3")
version = project.version(2)
dataset = version.download("yolov8")
