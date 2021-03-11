from celery import Celery

app = Celery('tasks', broker='pyamqp://guest@localhost//')  # TODO: adapt


@app.task
def predict_handwriting_class(image):
    prediction = predict_image(image)
