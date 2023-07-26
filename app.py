from fastai.vision.all import *
import gradio as gr
from PIL import Image as PILImage
import numpy as np

learn = load_learner('khachapuri_classifier.pkl')

categories = learn.dls.vocab




def dosomething(img):
    print(1)

def donothing(img):
    return (img, img)

def doeverything(img):
    print(1)
    return img

def predict(img):
    # img = PILImage.create(img)
    _, _, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

def dosequence():
    return np.arange(10)



image = gr.inputs.Image(shape=(256, 256))
label = gr.outputs.Label()
examples = ['achma.jpg', 'guruli.jpg', 'penovani.jpg', 'ajaruli.jpg']

more_examples = ["https://www.196flavors.com/wp-content/uploads/2014/10/achma-3-FP.jpg"]


# write function that rotates images
def rotate(img):
    img = PILImage.create(img)
    img.rotate(11)
    return img


# intf = gr.Interface(fn=predict, inputs=image, outputs=label, examples = examples)
intf = gr.Interface(fn=predict,
                    inputs=image,
                    outputs=gr.outputs.Label(num_top_classes=3),
                    examples=examples)

intf.launch(inline=True)
