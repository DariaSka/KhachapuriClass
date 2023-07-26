from fastai.vision.all import *
import gradio as gr
from PIL import Image as PILImage

learn = load_learner('khachapuri_classifier.pkl')

categories = learn.dls.vocab




def dosomething(img):
    print(2)
    
def donothing(img):
    return (img, img)

def doeverything(img):
    print(1)
    return img

def predict(img):
    # img = PILImage.create(img)
    _, _, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))


image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['achma.jpg', 'ajaruli.jpg', 'guruli.jpg', 'penovani.jpg']

more_examples = ["https://www.196flavors.com/wp-content/uploads/2014/10/achma-3-FP.jpg"]


# write function that rotates images
def rotate(img):
    img = PILImage.create(img)
    img.rotate(45)
    return img


# intf = gr.Interface(fn=predict, inputs=image, outputs=label, examples = examples)
intf = gr.Interface(fn=predict,
                    inputs=image,
                    outputs=gr.outputs.Label(num_top_classes=3),
                    examples=examples)

intf.launch(inline=False)
