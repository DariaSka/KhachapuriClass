from fastai.vision.all import *
import gradio as gr


learn = load_learner('khachapuri_classifier.pkl')

categories = learn.dls.vocab


def predict(img):
    # img = PILImage.create(img)
    _, _, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

def dosomething(img):
    print(1)
    
def donothing(img):
    return img



image = gr.inputs.Image(shape=(192, 192))
label = gr.outputs.Label()
examples = ['achma.jpg', 'ajaruli.jpg', 'guruli.jpg', 'penovani.jpg']

# intf = gr.Interface(fn=predict, inputs=image, outputs=label, examples = examples)
intf = gr.Interface(fn=predict,
                    inputs=image,
                    outputs=gr.outputs.Label(num_top_classes=3),
                    examples=examples)

intf.launch(inline=False)
