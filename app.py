import numpy as np
import tensorflow as tf
import cv2
#import faiss-gpu
import gradio as gr

from PIL import Image
from skimage.io import imread, imsave
from sklearn.preprocessing import normalize
from face_detector import YoloV5FaceDetector
    

def register(img1, img2, img3, name):
    imgs = [img1, img2, img3]
    print(len(imgs))
    if name=='':
        name = 'no_name'
        
    #load model
    model = tf.keras.models.load_model('models/GhostFaceNet_W1.3_S1_ArcFace.h5')
    model_interf = lambda imms: model((imms - 127.5) * 0.0078125).numpy()
    
    #load embedding vector database
    aa = np.load("models/vn2db.npz")
    embs, imm_classes, id_names = aa["embs"], aa["imm_classes"], aa["id_names"]
    embs, img_class = embs.astype("float32"), imm_classes.astype("int")
    
    #extract embedding vector
    identity = len(np.unique(imm_classes))
    yolo = YoloV5FaceDetector()
    faces = []
    for idx, img in enumerate(imgs):
        #detect and drop face
        _,_,_,nfaces = yolo.detect_in_image(img)
        face = nfaces[0].reshape((-1, 112, 112, 3))
        #faces = np.append(faces, np.uint8(face))
        #get embedding vector
        emb = model_interf(face)
        emb = normalize(np.array(emb).astype("float32"))[0]
        
        #add register info
        embs = np.append(embs, emb)
        imm_classes = np.append(imm_classes, identity)
        id_names = np.append(id_names, name)
        
        #save emb into hard disk
        imsave(f'images/{name}_{idx}.jpg', nfaces[0])
    
    #save embedding vector database
    np.savez("models/vn2db.npz", embs=embs, imm_classes=imm_classes, id_names=id_names)
    #notify success
    print('Done')
    #return np.array(faces[0]), np.array(faces[1]), np.array(faces[2])
    return f"Register user '{name}' success."

def checkin(img):
    #load model
    model = tf.keras.models.load_model('models/GhostFaceNet_W1.3_S1_ArcFace.h5')
    model_interf = lambda imms: model((imms - 127.5) * 0.0078125).numpy()
    
    #detect and drop face
    _,_,_,nfaces = YoloV5FaceDetector().detect_in_image(img)
    face = nfaces[0].reshape((-1, 112, 112, 3))
    
    #get embedding vector
    emb = model_interf(face)
    emb = normalize(np.array(emb).astype("float32"))[0]
    
    #load embedding vector database
    aa = np.load("models/vn2db.npz")
    embs, imm_classes, id_names = aa["embs"], aa["imm_classes"], aa["id_names"]
    embs, img_class = embs.astype("float32"), imm_classes.astype("int")
    
    

with gr.Blocks() as demo:
    with gr.Tab("Register"):
        txtIdName = gr.Textbox(label="Identity Name")
        with gr.Column(scale=1):
            with gr.Row():
                img1 = gr.Image()
                img2 = gr.Image()
                img3 = gr.Image()
        btnRegister = gr.Button("Register")
        label1 = gr.Label()
                
    with gr.Tab("Checkin"):
        with gr.Row():
            with gr.Column(scale=2):
                img4 = gr.Image()
                btnCheckin = gr.Button("Checkin")
            with gr.Column(scale=3):
                label2 = gr.Label()
                img5 = gr.Image()

    btnRegister.click(register, inputs=[img1, img2, img3, txtIdName], outputs=label1)
    #btnCheckin.click(checkin, inputs=img4, outputs=label2)  
    
demo.launch(share=True)