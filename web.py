# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
import os
import librosa
from hparams import create_hparams
from model import PGAN
from soundfile import write


app = Flask("app", template_folder="templates", static_folder="static")
@app.route('/', methods=['POST', 'GET'])

def upload():

    if request.method == 'POST':
        solver(request)
        return redirect(url_for('upload'))
    return render_template('web.html')

def solver(request):

    path_dir = 'static/audio'
    os.makedirs(path_dir, exist_ok=True)
    hparams = create_hparams()
    model_name = request.form.get('model')
    f = request.files['file']
    if model_name == 'U-Net':
        model = PGAN(hparams, 'G2', 'D2')
    else:
        model = PGAN(hparams, model_name[:2], model_name[-2:])
    checkpoint = os.path.join("checkpoints/trained_model", model_name, 'best_G.pkl')
    y, sr = librosa.load(f, sr=hparams.SR)
    insts_wav = model.separate(y.reshape(1, y.shape[0]), hparams, checkpoint)
    insts_name = ['drums', 'bass', 'other', 'vocals']
    for idx, inst_name in enumerate(insts_name):
        write(os.path.join(path_dir, inst_name+'.wav'), insts_wav[idx], hparams.SR)


if __name__ == '__main__':

    app.run(debug=True)

