import flask
import sounddevice as sd
from scipy.io.wavfile import write
from flask import Flask, render_template, request, jsonify
import h2o
h2o.init(ip="127.0.0.1",max_mem_size_GB = 2)
h2o.connect()

app = flask.Flask(__name__, template_folder='templates')
@app.route('/')
def main():
    return(flask.render_template('index.html'))

@app.route('/square/',methods=['POST'])
def background_process_test():
    fs = 44100  # Sample rate
    seconds = 10  # Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write('output.wav', fs, myrecording) # Save as WAV file 
    data = {'square': 'Recoding Done'}
    return data

@app.route('/process/',methods=['POST'])
def processAudio():
    import librosa
    import pandas as pd
    import numpy as np
    import librosa.display
    import parselmouth
    from parselmouth.praat import call
    from parselmouth import MFCC
    import matplotlib.pyplot as plt
    
    import h2o
    
    from h2o.grid.grid_search import H2OGridSearch
    import pandas as pd
    import numpy as np
    from sklearn import preprocessing
    from h2o.estimators.deeplearning import H2OAutoEncoderEstimator, H2ODeepLearningEstimator
    import os
    h2o.init(ip="127.0.0.1",max_mem_size_GB = 2)
    h2o.connect()
    f0min,f0max=70,600
    unit="Hertz"
    wave_file='Audio5780917.wav'
    y, sr = librosa.load(wave_file)
    time=librosa.get_duration(y=y, sr=sr)
    sound = parselmouth.Sound(wave_file)
    print("Processing {}...".format(wave_file))
    duration = call(sound, "Get total duration") # duration
    #ff0min, f0max=75,600 default
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    pitchMean = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    PitchStdev = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
  #  mfcc = call(sound , 'To MelSpectrogram...', 0, 0, 0.0001, 0.02, 1.3, 1.6)
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max) 
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    formants = call(sound, "To Formant (burg)", 0.0025, 5, 5000, 0.025, 50)
    numPoints = call(pointProcess, "Get number of points")
    f1_list = []
    f2_list = []
    f3_list = []
    f4_list = []
    
    # Measure formants only at glottal pulses
    for point in range(0, numPoints):
        point += 1
        t = call(pointProcess, "Get time from index", point)
        f1 = call(formants, "Get value at time", 1, t, 'Hertz', 'Linear')
        f2 = call(formants, "Get value at time", 2, t, 'Hertz', 'Linear')
        f3 = call(formants, "Get value at time", 3, t, 'Hertz', 'Linear')
        f4 = call(formants, "Get value at time", 4, t, 'Hertz', 'Linear')
        f1_list.append(f1)
        f2_list.append(f2)
        f3_list.append(f3)
        f4_list.append(f4)
    
    f1_list = [f1 for f1 in f1_list if str(f1) != 'nan']
    f2_list = [f2 for f2 in f2_list if str(f2) != 'nan']
    f3_list = [f3 for f3 in f3_list if str(f3) != 'nan']
    f4_list = [f4 for f4 in f4_list if str(f4) != 'nan']
    
    # calculate mean formants across pulses
    f1_mean = np.mean(f1_list)
    f2_mean = np.mean(f2_list)
    f3_mean = np.mean(f3_list)
    f4_mean = np.mean(f4_list)


    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    df = pd.DataFrame([[pitchMean ,PitchStdev , hnr ,  np.mean(chroma_stft) ,  np.mean(rmse)  , np.mean(spec_cent)  ,
                  np.mean(spec_bw)  , np.mean(rolloff) ,  np.round(localJitter,6)  , np.round(localabsoluteJitter,6)  ,
                  np.round(rapJitter,6)  , np.round(ppq5Jitter,6)  ,  np.round(ddpJitter,6)  , np.round(localShimmer,6)  ,
                  np.round(localdbShimmer,6)  , np.round(aqpq5Shimmer,6)  , np.round(apq11Shimmer,6)  , np.round(ddaShimmer,6) ,
                  f1_mean  , f2_mean ,  f3_mean  , f4_mean, mfcc[0].mean(),mfcc[1].mean(),mfcc[2].mean(), mfcc[3].mean(),
                  mfcc[4].mean(),mfcc[5].mean(),mfcc[6].mean(),mfcc[7].mean(),mfcc[8].mean(),mfcc[9].mean(),mfcc[10].mean(),
                  mfcc[11].mean(),mfcc[12].mean(),mfcc[13].mean(),mfcc[14].mean(),mfcc[15].mean(),mfcc[16].mean(),
                  mfcc[17].mean(),mfcc[18].mean(),mfcc[19].mean()]] ,
                  columns=['pitchMean' ,'pitchStdev', 'hnr', 'chroma_stft' ,'rmse' ,'spectral_centroid' ,
                 'spectral_bandwidth', 'rolloff', 'localJitter', 'localabsoluteJitter' ,'rapJitter', 'ppq5Jitter' ,'ddpJitter' ,
                 'localShimmer' ,'localdbShimmer' ,'aqpq5Shimmer' ,'apq11Shimmer', 'ddaShimmer' ,'formant1Mean' ,'formant2Mean' ,
                 'formant3Mean' ,'formant4Mean', 'mfcc1','mfcc2','mfcc3','mfcc4','mfcc5','mfcc6','mfcc7','mfcc8','mfcc9','mfcc10',
                 'mfcc11','mfcc12','mfcc13','mfcc14','mfcc15','mfcc16','mfcc17','mfcc18','mfcc19','mfcc20']) 
    df.fillna(0)
    hf = h2o.H2OFrame(df)
    #min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    #X_scale = min_max_scaler.fit_transform(df)
    #saved_model = load_model('ANNModel.h5')
    saved_model = h2o.load_model('DNNH2OModel')
    testPerformance=saved_model.predict(hf)
    #testPerformance=saved_model.predict_classes(X_scale)
    prediction=testPerformance.as_data_frame()
    predict=prediction['predict'][0]
    if (predict == 'NoParkinson'):
        data={'df':'Congrats test is Negative'}
    if (predict == 'Parkinson'):
        data={'df':'Need to go for deep testing'}
    #predict=testPerformance[0]
    #tt=predict[0]
    #if(tt==1):
     #   data={'df':'Test Positve Go For Deep Testing'}
    #if(tt==0):
     #   data={'df':'Congrats Test is Negative'}
    data = jsonify(data)
    return data
if __name__ == '__main__':
    app.run(debug=True)
