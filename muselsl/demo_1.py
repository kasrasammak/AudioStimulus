
from psychopy import prefs
#print(prefs)
#prefs.general['audioLib'] = ['sounddevice']
#print(prefs)

from psychopy import sound, core, visual, event


#snd = sound.backend_sounddevice.SoundDeviceSound(value='C', secs=0.5, 
#                                                 octave=4, stereo=-1, 
#                                                 volume=1.0, loops=0, 
#                                                 sampleRate=None, blockSize=128, 
#                                                 preBuffer=-1, hamming=True, 
#                                                 startTime=0, stopTime=-1, 
#                                                 name='', autoLog=True)

snd = sound.Sound('sinebasswave.wav', sampleRate=41000, stereo = True)
#snd = setVo!lume(0.5)
snd.play()
core.wait(1.0)