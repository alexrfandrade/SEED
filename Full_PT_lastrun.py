#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.1),
    on Wed Feb 11 16:10:55 2026
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.1'
expName = 'Full_PT'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = False
_winSize = [1440, 900]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='/Users/vera/Documents/GitHub/SEED/Full_PT_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height', 
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.mouseVisible = True
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    if deviceManager.getDevice('key_welcome') is None:
        # initialise key_welcome
        key_welcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_welcome',
        )
    if deviceManager.getDevice('key_explain_PAL') is None:
        # initialise key_explain_PAL
        key_explain_PAL = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_explain_PAL',
        )
    if deviceManager.getDevice('key_explain_rate') is None:
        # initialise key_explain_rate
        key_explain_rate = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_explain_rate',
        )
    if deviceManager.getDevice('key_ex_rate') is None:
        # initialise key_ex_rate
        key_ex_rate = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_ex_rate',
        )
    if deviceManager.getDevice('key_start_PAL') is None:
        # initialise key_start_PAL
        key_start_PAL = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_start_PAL',
        )
    if deviceManager.getDevice('key_rate') is None:
        # initialise key_rate
        key_rate = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_rate',
        )
    if deviceManager.getDevice('key_Start_Cog') is None:
        # initialise key_Start_Cog
        key_Start_Cog = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_Start_Cog',
        )
    if deviceManager.getDevice('key_end_cog') is None:
        # initialise key_end_cog
        key_end_cog = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_end_cog',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('key_continue') is None:
        # initialise key_continue
        key_continue = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_continue',
        )
    if deviceManager.getDevice('key_explain_old') is None:
        # initialise key_explain_old
        key_explain_old = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_explain_old',
        )
    if deviceManager.getDevice('key_ex_old') is None:
        # initialise key_ex_old
        key_ex_old = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_ex_old',
        )
    if deviceManager.getDevice('key_start_recall') is None:
        # initialise key_start_recall
        key_start_recall = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_start_recall',
        )
    if deviceManager.getDevice('key_item') is None:
        # initialise key_item
        key_item = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_item',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('key_end_explain_2afc') is None:
        # initialise key_end_explain_2afc
        key_end_explain_2afc = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_end_explain_2afc',
        )
    if deviceManager.getDevice('key_ex_2AFC') is None:
        # initialise key_ex_2AFC
        key_ex_2AFC = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_ex_2AFC',
        )
    if deviceManager.getDevice('key_2AFC') is None:
        # initialise key_2AFC
        key_2AFC = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_2AFC',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "Go_Nogo_task" ---
    
    # --- Initialize components for Routine "Welcome" ---
    text_hi = visual.TextStim(win=win, name='text_hi',
        text='Bem-vindo!\n\nObrigada por participar nesta experiência.\n\n\n\nPrime ESPAÇO para avançar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_welcome = keyboard.Keyboard(deviceName='key_welcome')
    
    # --- Initialize components for Routine "Explanation_PAL" ---
    text_Explain_PAL = visual.TextStim(win=win, name='text_Explain_PAL',
        text='Na primeira tarefa, vamos mostrar um par de imagens : uma cena e um objeto. Avalia o nível de congruência entre o par de 1 a 3, onde:\n\n1 = Incongruente (o objeto não faz sentido nesta cena)\n2 = Intermédio\n3 = Congruente (o objeto faz sentido nesta cena)\n\n\nPrime ESPAÇO para avançar\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=1.4, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_explain_PAL = keyboard.Keyboard(deviceName='key_explain_PAL')
    
    # --- Initialize components for Routine "Explain_Rating" ---
    text_explain_rate = visual.TextStim(win=win, name='text_explain_rate',
        text="\nPrime a tecla indicada como '1', '2', ou '3' no teclado numérico (ao lado das flechas) para selecionar o nível de congruência /compatibilidade do par.\n\n\nPrime ESPAÇO para um exemplo",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_explain_rate = keyboard.Keyboard(deviceName='key_explain_rate')
    
    # --- Initialize components for Routine "blank" ---
    cross_25 = visual.ShapeStim(
        win=win, name='cross_25', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    image_blank = visual.ImageStim(
        win=win,
        name='image_blank', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "Example_PAL" ---
    image_ex_scene = visual.ImageStim(
        win=win,
        name='image_ex_scene', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-200,0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_ex_object = visual.ImageStim(
        win=win,
        name='image_ex_object', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(300, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    slider_example = visual.Slider(win=win, name='slider_example',
        startValue=None, size=(0.8, 0.04), pos=(0, -0.3), units=win.units,
        labels=['1 - Incongruente','2 - Intermédio','3 - Congruente'], ticks=(1, 2, 3), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.04,
        flip=False, ori=0.0, depth=-2, readOnly=False)
    text_example = visual.TextStim(win=win, name='text_example',
        text='Qual o nível de congruência/compatibilidade entre as duas imagens?\n',
        font='Arial',
        pos=(0, 0.25), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    key_ex_rate = keyboard.Keyboard(deviceName='key_ex_rate')
    
    # --- Initialize components for Routine "Start_PAL" ---
    text_end_ex = visual.TextStim(win=win, name='text_end_ex',
        text='Exemplo concluído.\n\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_start_PAL = visual.TextStim(win=win, name='text_start_PAL',
        text='Vamos começar? \n\nSe tiveres alguma dúvida chama o experimentador.\n\n\nPrime a tecla ESPAÇO quando estiveres pronto',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_start_PAL = keyboard.Keyboard(deviceName='key_start_PAL')
    
    # --- Initialize components for Routine "blank" ---
    cross_25 = visual.ShapeStim(
        win=win, name='cross_25', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    image_blank = visual.ImageStim(
        win=win,
        name='image_blank', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "Encode" ---
    image_scene_encode = visual.ImageStim(
        win=win,
        name='image_scene_encode', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-200, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_object_encode = visual.ImageStim(
        win=win,
        name='image_object_encode', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(300, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    text_rating = visual.TextStim(win=win, name='text_rating',
        text='Qual o nível de congruência / compatibilidade entre o par?\n',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-3.0);
    slider_cong = visual.Slider(win=win, name='slider_cong',
        startValue=None, size=(0.8, 0.04), pos=(0, -0.3), units=win.units,
        labels=['1 - Incongruente','2 - Intermédio','3 - Congruente'], ticks=(1, 2, 3), granularity=1.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Ariel', labelHeight=0.04,
        flip=False, ori=0.0, depth=-4, readOnly=False)
    key_rate = keyboard.Keyboard(deviceName='key_rate')
    
    # --- Initialize components for Routine "Pause_1" ---
    text_pausa1 = visual.TextStim(win=win, name='text_pausa1',
        text='A primeira parte está concluída. Bom trabalho!\nDescansa um momento.\n\nPrime ESPAÇO quando estiveres pronto para a próxima tarefa.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_Start_Cog = keyboard.Keyboard(deviceName='key_Start_Cog')
    
    # --- Initialize components for Routine "Explanation_GoNo" ---
    text_explanation_GoNo = visual.TextStim(win=win, name='text_explanation_GoNo',
        text='INSERT EXPLANATION GO/NO-GO',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "Cog_Oral" ---
    text_endGo = visual.TextStim(win=win, name='text_endGo',
        text='Tarefa concluída.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    text_call = visual.TextStim(win=win, name='text_call',
        text='O experimentador vem agora ter contigo para duas tarefas orais.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_end_cog = keyboard.Keyboard(deviceName='key_end_cog')
    
    # --- Initialize components for Routine "Retomar" ---
    text_retomar = visual.TextStim(win=win, name='text_retomar',
        text='Pronto para as próximas tarefas?\nPrime ESPAÇO para a explicação.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "Explanation_Recall" ---
    text_explain_recall = visual.TextStim(win=win, name='text_explain_recall',
        text='Vamos fazer agora duas tarefas para testar a tua memória. \nSerá que te lembras dos pares de imagens?\n\n\nPrime ESPAÇO para continuar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_continue = keyboard.Keyboard(deviceName='key_continue')
    
    # --- Initialize components for Routine "Explain_Old_New" ---
    text_explain_old_new = visual.TextStim(win=win, name='text_explain_old_new',
        text="No primeiro teste de memória, vamos apresentar imagens de cenas.\n\nSe reconheceres a imagem como sendo uma das que viste na primeira tarefa, prime a FLECHA ESQUERDA '<-'\n\nSe a imagem for nova, que não faz parte das imagens que aprendeste, então prime a FLECHA DIREITA '->'\n\nTens 3 SEGUNDOS para responder o mais acertadamente possível. Se não conseguires responder a tempo ou errares alguma(s), não faz mal, continua. Faz o teu melhor!\n\nPrime ESPAÇO para um exemplo",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_explain_old = keyboard.Keyboard(deviceName='key_explain_old')
    
    # --- Initialize components for Routine "blank" ---
    cross_25 = visual.ShapeStim(
        win=win, name='cross_25', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    image_blank = visual.ImageStim(
        win=win,
        name='image_blank', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "Exemplo_old" ---
    image_exemplo_old_new = visual.ImageStim(
        win=win,
        name='image_exemplo_old_new', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_ex_old = keyboard.Keyboard(deviceName='key_ex_old')
    text_ex_old = visual.TextStim(win=win, name='text_ex_old',
        text='Prime <- se a imagem for ANTIGA\n\nPrime -> se a imagem for NOVA',
        font='Arial',
        pos=(0, 0.3), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "Start_Recall" ---
    text_start_recall = visual.TextStim(win=win, name='text_start_recall',
        text='Exemplo concluído. Se tiveres qualquer dúvida, chama o experimentador.\n\n\nPrime ESPAÇO para começar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_start_recall = keyboard.Keyboard(deviceName='key_start_recall')
    
    # --- Initialize components for Routine "blank" ---
    cross_25 = visual.ShapeStim(
        win=win, name='cross_25', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    image_blank = visual.ImageStim(
        win=win,
        name='image_blank', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "Old_New_Task" ---
    image_old_new = visual.ImageStim(
        win=win,
        name='image_old_new', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.12), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_item = keyboard.Keyboard(deviceName='key_item')
    text_item = visual.TextStim(win=win, name='text_item',
        text='Prime <- se a imagem for ANTIGA\n\nPrime -> se a imagem for NOVA',
        font='Arial',
        pos=(0, 0.35), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "End_Old_New" ---
    text_end_item = visual.TextStim(win=win, name='text_end_item',
        text='Tarefa concluída, bom trabalho! Descansa um momento.\n\nSó falta mais uma.\n\nPrime ESPAÇO quando estiveres pronto para a explicação da última tarefa.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "Explain_2AFC" ---
    text_explain_2afc = visual.TextStim(win=win, name='text_explain_2afc',
        text='Vão aparecer três imagens: uma cena que viste anteriormente, e dois objetos. Um dos objetos viste antes associado à cena, durante a primeira tarefa. \n\nSe o objeto associado estiver em cima, prime a FLECHA de CIMA. \nSe o objeto associado for o de baixo, prime a FLECHA de BAIXO.\n\nTens 3 SEGUNDOS para responder o mais acertadamente possível. Se não conseguires responder a tempo ou errares alguma(s), não faz mal, continua. Faz o teu melhor!\n\nPrime ESPAÇO para um exemplo. ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_end_explain_2afc = keyboard.Keyboard(deviceName='key_end_explain_2afc')
    
    # --- Initialize components for Routine "blank" ---
    cross_25 = visual.ShapeStim(
        win=win, name='cross_25', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    image_blank = visual.ImageStim(
        win=win,
        name='image_blank', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "Exemplo_2AFC" ---
    image_ex_2AFC = visual.ImageStim(
        win=win,
        name='image_ex_2AFC', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-200, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    image_ex_target = visual.ImageStim(
        win=win,
        name='image_ex_target', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    image_ex_lure = visual.ImageStim(
        win=win,
        name='image_ex_lure', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    key_ex_2AFC = keyboard.Keyboard(deviceName='key_ex_2AFC')
    
    # --- Initialize components for Routine "Start_Recall" ---
    text_start_recall = visual.TextStim(win=win, name='text_start_recall',
        text='Exemplo concluído. Se tiveres qualquer dúvida, chama o experimentador.\n\n\nPrime ESPAÇO para começar',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_start_recall = keyboard.Keyboard(deviceName='key_start_recall')
    
    # --- Initialize components for Routine "blank" ---
    cross_25 = visual.ShapeStim(
        win=win, name='cross_25', vertices='cross',
        size=(0.25, 0.25),
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor='white', fillColor='white',
        opacity=None, depth=0.0, interpolate=True)
    image_blank = visual.ImageStim(
        win=win,
        name='image_blank', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    
    # --- Initialize components for Routine "trial_2AFC" ---
    image_scene_2AFC = visual.ImageStim(
        win=win,
        name='image_scene_2AFC', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(-250, 0), draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    key_2AFC = keyboard.Keyboard(deviceName='key_2AFC')
    image_target = visual.ImageStim(
        win=win,
        name='image_target', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=True, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    image_lure = visual.ImageStim(
        win=win,
        name='image_lure', units='pix', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=[0,0], draggable=False, size=1.0,
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    
    # --- Initialize components for Routine "end" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='Experiência concluída.\nObrigada por participares!',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "Go_Nogo_task" ---
    # create an object to store info about Routine Go_Nogo_task
    Go_Nogo_task = data.Routine(
        name='Go_Nogo_task',
        components=[],
    )
    Go_Nogo_task.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_GoNo
    # Importa a função
    import sys
    sys.path.append('caminho/para/pasta')  
    from Go_NoGo import run_gonogo_task
    
    # 1. Roda a tarefa – TODOS os 100 trials numa única chamada
    resultados = run_gonogo_task(win)
    
    
    # 2. Se a tarefa correu bem, calcula estatísticas e adiciona ao ficheiro do Builder
    if resultados is not None and len(resultados) > 0:
        import pandas as pd
        df = pd.DataFrame(resultados)
        
        #Estatísticas globais
        total_go    = df[df['trial_type'] == 'go'].shape[0]
        total_nogo  = df[df['trial_type'] == 'nogo'].shape[0]
        
        hits        = df[(df['trial_type'] == 'go') & (df['correct'] == 1)].shape[0]
        misses      = df[(df['trial_type'] == 'go') & (df['correct'] == 0)].shape[0]
        corr_reject = df[(df['trial_type'] == 'nogo') & (df['correct'] == 1)].shape[0]
        false_alarms= df[(df['trial_type'] == 'nogo') & (df['correct'] == 0)].shape[0]
        
        # Tempo médio de reação (apenas respostas corretas GO)
        rt_go_correct = df[(df['trial_type'] == 'go') & (df['correct'] == 1)]['rt']
        mean_rt = rt_go_correct.mean() if len(rt_go_correct) > 0 else 0
        
        #Adiciona ao ficheiro do Builder 
        thisExp.addData('n_trials_total', len(df))
        thisExp.addData('n_go', total_go)
        thisExp.addData('n_nogo', total_nogo)
        thisExp.addData('hits', hits)
        thisExp.addData('misses', misses)
        thisExp.addData('correct_rejections', corr_reject)
        thisExp.addData('false_alarms', false_alarms)
        thisExp.addData('go_accuracy_percent', hits/total_go*100 if total_go>0 else 0)
        thisExp.addData('nogo_accuracy_percent', corr_reject/total_nogo*100 if total_nogo>0 else 0)
        thisExp.addData('mean_rt_go_correct', mean_rt)
        
        print(f" Tarefa concluída. {len(resultados)} ensaios registados.")
        print(" Estatísticas adicionadas ao ficheiro do Builder.")
        
    else:
        print("⚠️ Tarefa cancelada pelo utilizador ou erro – nada adicionado ao Builder.")
    # store start times for Go_Nogo_task
    Go_Nogo_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Go_Nogo_task.tStart = globalClock.getTime(format='float')
    Go_Nogo_task.status = STARTED
    thisExp.addData('Go_Nogo_task.started', Go_Nogo_task.tStart)
    Go_Nogo_task.maxDuration = None
    # keep track of which components have finished
    Go_Nogo_taskComponents = Go_Nogo_task.components
    for thisComponent in Go_Nogo_task.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Go_Nogo_task" ---
    Go_Nogo_task.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Go_Nogo_task.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Go_Nogo_task.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Go_Nogo_task" ---
    for thisComponent in Go_Nogo_task.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Go_Nogo_task
    Go_Nogo_task.tStop = globalClock.getTime(format='float')
    Go_Nogo_task.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Go_Nogo_task.stopped', Go_Nogo_task.tStop)
    thisExp.nextEntry()
    # the Routine "Go_Nogo_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Welcome" ---
    # create an object to store info about Routine Welcome
    Welcome = data.Routine(
        name='Welcome',
        components=[text_hi, key_welcome],
    )
    Welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_welcome
    key_welcome.keys = []
    key_welcome.rt = []
    _key_welcome_allKeys = []
    # store start times for Welcome
    Welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Welcome.tStart = globalClock.getTime(format='float')
    Welcome.status = STARTED
    thisExp.addData('Welcome.started', Welcome.tStart)
    Welcome.maxDuration = None
    # keep track of which components have finished
    WelcomeComponents = Welcome.components
    for thisComponent in Welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Welcome" ---
    Welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_hi* updates
        
        # if text_hi is starting this frame...
        if text_hi.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_hi.frameNStart = frameN  # exact frame index
            text_hi.tStart = t  # local t and not account for scr refresh
            text_hi.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_hi, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_hi.started')
            # update status
            text_hi.status = STARTED
            text_hi.setAutoDraw(True)
        
        # if text_hi is active this frame...
        if text_hi.status == STARTED:
            # update params
            pass
        
        # *key_welcome* updates
        waitOnFlip = False
        
        # if key_welcome is starting this frame...
        if key_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_welcome.frameNStart = frameN  # exact frame index
            key_welcome.tStart = t  # local t and not account for scr refresh
            key_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_welcome.started')
            # update status
            key_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_welcome.status == STARTED and not waitOnFlip:
            theseKeys = key_welcome.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_welcome_allKeys.extend(theseKeys)
            if len(_key_welcome_allKeys):
                key_welcome.keys = _key_welcome_allKeys[-1].name  # just the last key pressed
                key_welcome.rt = _key_welcome_allKeys[-1].rt
                key_welcome.duration = _key_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Welcome" ---
    for thisComponent in Welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Welcome
    Welcome.tStop = globalClock.getTime(format='float')
    Welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Welcome.stopped', Welcome.tStop)
    # check responses
    if key_welcome.keys in ['', [], None]:  # No response was made
        key_welcome.keys = None
    thisExp.addData('key_welcome.keys',key_welcome.keys)
    if key_welcome.keys != None:  # we had a response
        thisExp.addData('key_welcome.rt', key_welcome.rt)
        thisExp.addData('key_welcome.duration', key_welcome.duration)
    thisExp.nextEntry()
    # the Routine "Welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Explanation_PAL" ---
    # create an object to store info about Routine Explanation_PAL
    Explanation_PAL = data.Routine(
        name='Explanation_PAL',
        components=[text_Explain_PAL, key_explain_PAL],
    )
    Explanation_PAL.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_explain_PAL
    key_explain_PAL.keys = []
    key_explain_PAL.rt = []
    _key_explain_PAL_allKeys = []
    # store start times for Explanation_PAL
    Explanation_PAL.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Explanation_PAL.tStart = globalClock.getTime(format='float')
    Explanation_PAL.status = STARTED
    thisExp.addData('Explanation_PAL.started', Explanation_PAL.tStart)
    Explanation_PAL.maxDuration = None
    # keep track of which components have finished
    Explanation_PALComponents = Explanation_PAL.components
    for thisComponent in Explanation_PAL.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Explanation_PAL" ---
    Explanation_PAL.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_Explain_PAL* updates
        
        # if text_Explain_PAL is starting this frame...
        if text_Explain_PAL.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_Explain_PAL.frameNStart = frameN  # exact frame index
            text_Explain_PAL.tStart = t  # local t and not account for scr refresh
            text_Explain_PAL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_Explain_PAL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_Explain_PAL.started')
            # update status
            text_Explain_PAL.status = STARTED
            text_Explain_PAL.setAutoDraw(True)
        
        # if text_Explain_PAL is active this frame...
        if text_Explain_PAL.status == STARTED:
            # update params
            pass
        
        # *key_explain_PAL* updates
        waitOnFlip = False
        
        # if key_explain_PAL is starting this frame...
        if key_explain_PAL.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            key_explain_PAL.frameNStart = frameN  # exact frame index
            key_explain_PAL.tStart = t  # local t and not account for scr refresh
            key_explain_PAL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_explain_PAL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_explain_PAL.started')
            # update status
            key_explain_PAL.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_explain_PAL.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_explain_PAL.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_explain_PAL.status == STARTED and not waitOnFlip:
            theseKeys = key_explain_PAL.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_explain_PAL_allKeys.extend(theseKeys)
            if len(_key_explain_PAL_allKeys):
                key_explain_PAL.keys = _key_explain_PAL_allKeys[-1].name  # just the last key pressed
                key_explain_PAL.rt = _key_explain_PAL_allKeys[-1].rt
                key_explain_PAL.duration = _key_explain_PAL_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Explanation_PAL.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Explanation_PAL.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Explanation_PAL" ---
    for thisComponent in Explanation_PAL.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Explanation_PAL
    Explanation_PAL.tStop = globalClock.getTime(format='float')
    Explanation_PAL.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Explanation_PAL.stopped', Explanation_PAL.tStop)
    # check responses
    if key_explain_PAL.keys in ['', [], None]:  # No response was made
        key_explain_PAL.keys = None
    thisExp.addData('key_explain_PAL.keys',key_explain_PAL.keys)
    if key_explain_PAL.keys != None:  # we had a response
        thisExp.addData('key_explain_PAL.rt', key_explain_PAL.rt)
        thisExp.addData('key_explain_PAL.duration', key_explain_PAL.duration)
    thisExp.nextEntry()
    # the Routine "Explanation_PAL" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Explain_Rating" ---
    # create an object to store info about Routine Explain_Rating
    Explain_Rating = data.Routine(
        name='Explain_Rating',
        components=[text_explain_rate, key_explain_rate],
    )
    Explain_Rating.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_explain_rate
    key_explain_rate.keys = []
    key_explain_rate.rt = []
    _key_explain_rate_allKeys = []
    # store start times for Explain_Rating
    Explain_Rating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Explain_Rating.tStart = globalClock.getTime(format='float')
    Explain_Rating.status = STARTED
    thisExp.addData('Explain_Rating.started', Explain_Rating.tStart)
    Explain_Rating.maxDuration = None
    # keep track of which components have finished
    Explain_RatingComponents = Explain_Rating.components
    for thisComponent in Explain_Rating.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Explain_Rating" ---
    Explain_Rating.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_explain_rate* updates
        
        # if text_explain_rate is starting this frame...
        if text_explain_rate.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_explain_rate.frameNStart = frameN  # exact frame index
            text_explain_rate.tStart = t  # local t and not account for scr refresh
            text_explain_rate.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_explain_rate, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_explain_rate.started')
            # update status
            text_explain_rate.status = STARTED
            text_explain_rate.setAutoDraw(True)
        
        # if text_explain_rate is active this frame...
        if text_explain_rate.status == STARTED:
            # update params
            pass
        
        # *key_explain_rate* updates
        waitOnFlip = False
        
        # if key_explain_rate is starting this frame...
        if key_explain_rate.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            key_explain_rate.frameNStart = frameN  # exact frame index
            key_explain_rate.tStart = t  # local t and not account for scr refresh
            key_explain_rate.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_explain_rate, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_explain_rate.started')
            # update status
            key_explain_rate.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_explain_rate.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_explain_rate.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_explain_rate.status == STARTED and not waitOnFlip:
            theseKeys = key_explain_rate.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_explain_rate_allKeys.extend(theseKeys)
            if len(_key_explain_rate_allKeys):
                key_explain_rate.keys = _key_explain_rate_allKeys[-1].name  # just the last key pressed
                key_explain_rate.rt = _key_explain_rate_allKeys[-1].rt
                key_explain_rate.duration = _key_explain_rate_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Explain_Rating.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Explain_Rating.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Explain_Rating" ---
    for thisComponent in Explain_Rating.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Explain_Rating
    Explain_Rating.tStop = globalClock.getTime(format='float')
    Explain_Rating.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Explain_Rating.stopped', Explain_Rating.tStop)
    # check responses
    if key_explain_rate.keys in ['', [], None]:  # No response was made
        key_explain_rate.keys = None
    thisExp.addData('key_explain_rate.keys',key_explain_rate.keys)
    if key_explain_rate.keys != None:  # we had a response
        thisExp.addData('key_explain_rate.rt', key_explain_rate.rt)
        thisExp.addData('key_explain_rate.duration', key_explain_rate.duration)
    thisExp.nextEntry()
    # the Routine "Explain_Rating" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    exemplo_PAL = data.TrialHandler2(
        name='exemplo_PAL',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Exemplo_encode.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(exemplo_PAL)  # add the loop to the experiment
    thisExemplo_PAL = exemplo_PAL.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisExemplo_PAL.rgb)
    if thisExemplo_PAL != None:
        for paramName in thisExemplo_PAL:
            globals()[paramName] = thisExemplo_PAL[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisExemplo_PAL in exemplo_PAL:
        currentLoop = exemplo_PAL
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisExemplo_PAL.rgb)
        if thisExemplo_PAL != None:
            for paramName in thisExemplo_PAL:
                globals()[paramName] = thisExemplo_PAL[paramName]
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[cross_25, image_blank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(exemplo_PAL, data.TrialHandler2) and thisExemplo_PAL.thisN != exemplo_PAL.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_25* updates
            
            # if cross_25 is starting this frame...
            if cross_25.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                cross_25.frameNStart = frameN  # exact frame index
                cross_25.tStart = t  # local t and not account for scr refresh
                cross_25.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_25, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_25.started')
                # update status
                cross_25.status = STARTED
                cross_25.setAutoDraw(True)
            
            # if cross_25 is active this frame...
            if cross_25.status == STARTED:
                # update params
                pass
            
            # if cross_25 is stopping this frame...
            if cross_25.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_25.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_25.tStop = t  # not accounting for scr refresh
                    cross_25.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_25.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_25.stopped')
                    # update status
                    cross_25.status = FINISHED
                    cross_25.setAutoDraw(False)
            
            # *image_blank* updates
            
            # if image_blank is starting this frame...
            if image_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_blank.frameNStart = frameN  # exact frame index
                image_blank.tStart = t  # local t and not account for scr refresh
                image_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_blank.started')
                # update status
                image_blank.status = STARTED
                image_blank.setAutoDraw(True)
            
            # if image_blank is active this frame...
            if image_blank.status == STARTED:
                # update params
                pass
            
            # if image_blank is stopping this frame...
            if image_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_blank.tStartRefresh + jitter_ITI-frameTolerance:
                    # keep track of stop time/frame for later
                    image_blank.tStop = t  # not accounting for scr refresh
                    image_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    image_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_blank.stopped')
                    # update status
                    image_blank.status = FINISHED
                    image_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # the Routine "blank" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Example_PAL" ---
        # create an object to store info about Routine Example_PAL
        Example_PAL = data.Routine(
            name='Example_PAL',
            components=[image_ex_scene, image_ex_object, slider_example, text_example, key_ex_rate],
        )
        Example_PAL.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_ex_scene.setSize(scene_size)
        image_ex_scene.setImage(scene_encode)
        image_ex_object.setSize(object_size)
        image_ex_object.setImage(object_encode)
        slider_example.reset()
        # create starting attributes for key_ex_rate
        key_ex_rate.keys = []
        key_ex_rate.rt = []
        _key_ex_rate_allKeys = []
        # store start times for Example_PAL
        Example_PAL.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Example_PAL.tStart = globalClock.getTime(format='float')
        Example_PAL.status = STARTED
        thisExp.addData('Example_PAL.started', Example_PAL.tStart)
        Example_PAL.maxDuration = None
        # keep track of which components have finished
        Example_PALComponents = Example_PAL.components
        for thisComponent in Example_PAL.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Example_PAL" ---
        # if trial has changed, end Routine now
        if isinstance(exemplo_PAL, data.TrialHandler2) and thisExemplo_PAL.thisN != exemplo_PAL.thisTrial.thisN:
            continueRoutine = False
        Example_PAL.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_ex_scene* updates
            
            # if image_ex_scene is starting this frame...
            if image_ex_scene.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_ex_scene.frameNStart = frameN  # exact frame index
                image_ex_scene.tStart = t  # local t and not account for scr refresh
                image_ex_scene.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_ex_scene, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_ex_scene.started')
                # update status
                image_ex_scene.status = STARTED
                image_ex_scene.setAutoDraw(True)
            
            # if image_ex_scene is active this frame...
            if image_ex_scene.status == STARTED:
                # update params
                pass
            
            # *image_ex_object* updates
            
            # if image_ex_object is starting this frame...
            if image_ex_object.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_ex_object.frameNStart = frameN  # exact frame index
                image_ex_object.tStart = t  # local t and not account for scr refresh
                image_ex_object.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_ex_object, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_ex_object.started')
                # update status
                image_ex_object.status = STARTED
                image_ex_object.setAutoDraw(True)
            
            # if image_ex_object is active this frame...
            if image_ex_object.status == STARTED:
                # update params
                pass
            
            # *slider_example* updates
            
            # if slider_example is starting this frame...
            if slider_example.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                slider_example.frameNStart = frameN  # exact frame index
                slider_example.tStart = t  # local t and not account for scr refresh
                slider_example.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_example, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_example.started')
                # update status
                slider_example.status = STARTED
                slider_example.setAutoDraw(True)
            
            # if slider_example is active this frame...
            if slider_example.status == STARTED:
                # update params
                pass
            
            # *text_example* updates
            
            # if text_example is starting this frame...
            if text_example.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                text_example.frameNStart = frameN  # exact frame index
                text_example.tStart = t  # local t and not account for scr refresh
                text_example.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_example, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_example.started')
                # update status
                text_example.status = STARTED
                text_example.setAutoDraw(True)
            
            # if text_example is active this frame...
            if text_example.status == STARTED:
                # update params
                pass
            
            # *key_ex_rate* updates
            waitOnFlip = False
            
            # if key_ex_rate is starting this frame...
            if key_ex_rate.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                key_ex_rate.frameNStart = frameN  # exact frame index
                key_ex_rate.tStart = t  # local t and not account for scr refresh
                key_ex_rate.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_ex_rate, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_ex_rate.started')
                # update status
                key_ex_rate.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_ex_rate.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_ex_rate.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_ex_rate.status == STARTED and not waitOnFlip:
                theseKeys = key_ex_rate.getKeys(keyList=['z','x','c','num_1','num_2','num_3'], ignoreKeys=["escape"], waitRelease=False)
                _key_ex_rate_allKeys.extend(theseKeys)
                if len(_key_ex_rate_allKeys):
                    key_ex_rate.keys = _key_ex_rate_allKeys[-1].name  # just the last key pressed
                    key_ex_rate.rt = _key_ex_rate_allKeys[-1].rt
                    key_ex_rate.duration = _key_ex_rate_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Example_PAL.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Example_PAL.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Example_PAL" ---
        for thisComponent in Example_PAL.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Example_PAL
        Example_PAL.tStop = globalClock.getTime(format='float')
        Example_PAL.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Example_PAL.stopped', Example_PAL.tStop)
        exemplo_PAL.addData('slider_example.response', slider_example.getRating())
        exemplo_PAL.addData('slider_example.rt', slider_example.getRT())
        # check responses
        if key_ex_rate.keys in ['', [], None]:  # No response was made
            key_ex_rate.keys = None
        exemplo_PAL.addData('key_ex_rate.keys',key_ex_rate.keys)
        if key_ex_rate.keys != None:  # we had a response
            exemplo_PAL.addData('key_ex_rate.rt', key_ex_rate.rt)
            exemplo_PAL.addData('key_ex_rate.duration', key_ex_rate.duration)
        # the Routine "Example_PAL" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'exemplo_PAL'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Start_PAL" ---
    # create an object to store info about Routine Start_PAL
    Start_PAL = data.Routine(
        name='Start_PAL',
        components=[text_end_ex, text_start_PAL, key_start_PAL],
    )
    Start_PAL.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_start_PAL
    key_start_PAL.keys = []
    key_start_PAL.rt = []
    _key_start_PAL_allKeys = []
    # store start times for Start_PAL
    Start_PAL.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start_PAL.tStart = globalClock.getTime(format='float')
    Start_PAL.status = STARTED
    thisExp.addData('Start_PAL.started', Start_PAL.tStart)
    Start_PAL.maxDuration = None
    # keep track of which components have finished
    Start_PALComponents = Start_PAL.components
    for thisComponent in Start_PAL.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start_PAL" ---
    Start_PAL.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end_ex* updates
        
        # if text_end_ex is starting this frame...
        if text_end_ex.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end_ex.frameNStart = frameN  # exact frame index
            text_end_ex.tStart = t  # local t and not account for scr refresh
            text_end_ex.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end_ex, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end_ex.started')
            # update status
            text_end_ex.status = STARTED
            text_end_ex.setAutoDraw(True)
        
        # if text_end_ex is active this frame...
        if text_end_ex.status == STARTED:
            # update params
            pass
        
        # if text_end_ex is stopping this frame...
        if text_end_ex.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end_ex.tStartRefresh + 1.5-frameTolerance:
                # keep track of stop time/frame for later
                text_end_ex.tStop = t  # not accounting for scr refresh
                text_end_ex.tStopRefresh = tThisFlipGlobal  # on global time
                text_end_ex.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_end_ex.stopped')
                # update status
                text_end_ex.status = FINISHED
                text_end_ex.setAutoDraw(False)
        
        # *text_start_PAL* updates
        
        # if text_start_PAL is starting this frame...
        if text_start_PAL.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
            # keep track of start time/frame for later
            text_start_PAL.frameNStart = frameN  # exact frame index
            text_start_PAL.tStart = t  # local t and not account for scr refresh
            text_start_PAL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start_PAL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_start_PAL.started')
            # update status
            text_start_PAL.status = STARTED
            text_start_PAL.setAutoDraw(True)
        
        # if text_start_PAL is active this frame...
        if text_start_PAL.status == STARTED:
            # update params
            pass
        
        # *key_start_PAL* updates
        waitOnFlip = False
        
        # if key_start_PAL is starting this frame...
        if key_start_PAL.status == NOT_STARTED and tThisFlip >= 1.5-frameTolerance:
            # keep track of start time/frame for later
            key_start_PAL.frameNStart = frameN  # exact frame index
            key_start_PAL.tStart = t  # local t and not account for scr refresh
            key_start_PAL.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_start_PAL, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_start_PAL.started')
            # update status
            key_start_PAL.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_start_PAL.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_start_PAL.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_start_PAL.status == STARTED and not waitOnFlip:
            theseKeys = key_start_PAL.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_start_PAL_allKeys.extend(theseKeys)
            if len(_key_start_PAL_allKeys):
                key_start_PAL.keys = _key_start_PAL_allKeys[-1].name  # just the last key pressed
                key_start_PAL.rt = _key_start_PAL_allKeys[-1].rt
                key_start_PAL.duration = _key_start_PAL_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Start_PAL.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Start_PAL.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_PAL" ---
    for thisComponent in Start_PAL.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start_PAL
    Start_PAL.tStop = globalClock.getTime(format='float')
    Start_PAL.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start_PAL.stopped', Start_PAL.tStop)
    # check responses
    if key_start_PAL.keys in ['', [], None]:  # No response was made
        key_start_PAL.keys = None
    thisExp.addData('key_start_PAL.keys',key_start_PAL.keys)
    if key_start_PAL.keys != None:  # we had a response
        thisExp.addData('key_start_PAL.rt', key_start_PAL.rt)
        thisExp.addData('key_start_PAL.duration', key_start_PAL.duration)
    thisExp.nextEntry()
    # the Routine "Start_PAL" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_encode = data.TrialHandler2(
        name='trial_encode',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'encode_pairs.xlsx', 
        selection='0:60'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(trial_encode)  # add the loop to the experiment
    thisTrial_encode = trial_encode.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_encode.rgb)
    if thisTrial_encode != None:
        for paramName in thisTrial_encode:
            globals()[paramName] = thisTrial_encode[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_encode in trial_encode:
        currentLoop = trial_encode
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_encode.rgb)
        if thisTrial_encode != None:
            for paramName in thisTrial_encode:
                globals()[paramName] = thisTrial_encode[paramName]
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[cross_25, image_blank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(trial_encode, data.TrialHandler2) and thisTrial_encode.thisN != trial_encode.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_25* updates
            
            # if cross_25 is starting this frame...
            if cross_25.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                cross_25.frameNStart = frameN  # exact frame index
                cross_25.tStart = t  # local t and not account for scr refresh
                cross_25.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_25, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_25.started')
                # update status
                cross_25.status = STARTED
                cross_25.setAutoDraw(True)
            
            # if cross_25 is active this frame...
            if cross_25.status == STARTED:
                # update params
                pass
            
            # if cross_25 is stopping this frame...
            if cross_25.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_25.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_25.tStop = t  # not accounting for scr refresh
                    cross_25.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_25.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_25.stopped')
                    # update status
                    cross_25.status = FINISHED
                    cross_25.setAutoDraw(False)
            
            # *image_blank* updates
            
            # if image_blank is starting this frame...
            if image_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_blank.frameNStart = frameN  # exact frame index
                image_blank.tStart = t  # local t and not account for scr refresh
                image_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_blank.started')
                # update status
                image_blank.status = STARTED
                image_blank.setAutoDraw(True)
            
            # if image_blank is active this frame...
            if image_blank.status == STARTED:
                # update params
                pass
            
            # if image_blank is stopping this frame...
            if image_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_blank.tStartRefresh + jitter_ITI-frameTolerance:
                    # keep track of stop time/frame for later
                    image_blank.tStop = t  # not accounting for scr refresh
                    image_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    image_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_blank.stopped')
                    # update status
                    image_blank.status = FINISHED
                    image_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # the Routine "blank" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Encode" ---
        # create an object to store info about Routine Encode
        Encode = data.Routine(
            name='Encode',
            components=[image_scene_encode, image_object_encode, text_rating, slider_cong, key_rate],
        )
        Encode.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_scene_encode.setSize(scene_size)
        image_scene_encode.setImage(scene)
        image_object_encode.setSize(object_size)
        image_object_encode.setImage(object_encode)
        # Run 'Begin Routine' code from code_debugPAL
        print(f"trial {trial_encode.thisN}, image = {scene}")
        print(f"trial {trial_encode.thisN}, image = {object_encode}")
        
        slider_cong.reset()
        # create starting attributes for key_rate
        key_rate.keys = []
        key_rate.rt = []
        _key_rate_allKeys = []
        # store start times for Encode
        Encode.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Encode.tStart = globalClock.getTime(format='float')
        Encode.status = STARTED
        thisExp.addData('Encode.started', Encode.tStart)
        Encode.maxDuration = None
        # keep track of which components have finished
        EncodeComponents = Encode.components
        for thisComponent in Encode.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Encode" ---
        # if trial has changed, end Routine now
        if isinstance(trial_encode, data.TrialHandler2) and thisTrial_encode.thisN != trial_encode.thisTrial.thisN:
            continueRoutine = False
        Encode.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_scene_encode* updates
            
            # if image_scene_encode is starting this frame...
            if image_scene_encode.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_scene_encode.frameNStart = frameN  # exact frame index
                image_scene_encode.tStart = t  # local t and not account for scr refresh
                image_scene_encode.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_scene_encode, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_scene_encode.started')
                # update status
                image_scene_encode.status = STARTED
                image_scene_encode.setAutoDraw(True)
            
            # if image_scene_encode is active this frame...
            if image_scene_encode.status == STARTED:
                # update params
                pass
            
            # *image_object_encode* updates
            
            # if image_object_encode is starting this frame...
            if image_object_encode.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_object_encode.frameNStart = frameN  # exact frame index
                image_object_encode.tStart = t  # local t and not account for scr refresh
                image_object_encode.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_object_encode, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_object_encode.started')
                # update status
                image_object_encode.status = STARTED
                image_object_encode.setAutoDraw(True)
            
            # if image_object_encode is active this frame...
            if image_object_encode.status == STARTED:
                # update params
                pass
            
            # *text_rating* updates
            
            # if text_rating is starting this frame...
            if text_rating.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                text_rating.frameNStart = frameN  # exact frame index
                text_rating.tStart = t  # local t and not account for scr refresh
                text_rating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_rating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_rating.started')
                # update status
                text_rating.status = STARTED
                text_rating.setAutoDraw(True)
            
            # if text_rating is active this frame...
            if text_rating.status == STARTED:
                # update params
                pass
            
            # *slider_cong* updates
            
            # if slider_cong is starting this frame...
            if slider_cong.status == NOT_STARTED and tThisFlip >= 2.5-frameTolerance:
                # keep track of start time/frame for later
                slider_cong.frameNStart = frameN  # exact frame index
                slider_cong.tStart = t  # local t and not account for scr refresh
                slider_cong.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(slider_cong, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'slider_cong.started')
                # update status
                slider_cong.status = STARTED
                slider_cong.setAutoDraw(True)
            
            # if slider_cong is active this frame...
            if slider_cong.status == STARTED:
                # update params
                pass
            
            # Check slider_cong for response to end Routine
            if slider_cong.getRating() is not None and slider_cong.status == STARTED:
                continueRoutine = False
            
            # *key_rate* updates
            waitOnFlip = False
            
            # if key_rate is starting this frame...
            if key_rate.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_rate.frameNStart = frameN  # exact frame index
                key_rate.tStart = t  # local t and not account for scr refresh
                key_rate.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_rate, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_rate.started')
                # update status
                key_rate.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_rate.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_rate.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_rate.status == STARTED and not waitOnFlip:
                theseKeys = key_rate.getKeys(keyList=['z','x','c','num_1','num_2','num_3'], ignoreKeys=["escape"], waitRelease=False)
                _key_rate_allKeys.extend(theseKeys)
                if len(_key_rate_allKeys):
                    key_rate.keys = _key_rate_allKeys[-1].name  # just the last key pressed
                    key_rate.rt = _key_rate_allKeys[-1].rt
                    key_rate.duration = _key_rate_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Encode.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Encode.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Encode" ---
        for thisComponent in Encode.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Encode
        Encode.tStop = globalClock.getTime(format='float')
        Encode.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Encode.stopped', Encode.tStop)
        trial_encode.addData('slider_cong.response', slider_cong.getRating())
        trial_encode.addData('slider_cong.rt', slider_cong.getRT())
        # check responses
        if key_rate.keys in ['', [], None]:  # No response was made
            key_rate.keys = None
        trial_encode.addData('key_rate.keys',key_rate.keys)
        if key_rate.keys != None:  # we had a response
            trial_encode.addData('key_rate.rt', key_rate.rt)
            trial_encode.addData('key_rate.duration', key_rate.duration)
        # the Routine "Encode" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trial_encode'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Pause_1" ---
    # create an object to store info about Routine Pause_1
    Pause_1 = data.Routine(
        name='Pause_1',
        components=[text_pausa1, key_Start_Cog],
    )
    Pause_1.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_Start_Cog
    key_Start_Cog.keys = []
    key_Start_Cog.rt = []
    _key_Start_Cog_allKeys = []
    # store start times for Pause_1
    Pause_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Pause_1.tStart = globalClock.getTime(format='float')
    Pause_1.status = STARTED
    thisExp.addData('Pause_1.started', Pause_1.tStart)
    Pause_1.maxDuration = None
    # keep track of which components have finished
    Pause_1Components = Pause_1.components
    for thisComponent in Pause_1.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Pause_1" ---
    Pause_1.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_pausa1* updates
        
        # if text_pausa1 is starting this frame...
        if text_pausa1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_pausa1.frameNStart = frameN  # exact frame index
            text_pausa1.tStart = t  # local t and not account for scr refresh
            text_pausa1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_pausa1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_pausa1.started')
            # update status
            text_pausa1.status = STARTED
            text_pausa1.setAutoDraw(True)
        
        # if text_pausa1 is active this frame...
        if text_pausa1.status == STARTED:
            # update params
            pass
        
        # *key_Start_Cog* updates
        waitOnFlip = False
        
        # if key_Start_Cog is starting this frame...
        if key_Start_Cog.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_Start_Cog.frameNStart = frameN  # exact frame index
            key_Start_Cog.tStart = t  # local t and not account for scr refresh
            key_Start_Cog.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_Start_Cog, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_Start_Cog.started')
            # update status
            key_Start_Cog.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_Start_Cog.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_Start_Cog.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_Start_Cog.status == STARTED and not waitOnFlip:
            theseKeys = key_Start_Cog.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_Start_Cog_allKeys.extend(theseKeys)
            if len(_key_Start_Cog_allKeys):
                key_Start_Cog.keys = _key_Start_Cog_allKeys[-1].name  # just the last key pressed
                key_Start_Cog.rt = _key_Start_Cog_allKeys[-1].rt
                key_Start_Cog.duration = _key_Start_Cog_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Pause_1.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Pause_1.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Pause_1" ---
    for thisComponent in Pause_1.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Pause_1
    Pause_1.tStop = globalClock.getTime(format='float')
    Pause_1.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Pause_1.stopped', Pause_1.tStop)
    # check responses
    if key_Start_Cog.keys in ['', [], None]:  # No response was made
        key_Start_Cog.keys = None
    thisExp.addData('key_Start_Cog.keys',key_Start_Cog.keys)
    if key_Start_Cog.keys != None:  # we had a response
        thisExp.addData('key_Start_Cog.rt', key_Start_Cog.rt)
        thisExp.addData('key_Start_Cog.duration', key_Start_Cog.duration)
    thisExp.nextEntry()
    # the Routine "Pause_1" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Explanation_GoNo" ---
    # create an object to store info about Routine Explanation_GoNo
    Explanation_GoNo = data.Routine(
        name='Explanation_GoNo',
        components=[text_explanation_GoNo],
    )
    Explanation_GoNo.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for Explanation_GoNo
    Explanation_GoNo.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Explanation_GoNo.tStart = globalClock.getTime(format='float')
    Explanation_GoNo.status = STARTED
    thisExp.addData('Explanation_GoNo.started', Explanation_GoNo.tStart)
    Explanation_GoNo.maxDuration = None
    # keep track of which components have finished
    Explanation_GoNoComponents = Explanation_GoNo.components
    for thisComponent in Explanation_GoNo.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Explanation_GoNo" ---
    Explanation_GoNo.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 1.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_explanation_GoNo* updates
        
        # if text_explanation_GoNo is starting this frame...
        if text_explanation_GoNo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_explanation_GoNo.frameNStart = frameN  # exact frame index
            text_explanation_GoNo.tStart = t  # local t and not account for scr refresh
            text_explanation_GoNo.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_explanation_GoNo, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_explanation_GoNo.started')
            # update status
            text_explanation_GoNo.status = STARTED
            text_explanation_GoNo.setAutoDraw(True)
        
        # if text_explanation_GoNo is active this frame...
        if text_explanation_GoNo.status == STARTED:
            # update params
            pass
        
        # if text_explanation_GoNo is stopping this frame...
        if text_explanation_GoNo.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_explanation_GoNo.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_explanation_GoNo.tStop = t  # not accounting for scr refresh
                text_explanation_GoNo.tStopRefresh = tThisFlipGlobal  # on global time
                text_explanation_GoNo.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_explanation_GoNo.stopped')
                # update status
                text_explanation_GoNo.status = FINISHED
                text_explanation_GoNo.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Explanation_GoNo.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Explanation_GoNo.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Explanation_GoNo" ---
    for thisComponent in Explanation_GoNo.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Explanation_GoNo
    Explanation_GoNo.tStop = globalClock.getTime(format='float')
    Explanation_GoNo.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Explanation_GoNo.stopped', Explanation_GoNo.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if Explanation_GoNo.maxDurationReached:
        routineTimer.addTime(-Explanation_GoNo.maxDuration)
    elif Explanation_GoNo.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-1.000000)
    thisExp.nextEntry()
    
    # --- Prepare to start Routine "Cog_Oral" ---
    # create an object to store info about Routine Cog_Oral
    Cog_Oral = data.Routine(
        name='Cog_Oral',
        components=[text_endGo, text_call, key_end_cog],
    )
    Cog_Oral.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_end_cog
    key_end_cog.keys = []
    key_end_cog.rt = []
    _key_end_cog_allKeys = []
    # store start times for Cog_Oral
    Cog_Oral.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Cog_Oral.tStart = globalClock.getTime(format='float')
    Cog_Oral.status = STARTED
    thisExp.addData('Cog_Oral.started', Cog_Oral.tStart)
    Cog_Oral.maxDuration = None
    # keep track of which components have finished
    Cog_OralComponents = Cog_Oral.components
    for thisComponent in Cog_Oral.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Cog_Oral" ---
    Cog_Oral.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_endGo* updates
        
        # if text_endGo is starting this frame...
        if text_endGo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_endGo.frameNStart = frameN  # exact frame index
            text_endGo.tStart = t  # local t and not account for scr refresh
            text_endGo.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_endGo, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_endGo.started')
            # update status
            text_endGo.status = STARTED
            text_endGo.setAutoDraw(True)
        
        # if text_endGo is active this frame...
        if text_endGo.status == STARTED:
            # update params
            pass
        
        # if text_endGo is stopping this frame...
        if text_endGo.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_endGo.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                text_endGo.tStop = t  # not accounting for scr refresh
                text_endGo.tStopRefresh = tThisFlipGlobal  # on global time
                text_endGo.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_endGo.stopped')
                # update status
                text_endGo.status = FINISHED
                text_endGo.setAutoDraw(False)
        
        # *text_call* updates
        
        # if text_call is starting this frame...
        if text_call.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
            # keep track of start time/frame for later
            text_call.frameNStart = frameN  # exact frame index
            text_call.tStart = t  # local t and not account for scr refresh
            text_call.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_call, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_call.started')
            # update status
            text_call.status = STARTED
            text_call.setAutoDraw(True)
        
        # if text_call is active this frame...
        if text_call.status == STARTED:
            # update params
            pass
        
        # if text_call is stopping this frame...
        if text_call.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_call.tStartRefresh + 10-frameTolerance:
                # keep track of stop time/frame for later
                text_call.tStop = t  # not accounting for scr refresh
                text_call.tStopRefresh = tThisFlipGlobal  # on global time
                text_call.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_call.stopped')
                # update status
                text_call.status = FINISHED
                text_call.setAutoDraw(False)
        
        # *key_end_cog* updates
        waitOnFlip = False
        
        # if key_end_cog is starting this frame...
        if key_end_cog.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
            # keep track of start time/frame for later
            key_end_cog.frameNStart = frameN  # exact frame index
            key_end_cog.tStart = t  # local t and not account for scr refresh
            key_end_cog.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_end_cog, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_end_cog.started')
            # update status
            key_end_cog.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_end_cog.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_end_cog.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_end_cog.status == STARTED and not waitOnFlip:
            theseKeys = key_end_cog.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_end_cog_allKeys.extend(theseKeys)
            if len(_key_end_cog_allKeys):
                key_end_cog.keys = _key_end_cog_allKeys[-1].name  # just the last key pressed
                key_end_cog.rt = _key_end_cog_allKeys[-1].rt
                key_end_cog.duration = _key_end_cog_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Cog_Oral.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Cog_Oral.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Cog_Oral" ---
    for thisComponent in Cog_Oral.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Cog_Oral
    Cog_Oral.tStop = globalClock.getTime(format='float')
    Cog_Oral.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Cog_Oral.stopped', Cog_Oral.tStop)
    # check responses
    if key_end_cog.keys in ['', [], None]:  # No response was made
        key_end_cog.keys = None
    thisExp.addData('key_end_cog.keys',key_end_cog.keys)
    if key_end_cog.keys != None:  # we had a response
        thisExp.addData('key_end_cog.rt', key_end_cog.rt)
        thisExp.addData('key_end_cog.duration', key_end_cog.duration)
    thisExp.nextEntry()
    # the Routine "Cog_Oral" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Retomar" ---
    # create an object to store info about Routine Retomar
    Retomar = data.Routine(
        name='Retomar',
        components=[text_retomar, key_resp],
    )
    Retomar.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for Retomar
    Retomar.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Retomar.tStart = globalClock.getTime(format='float')
    Retomar.status = STARTED
    thisExp.addData('Retomar.started', Retomar.tStart)
    Retomar.maxDuration = None
    # keep track of which components have finished
    RetomarComponents = Retomar.components
    for thisComponent in Retomar.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Retomar" ---
    Retomar.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_retomar* updates
        
        # if text_retomar is starting this frame...
        if text_retomar.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_retomar.frameNStart = frameN  # exact frame index
            text_retomar.tStart = t  # local t and not account for scr refresh
            text_retomar.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_retomar, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_retomar.started')
            # update status
            text_retomar.status = STARTED
            text_retomar.setAutoDraw(True)
        
        # if text_retomar is active this frame...
        if text_retomar.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Retomar.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Retomar.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Retomar" ---
    for thisComponent in Retomar.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Retomar
    Retomar.tStop = globalClock.getTime(format='float')
    Retomar.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Retomar.stopped', Retomar.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "Retomar" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Explanation_Recall" ---
    # create an object to store info about Routine Explanation_Recall
    Explanation_Recall = data.Routine(
        name='Explanation_Recall',
        components=[text_explain_recall, key_continue],
    )
    Explanation_Recall.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_continue
    key_continue.keys = []
    key_continue.rt = []
    _key_continue_allKeys = []
    # store start times for Explanation_Recall
    Explanation_Recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Explanation_Recall.tStart = globalClock.getTime(format='float')
    Explanation_Recall.status = STARTED
    thisExp.addData('Explanation_Recall.started', Explanation_Recall.tStart)
    Explanation_Recall.maxDuration = None
    # keep track of which components have finished
    Explanation_RecallComponents = Explanation_Recall.components
    for thisComponent in Explanation_Recall.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Explanation_Recall" ---
    Explanation_Recall.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_explain_recall* updates
        
        # if text_explain_recall is starting this frame...
        if text_explain_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_explain_recall.frameNStart = frameN  # exact frame index
            text_explain_recall.tStart = t  # local t and not account for scr refresh
            text_explain_recall.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_explain_recall, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_explain_recall.started')
            # update status
            text_explain_recall.status = STARTED
            text_explain_recall.setAutoDraw(True)
        
        # if text_explain_recall is active this frame...
        if text_explain_recall.status == STARTED:
            # update params
            pass
        
        # *key_continue* updates
        waitOnFlip = False
        
        # if key_continue is starting this frame...
        if key_continue.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_continue.frameNStart = frameN  # exact frame index
            key_continue.tStart = t  # local t and not account for scr refresh
            key_continue.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_continue, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_continue.started')
            # update status
            key_continue.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_continue.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_continue.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_continue.status == STARTED and not waitOnFlip:
            theseKeys = key_continue.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_continue_allKeys.extend(theseKeys)
            if len(_key_continue_allKeys):
                key_continue.keys = _key_continue_allKeys[-1].name  # just the last key pressed
                key_continue.rt = _key_continue_allKeys[-1].rt
                key_continue.duration = _key_continue_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Explanation_Recall.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Explanation_Recall.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Explanation_Recall" ---
    for thisComponent in Explanation_Recall.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Explanation_Recall
    Explanation_Recall.tStop = globalClock.getTime(format='float')
    Explanation_Recall.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Explanation_Recall.stopped', Explanation_Recall.tStop)
    # check responses
    if key_continue.keys in ['', [], None]:  # No response was made
        key_continue.keys = None
    thisExp.addData('key_continue.keys',key_continue.keys)
    if key_continue.keys != None:  # we had a response
        thisExp.addData('key_continue.rt', key_continue.rt)
        thisExp.addData('key_continue.duration', key_continue.duration)
    thisExp.nextEntry()
    # the Routine "Explanation_Recall" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Explain_Old_New" ---
    # create an object to store info about Routine Explain_Old_New
    Explain_Old_New = data.Routine(
        name='Explain_Old_New',
        components=[text_explain_old_new, key_explain_old],
    )
    Explain_Old_New.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_explain_old
    key_explain_old.keys = []
    key_explain_old.rt = []
    _key_explain_old_allKeys = []
    # store start times for Explain_Old_New
    Explain_Old_New.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Explain_Old_New.tStart = globalClock.getTime(format='float')
    Explain_Old_New.status = STARTED
    thisExp.addData('Explain_Old_New.started', Explain_Old_New.tStart)
    Explain_Old_New.maxDuration = None
    # keep track of which components have finished
    Explain_Old_NewComponents = Explain_Old_New.components
    for thisComponent in Explain_Old_New.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Explain_Old_New" ---
    Explain_Old_New.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_explain_old_new* updates
        
        # if text_explain_old_new is starting this frame...
        if text_explain_old_new.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_explain_old_new.frameNStart = frameN  # exact frame index
            text_explain_old_new.tStart = t  # local t and not account for scr refresh
            text_explain_old_new.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_explain_old_new, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_explain_old_new.started')
            # update status
            text_explain_old_new.status = STARTED
            text_explain_old_new.setAutoDraw(True)
        
        # if text_explain_old_new is active this frame...
        if text_explain_old_new.status == STARTED:
            # update params
            pass
        
        # *key_explain_old* updates
        waitOnFlip = False
        
        # if key_explain_old is starting this frame...
        if key_explain_old.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_explain_old.frameNStart = frameN  # exact frame index
            key_explain_old.tStart = t  # local t and not account for scr refresh
            key_explain_old.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_explain_old, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_explain_old.started')
            # update status
            key_explain_old.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_explain_old.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_explain_old.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_explain_old.status == STARTED and not waitOnFlip:
            theseKeys = key_explain_old.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_explain_old_allKeys.extend(theseKeys)
            if len(_key_explain_old_allKeys):
                key_explain_old.keys = _key_explain_old_allKeys[-1].name  # just the last key pressed
                key_explain_old.rt = _key_explain_old_allKeys[-1].rt
                key_explain_old.duration = _key_explain_old_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Explain_Old_New.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Explain_Old_New.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Explain_Old_New" ---
    for thisComponent in Explain_Old_New.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Explain_Old_New
    Explain_Old_New.tStop = globalClock.getTime(format='float')
    Explain_Old_New.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Explain_Old_New.stopped', Explain_Old_New.tStop)
    # check responses
    if key_explain_old.keys in ['', [], None]:  # No response was made
        key_explain_old.keys = None
    thisExp.addData('key_explain_old.keys',key_explain_old.keys)
    if key_explain_old.keys != None:  # we had a response
        thisExp.addData('key_explain_old.rt', key_explain_old.rt)
        thisExp.addData('key_explain_old.duration', key_explain_old.duration)
    thisExp.nextEntry()
    # the Routine "Explain_Old_New" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_ex_old = data.TrialHandler2(
        name='trial_ex_old',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Exemplo_old.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trial_ex_old)  # add the loop to the experiment
    thisTrial_ex_old = trial_ex_old.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_ex_old.rgb)
    if thisTrial_ex_old != None:
        for paramName in thisTrial_ex_old:
            globals()[paramName] = thisTrial_ex_old[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_ex_old in trial_ex_old:
        currentLoop = trial_ex_old
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_ex_old.rgb)
        if thisTrial_ex_old != None:
            for paramName in thisTrial_ex_old:
                globals()[paramName] = thisTrial_ex_old[paramName]
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[cross_25, image_blank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(trial_ex_old, data.TrialHandler2) and thisTrial_ex_old.thisN != trial_ex_old.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_25* updates
            
            # if cross_25 is starting this frame...
            if cross_25.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                cross_25.frameNStart = frameN  # exact frame index
                cross_25.tStart = t  # local t and not account for scr refresh
                cross_25.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_25, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_25.started')
                # update status
                cross_25.status = STARTED
                cross_25.setAutoDraw(True)
            
            # if cross_25 is active this frame...
            if cross_25.status == STARTED:
                # update params
                pass
            
            # if cross_25 is stopping this frame...
            if cross_25.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_25.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_25.tStop = t  # not accounting for scr refresh
                    cross_25.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_25.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_25.stopped')
                    # update status
                    cross_25.status = FINISHED
                    cross_25.setAutoDraw(False)
            
            # *image_blank* updates
            
            # if image_blank is starting this frame...
            if image_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_blank.frameNStart = frameN  # exact frame index
                image_blank.tStart = t  # local t and not account for scr refresh
                image_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_blank.started')
                # update status
                image_blank.status = STARTED
                image_blank.setAutoDraw(True)
            
            # if image_blank is active this frame...
            if image_blank.status == STARTED:
                # update params
                pass
            
            # if image_blank is stopping this frame...
            if image_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_blank.tStartRefresh + jitter_ITI-frameTolerance:
                    # keep track of stop time/frame for later
                    image_blank.tStop = t  # not accounting for scr refresh
                    image_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    image_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_blank.stopped')
                    # update status
                    image_blank.status = FINISHED
                    image_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # the Routine "blank" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Exemplo_old" ---
        # create an object to store info about Routine Exemplo_old
        Exemplo_old = data.Routine(
            name='Exemplo_old',
            components=[image_exemplo_old_new, key_ex_old, text_ex_old],
        )
        Exemplo_old.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_exemplo_old_new.setSize(scene_size)
        image_exemplo_old_new.setImage(ex_scene_old_new)
        # create starting attributes for key_ex_old
        key_ex_old.keys = []
        key_ex_old.rt = []
        _key_ex_old_allKeys = []
        # store start times for Exemplo_old
        Exemplo_old.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Exemplo_old.tStart = globalClock.getTime(format='float')
        Exemplo_old.status = STARTED
        thisExp.addData('Exemplo_old.started', Exemplo_old.tStart)
        Exemplo_old.maxDuration = None
        # keep track of which components have finished
        Exemplo_oldComponents = Exemplo_old.components
        for thisComponent in Exemplo_old.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Exemplo_old" ---
        # if trial has changed, end Routine now
        if isinstance(trial_ex_old, data.TrialHandler2) and thisTrial_ex_old.thisN != trial_ex_old.thisTrial.thisN:
            continueRoutine = False
        Exemplo_old.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_exemplo_old_new* updates
            
            # if image_exemplo_old_new is starting this frame...
            if image_exemplo_old_new.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                image_exemplo_old_new.frameNStart = frameN  # exact frame index
                image_exemplo_old_new.tStart = t  # local t and not account for scr refresh
                image_exemplo_old_new.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_exemplo_old_new, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_exemplo_old_new.started')
                # update status
                image_exemplo_old_new.status = STARTED
                image_exemplo_old_new.setAutoDraw(True)
            
            # if image_exemplo_old_new is active this frame...
            if image_exemplo_old_new.status == STARTED:
                # update params
                pass
            
            # if image_exemplo_old_new is stopping this frame...
            if image_exemplo_old_new.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_exemplo_old_new.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_exemplo_old_new.tStop = t  # not accounting for scr refresh
                    image_exemplo_old_new.tStopRefresh = tThisFlipGlobal  # on global time
                    image_exemplo_old_new.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_exemplo_old_new.stopped')
                    # update status
                    image_exemplo_old_new.status = FINISHED
                    image_exemplo_old_new.setAutoDraw(False)
            
            # *key_ex_old* updates
            waitOnFlip = False
            
            # if key_ex_old is starting this frame...
            if key_ex_old.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_ex_old.frameNStart = frameN  # exact frame index
                key_ex_old.tStart = t  # local t and not account for scr refresh
                key_ex_old.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_ex_old, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_ex_old.started')
                # update status
                key_ex_old.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_ex_old.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_ex_old.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_ex_old is stopping this frame...
            if key_ex_old.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_ex_old.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    key_ex_old.tStop = t  # not accounting for scr refresh
                    key_ex_old.tStopRefresh = tThisFlipGlobal  # on global time
                    key_ex_old.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_ex_old.stopped')
                    # update status
                    key_ex_old.status = FINISHED
                    key_ex_old.status = FINISHED
            if key_ex_old.status == STARTED and not waitOnFlip:
                theseKeys = key_ex_old.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_ex_old_allKeys.extend(theseKeys)
                if len(_key_ex_old_allKeys):
                    key_ex_old.keys = [key.name for key in _key_ex_old_allKeys]  # storing all keys
                    key_ex_old.rt = [key.rt for key in _key_ex_old_allKeys]
                    key_ex_old.duration = [key.duration for key in _key_ex_old_allKeys]
            
            # *text_ex_old* updates
            
            # if text_ex_old is starting this frame...
            if text_ex_old.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_ex_old.frameNStart = frameN  # exact frame index
                text_ex_old.tStart = t  # local t and not account for scr refresh
                text_ex_old.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_ex_old, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_ex_old.started')
                # update status
                text_ex_old.status = STARTED
                text_ex_old.setAutoDraw(True)
            
            # if text_ex_old is active this frame...
            if text_ex_old.status == STARTED:
                # update params
                pass
            
            # if text_ex_old is stopping this frame...
            if text_ex_old.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_ex_old.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_ex_old.tStop = t  # not accounting for scr refresh
                    text_ex_old.tStopRefresh = tThisFlipGlobal  # on global time
                    text_ex_old.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_ex_old.stopped')
                    # update status
                    text_ex_old.status = FINISHED
                    text_ex_old.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Exemplo_old.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Exemplo_old.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Exemplo_old" ---
        for thisComponent in Exemplo_old.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Exemplo_old
        Exemplo_old.tStop = globalClock.getTime(format='float')
        Exemplo_old.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Exemplo_old.stopped', Exemplo_old.tStop)
        # check responses
        if key_ex_old.keys in ['', [], None]:  # No response was made
            key_ex_old.keys = None
        trial_ex_old.addData('key_ex_old.keys',key_ex_old.keys)
        if key_ex_old.keys != None:  # we had a response
            trial_ex_old.addData('key_ex_old.rt', key_ex_old.rt)
            trial_ex_old.addData('key_ex_old.duration', key_ex_old.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Exemplo_old.maxDurationReached:
            routineTimer.addTime(-Exemplo_old.maxDuration)
        elif Exemplo_old.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trial_ex_old'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Start_Recall" ---
    # create an object to store info about Routine Start_Recall
    Start_Recall = data.Routine(
        name='Start_Recall',
        components=[text_start_recall, key_start_recall],
    )
    Start_Recall.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_start_recall
    key_start_recall.keys = []
    key_start_recall.rt = []
    _key_start_recall_allKeys = []
    # store start times for Start_Recall
    Start_Recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start_Recall.tStart = globalClock.getTime(format='float')
    Start_Recall.status = STARTED
    thisExp.addData('Start_Recall.started', Start_Recall.tStart)
    Start_Recall.maxDuration = None
    # keep track of which components have finished
    Start_RecallComponents = Start_Recall.components
    for thisComponent in Start_Recall.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start_Recall" ---
    Start_Recall.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_start_recall* updates
        
        # if text_start_recall is starting this frame...
        if text_start_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_start_recall.frameNStart = frameN  # exact frame index
            text_start_recall.tStart = t  # local t and not account for scr refresh
            text_start_recall.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start_recall, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_start_recall.started')
            # update status
            text_start_recall.status = STARTED
            text_start_recall.setAutoDraw(True)
        
        # if text_start_recall is active this frame...
        if text_start_recall.status == STARTED:
            # update params
            pass
        
        # *key_start_recall* updates
        waitOnFlip = False
        
        # if key_start_recall is starting this frame...
        if key_start_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_start_recall.frameNStart = frameN  # exact frame index
            key_start_recall.tStart = t  # local t and not account for scr refresh
            key_start_recall.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_start_recall, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_start_recall.started')
            # update status
            key_start_recall.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_start_recall.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_start_recall.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_start_recall.status == STARTED and not waitOnFlip:
            theseKeys = key_start_recall.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_start_recall_allKeys.extend(theseKeys)
            if len(_key_start_recall_allKeys):
                key_start_recall.keys = _key_start_recall_allKeys[-1].name  # just the last key pressed
                key_start_recall.rt = _key_start_recall_allKeys[-1].rt
                key_start_recall.duration = _key_start_recall_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Start_Recall.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Start_Recall.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_Recall" ---
    for thisComponent in Start_Recall.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start_Recall
    Start_Recall.tStop = globalClock.getTime(format='float')
    Start_Recall.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start_Recall.stopped', Start_Recall.tStop)
    # check responses
    if key_start_recall.keys in ['', [], None]:  # No response was made
        key_start_recall.keys = None
    thisExp.addData('key_start_recall.keys',key_start_recall.keys)
    if key_start_recall.keys != None:  # we had a response
        thisExp.addData('key_start_recall.rt', key_start_recall.rt)
        thisExp.addData('key_start_recall.duration', key_start_recall.duration)
    thisExp.nextEntry()
    # the Routine "Start_Recall" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_item = data.TrialHandler2(
        name='trial_item',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('encode_pairs.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trial_item)  # add the loop to the experiment
    thisTrial_item = trial_item.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_item.rgb)
    if thisTrial_item != None:
        for paramName in thisTrial_item:
            globals()[paramName] = thisTrial_item[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_item in trial_item:
        currentLoop = trial_item
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_item.rgb)
        if thisTrial_item != None:
            for paramName in thisTrial_item:
                globals()[paramName] = thisTrial_item[paramName]
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[cross_25, image_blank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(trial_item, data.TrialHandler2) and thisTrial_item.thisN != trial_item.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_25* updates
            
            # if cross_25 is starting this frame...
            if cross_25.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                cross_25.frameNStart = frameN  # exact frame index
                cross_25.tStart = t  # local t and not account for scr refresh
                cross_25.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_25, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_25.started')
                # update status
                cross_25.status = STARTED
                cross_25.setAutoDraw(True)
            
            # if cross_25 is active this frame...
            if cross_25.status == STARTED:
                # update params
                pass
            
            # if cross_25 is stopping this frame...
            if cross_25.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_25.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_25.tStop = t  # not accounting for scr refresh
                    cross_25.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_25.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_25.stopped')
                    # update status
                    cross_25.status = FINISHED
                    cross_25.setAutoDraw(False)
            
            # *image_blank* updates
            
            # if image_blank is starting this frame...
            if image_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_blank.frameNStart = frameN  # exact frame index
                image_blank.tStart = t  # local t and not account for scr refresh
                image_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_blank.started')
                # update status
                image_blank.status = STARTED
                image_blank.setAutoDraw(True)
            
            # if image_blank is active this frame...
            if image_blank.status == STARTED:
                # update params
                pass
            
            # if image_blank is stopping this frame...
            if image_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_blank.tStartRefresh + jitter_ITI-frameTolerance:
                    # keep track of stop time/frame for later
                    image_blank.tStop = t  # not accounting for scr refresh
                    image_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    image_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_blank.stopped')
                    # update status
                    image_blank.status = FINISHED
                    image_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # the Routine "blank" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Old_New_Task" ---
        # create an object to store info about Routine Old_New_Task
        Old_New_Task = data.Routine(
            name='Old_New_Task',
            components=[image_old_new, key_item, text_item],
        )
        Old_New_Task.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_old_new.setSize(scene_size)
        image_old_new.setImage(scene)
        # create starting attributes for key_item
        key_item.keys = []
        key_item.rt = []
        _key_item_allKeys = []
        # store start times for Old_New_Task
        Old_New_Task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Old_New_Task.tStart = globalClock.getTime(format='float')
        Old_New_Task.status = STARTED
        thisExp.addData('Old_New_Task.started', Old_New_Task.tStart)
        Old_New_Task.maxDuration = None
        # keep track of which components have finished
        Old_New_TaskComponents = Old_New_Task.components
        for thisComponent in Old_New_Task.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Old_New_Task" ---
        # if trial has changed, end Routine now
        if isinstance(trial_item, data.TrialHandler2) and thisTrial_item.thisN != trial_item.thisTrial.thisN:
            continueRoutine = False
        Old_New_Task.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_old_new* updates
            
            # if image_old_new is starting this frame...
            if image_old_new.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_old_new.frameNStart = frameN  # exact frame index
                image_old_new.tStart = t  # local t and not account for scr refresh
                image_old_new.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_old_new, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_old_new.started')
                # update status
                image_old_new.status = STARTED
                image_old_new.setAutoDraw(True)
            
            # if image_old_new is active this frame...
            if image_old_new.status == STARTED:
                # update params
                pass
            
            # if image_old_new is stopping this frame...
            if image_old_new.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_old_new.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_old_new.tStop = t  # not accounting for scr refresh
                    image_old_new.tStopRefresh = tThisFlipGlobal  # on global time
                    image_old_new.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_old_new.stopped')
                    # update status
                    image_old_new.status = FINISHED
                    image_old_new.setAutoDraw(False)
            
            # *key_item* updates
            waitOnFlip = False
            
            # if key_item is starting this frame...
            if key_item.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_item.frameNStart = frameN  # exact frame index
                key_item.tStart = t  # local t and not account for scr refresh
                key_item.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_item, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_item.started')
                # update status
                key_item.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_item.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_item.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_item is stopping this frame...
            if key_item.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_item.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    key_item.tStop = t  # not accounting for scr refresh
                    key_item.tStopRefresh = tThisFlipGlobal  # on global time
                    key_item.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_item.stopped')
                    # update status
                    key_item.status = FINISHED
                    key_item.status = FINISHED
            if key_item.status == STARTED and not waitOnFlip:
                theseKeys = key_item.getKeys(keyList=['left','right'], ignoreKeys=["escape"], waitRelease=False)
                _key_item_allKeys.extend(theseKeys)
                if len(_key_item_allKeys):
                    key_item.keys = [key.name for key in _key_item_allKeys]  # storing all keys
                    key_item.rt = [key.rt for key in _key_item_allKeys]
                    key_item.duration = [key.duration for key in _key_item_allKeys]
                    # was this correct?
                    if (key_item.keys == str(correct_key_old)) or (key_item.keys == correct_key_old):
                        key_item.corr = 1
                    else:
                        key_item.corr = 0
            
            # *text_item* updates
            
            # if text_item is starting this frame...
            if text_item.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_item.frameNStart = frameN  # exact frame index
                text_item.tStart = t  # local t and not account for scr refresh
                text_item.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_item, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_item.started')
                # update status
                text_item.status = STARTED
                text_item.setAutoDraw(True)
            
            # if text_item is active this frame...
            if text_item.status == STARTED:
                # update params
                pass
            
            # if text_item is stopping this frame...
            if text_item.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_item.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    text_item.tStop = t  # not accounting for scr refresh
                    text_item.tStopRefresh = tThisFlipGlobal  # on global time
                    text_item.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_item.stopped')
                    # update status
                    text_item.status = FINISHED
                    text_item.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Old_New_Task.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Old_New_Task.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Old_New_Task" ---
        for thisComponent in Old_New_Task.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Old_New_Task
        Old_New_Task.tStop = globalClock.getTime(format='float')
        Old_New_Task.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Old_New_Task.stopped', Old_New_Task.tStop)
        # check responses
        if key_item.keys in ['', [], None]:  # No response was made
            key_item.keys = None
            # was no response the correct answer?!
            if str(correct_key_old).lower() == 'none':
               key_item.corr = 1;  # correct non-response
            else:
               key_item.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_item (TrialHandler)
        trial_item.addData('key_item.keys',key_item.keys)
        trial_item.addData('key_item.corr', key_item.corr)
        if key_item.keys != None:  # we had a response
            trial_item.addData('key_item.rt', key_item.rt)
            trial_item.addData('key_item.duration', key_item.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Old_New_Task.maxDurationReached:
            routineTimer.addTime(-Old_New_Task.maxDuration)
        elif Old_New_Task.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trial_item'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "End_Old_New" ---
    # create an object to store info about Routine End_Old_New
    End_Old_New = data.Routine(
        name='End_Old_New',
        components=[text_end_item, key_resp_2],
    )
    End_Old_New.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for End_Old_New
    End_Old_New.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    End_Old_New.tStart = globalClock.getTime(format='float')
    End_Old_New.status = STARTED
    thisExp.addData('End_Old_New.started', End_Old_New.tStart)
    End_Old_New.maxDuration = None
    # keep track of which components have finished
    End_Old_NewComponents = End_Old_New.components
    for thisComponent in End_Old_New.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "End_Old_New" ---
    End_Old_New.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end_item* updates
        
        # if text_end_item is starting this frame...
        if text_end_item.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end_item.frameNStart = frameN  # exact frame index
            text_end_item.tStart = t  # local t and not account for scr refresh
            text_end_item.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end_item, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end_item.started')
            # update status
            text_end_item.status = STARTED
            text_end_item.setAutoDraw(True)
        
        # if text_end_item is active this frame...
        if text_end_item.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            End_Old_New.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in End_Old_New.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "End_Old_New" ---
    for thisComponent in End_Old_New.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for End_Old_New
    End_Old_New.tStop = globalClock.getTime(format='float')
    End_Old_New.tStopRefresh = tThisFlipGlobal
    thisExp.addData('End_Old_New.stopped', End_Old_New.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "End_Old_New" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "Explain_2AFC" ---
    # create an object to store info about Routine Explain_2AFC
    Explain_2AFC = data.Routine(
        name='Explain_2AFC',
        components=[text_explain_2afc, key_end_explain_2afc],
    )
    Explain_2AFC.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_end_explain_2afc
    key_end_explain_2afc.keys = []
    key_end_explain_2afc.rt = []
    _key_end_explain_2afc_allKeys = []
    # store start times for Explain_2AFC
    Explain_2AFC.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Explain_2AFC.tStart = globalClock.getTime(format='float')
    Explain_2AFC.status = STARTED
    thisExp.addData('Explain_2AFC.started', Explain_2AFC.tStart)
    Explain_2AFC.maxDuration = None
    # keep track of which components have finished
    Explain_2AFCComponents = Explain_2AFC.components
    for thisComponent in Explain_2AFC.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Explain_2AFC" ---
    Explain_2AFC.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_explain_2afc* updates
        
        # if text_explain_2afc is starting this frame...
        if text_explain_2afc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_explain_2afc.frameNStart = frameN  # exact frame index
            text_explain_2afc.tStart = t  # local t and not account for scr refresh
            text_explain_2afc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_explain_2afc, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_explain_2afc.started')
            # update status
            text_explain_2afc.status = STARTED
            text_explain_2afc.setAutoDraw(True)
        
        # if text_explain_2afc is active this frame...
        if text_explain_2afc.status == STARTED:
            # update params
            pass
        
        # *key_end_explain_2afc* updates
        waitOnFlip = False
        
        # if key_end_explain_2afc is starting this frame...
        if key_end_explain_2afc.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_end_explain_2afc.frameNStart = frameN  # exact frame index
            key_end_explain_2afc.tStart = t  # local t and not account for scr refresh
            key_end_explain_2afc.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_end_explain_2afc, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_end_explain_2afc.started')
            # update status
            key_end_explain_2afc.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_end_explain_2afc.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_end_explain_2afc.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_end_explain_2afc.status == STARTED and not waitOnFlip:
            theseKeys = key_end_explain_2afc.getKeys(keyList=['y','n','left','right','space'], ignoreKeys=["escape"], waitRelease=False)
            _key_end_explain_2afc_allKeys.extend(theseKeys)
            if len(_key_end_explain_2afc_allKeys):
                key_end_explain_2afc.keys = _key_end_explain_2afc_allKeys[-1].name  # just the last key pressed
                key_end_explain_2afc.rt = _key_end_explain_2afc_allKeys[-1].rt
                key_end_explain_2afc.duration = _key_end_explain_2afc_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Explain_2AFC.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Explain_2AFC.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Explain_2AFC" ---
    for thisComponent in Explain_2AFC.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Explain_2AFC
    Explain_2AFC.tStop = globalClock.getTime(format='float')
    Explain_2AFC.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Explain_2AFC.stopped', Explain_2AFC.tStop)
    # check responses
    if key_end_explain_2afc.keys in ['', [], None]:  # No response was made
        key_end_explain_2afc.keys = None
    thisExp.addData('key_end_explain_2afc.keys',key_end_explain_2afc.keys)
    if key_end_explain_2afc.keys != None:  # we had a response
        thisExp.addData('key_end_explain_2afc.rt', key_end_explain_2afc.rt)
        thisExp.addData('key_end_explain_2afc.duration', key_end_explain_2afc.duration)
    thisExp.nextEntry()
    # the Routine "Explain_2AFC" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_ex_2AFC = data.TrialHandler2(
        name='trial_ex_2AFC',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('Exemplo_encode.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(trial_ex_2AFC)  # add the loop to the experiment
    thisTrial_ex_2AFC = trial_ex_2AFC.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_ex_2AFC.rgb)
    if thisTrial_ex_2AFC != None:
        for paramName in thisTrial_ex_2AFC:
            globals()[paramName] = thisTrial_ex_2AFC[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_ex_2AFC in trial_ex_2AFC:
        currentLoop = trial_ex_2AFC
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_ex_2AFC.rgb)
        if thisTrial_ex_2AFC != None:
            for paramName in thisTrial_ex_2AFC:
                globals()[paramName] = thisTrial_ex_2AFC[paramName]
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[cross_25, image_blank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(trial_ex_2AFC, data.TrialHandler2) and thisTrial_ex_2AFC.thisN != trial_ex_2AFC.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_25* updates
            
            # if cross_25 is starting this frame...
            if cross_25.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                cross_25.frameNStart = frameN  # exact frame index
                cross_25.tStart = t  # local t and not account for scr refresh
                cross_25.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_25, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_25.started')
                # update status
                cross_25.status = STARTED
                cross_25.setAutoDraw(True)
            
            # if cross_25 is active this frame...
            if cross_25.status == STARTED:
                # update params
                pass
            
            # if cross_25 is stopping this frame...
            if cross_25.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_25.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_25.tStop = t  # not accounting for scr refresh
                    cross_25.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_25.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_25.stopped')
                    # update status
                    cross_25.status = FINISHED
                    cross_25.setAutoDraw(False)
            
            # *image_blank* updates
            
            # if image_blank is starting this frame...
            if image_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_blank.frameNStart = frameN  # exact frame index
                image_blank.tStart = t  # local t and not account for scr refresh
                image_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_blank.started')
                # update status
                image_blank.status = STARTED
                image_blank.setAutoDraw(True)
            
            # if image_blank is active this frame...
            if image_blank.status == STARTED:
                # update params
                pass
            
            # if image_blank is stopping this frame...
            if image_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_blank.tStartRefresh + jitter_ITI-frameTolerance:
                    # keep track of stop time/frame for later
                    image_blank.tStop = t  # not accounting for scr refresh
                    image_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    image_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_blank.stopped')
                    # update status
                    image_blank.status = FINISHED
                    image_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # the Routine "blank" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "Exemplo_2AFC" ---
        # create an object to store info about Routine Exemplo_2AFC
        Exemplo_2AFC = data.Routine(
            name='Exemplo_2AFC',
            components=[image_ex_2AFC, image_ex_target, image_ex_lure, key_ex_2AFC],
        )
        Exemplo_2AFC.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_ex_2AFC.setSize(scene_size)
        image_ex_2AFC.setImage(scene_encode)
        image_ex_target.setPos(target_pos)
        image_ex_target.setSize(object_size)
        image_ex_target.setImage(object_encode)
        image_ex_lure.setPos(lure_pos)
        image_ex_lure.setSize(lure_size)
        image_ex_lure.setImage(lure)
        # create starting attributes for key_ex_2AFC
        key_ex_2AFC.keys = []
        key_ex_2AFC.rt = []
        _key_ex_2AFC_allKeys = []
        # store start times for Exemplo_2AFC
        Exemplo_2AFC.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        Exemplo_2AFC.tStart = globalClock.getTime(format='float')
        Exemplo_2AFC.status = STARTED
        thisExp.addData('Exemplo_2AFC.started', Exemplo_2AFC.tStart)
        Exemplo_2AFC.maxDuration = None
        # keep track of which components have finished
        Exemplo_2AFCComponents = Exemplo_2AFC.components
        for thisComponent in Exemplo_2AFC.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "Exemplo_2AFC" ---
        # if trial has changed, end Routine now
        if isinstance(trial_ex_2AFC, data.TrialHandler2) and thisTrial_ex_2AFC.thisN != trial_ex_2AFC.thisTrial.thisN:
            continueRoutine = False
        Exemplo_2AFC.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_ex_2AFC* updates
            
            # if image_ex_2AFC is starting this frame...
            if image_ex_2AFC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_ex_2AFC.frameNStart = frameN  # exact frame index
                image_ex_2AFC.tStart = t  # local t and not account for scr refresh
                image_ex_2AFC.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_ex_2AFC, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_ex_2AFC.started')
                # update status
                image_ex_2AFC.status = STARTED
                image_ex_2AFC.setAutoDraw(True)
            
            # if image_ex_2AFC is active this frame...
            if image_ex_2AFC.status == STARTED:
                # update params
                pass
            
            # if image_ex_2AFC is stopping this frame...
            if image_ex_2AFC.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_ex_2AFC.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_ex_2AFC.tStop = t  # not accounting for scr refresh
                    image_ex_2AFC.tStopRefresh = tThisFlipGlobal  # on global time
                    image_ex_2AFC.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_ex_2AFC.stopped')
                    # update status
                    image_ex_2AFC.status = FINISHED
                    image_ex_2AFC.setAutoDraw(False)
            
            # *image_ex_target* updates
            
            # if image_ex_target is starting this frame...
            if image_ex_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_ex_target.frameNStart = frameN  # exact frame index
                image_ex_target.tStart = t  # local t and not account for scr refresh
                image_ex_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_ex_target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_ex_target.started')
                # update status
                image_ex_target.status = STARTED
                image_ex_target.setAutoDraw(True)
            
            # if image_ex_target is active this frame...
            if image_ex_target.status == STARTED:
                # update params
                pass
            
            # if image_ex_target is stopping this frame...
            if image_ex_target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_ex_target.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_ex_target.tStop = t  # not accounting for scr refresh
                    image_ex_target.tStopRefresh = tThisFlipGlobal  # on global time
                    image_ex_target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_ex_target.stopped')
                    # update status
                    image_ex_target.status = FINISHED
                    image_ex_target.setAutoDraw(False)
            
            # *image_ex_lure* updates
            
            # if image_ex_lure is starting this frame...
            if image_ex_lure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_ex_lure.frameNStart = frameN  # exact frame index
                image_ex_lure.tStart = t  # local t and not account for scr refresh
                image_ex_lure.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_ex_lure, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_ex_lure.started')
                # update status
                image_ex_lure.status = STARTED
                image_ex_lure.setAutoDraw(True)
            
            # if image_ex_lure is active this frame...
            if image_ex_lure.status == STARTED:
                # update params
                pass
            
            # if image_ex_lure is stopping this frame...
            if image_ex_lure.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_ex_lure.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_ex_lure.tStop = t  # not accounting for scr refresh
                    image_ex_lure.tStopRefresh = tThisFlipGlobal  # on global time
                    image_ex_lure.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_ex_lure.stopped')
                    # update status
                    image_ex_lure.status = FINISHED
                    image_ex_lure.setAutoDraw(False)
            
            # *key_ex_2AFC* updates
            waitOnFlip = False
            
            # if key_ex_2AFC is starting this frame...
            if key_ex_2AFC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_ex_2AFC.frameNStart = frameN  # exact frame index
                key_ex_2AFC.tStart = t  # local t and not account for scr refresh
                key_ex_2AFC.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_ex_2AFC, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_ex_2AFC.started')
                # update status
                key_ex_2AFC.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_ex_2AFC.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_ex_2AFC.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_ex_2AFC is stopping this frame...
            if key_ex_2AFC.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_ex_2AFC.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    key_ex_2AFC.tStop = t  # not accounting for scr refresh
                    key_ex_2AFC.tStopRefresh = tThisFlipGlobal  # on global time
                    key_ex_2AFC.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_ex_2AFC.stopped')
                    # update status
                    key_ex_2AFC.status = FINISHED
                    key_ex_2AFC.status = FINISHED
            if key_ex_2AFC.status == STARTED and not waitOnFlip:
                theseKeys = key_ex_2AFC.getKeys(keyList=['up','down'], ignoreKeys=["escape"], waitRelease=False)
                _key_ex_2AFC_allKeys.extend(theseKeys)
                if len(_key_ex_2AFC_allKeys):
                    key_ex_2AFC.keys = [key.name for key in _key_ex_2AFC_allKeys]  # storing all keys
                    key_ex_2AFC.rt = [key.rt for key in _key_ex_2AFC_allKeys]
                    key_ex_2AFC.duration = [key.duration for key in _key_ex_2AFC_allKeys]
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                Exemplo_2AFC.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in Exemplo_2AFC.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "Exemplo_2AFC" ---
        for thisComponent in Exemplo_2AFC.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for Exemplo_2AFC
        Exemplo_2AFC.tStop = globalClock.getTime(format='float')
        Exemplo_2AFC.tStopRefresh = tThisFlipGlobal
        thisExp.addData('Exemplo_2AFC.stopped', Exemplo_2AFC.tStop)
        # check responses
        if key_ex_2AFC.keys in ['', [], None]:  # No response was made
            key_ex_2AFC.keys = None
        trial_ex_2AFC.addData('key_ex_2AFC.keys',key_ex_2AFC.keys)
        if key_ex_2AFC.keys != None:  # we had a response
            trial_ex_2AFC.addData('key_ex_2AFC.rt', key_ex_2AFC.rt)
            trial_ex_2AFC.addData('key_ex_2AFC.duration', key_ex_2AFC.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if Exemplo_2AFC.maxDurationReached:
            routineTimer.addTime(-Exemplo_2AFC.maxDuration)
        elif Exemplo_2AFC.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trial_ex_2AFC'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "Start_Recall" ---
    # create an object to store info about Routine Start_Recall
    Start_Recall = data.Routine(
        name='Start_Recall',
        components=[text_start_recall, key_start_recall],
    )
    Start_Recall.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_start_recall
    key_start_recall.keys = []
    key_start_recall.rt = []
    _key_start_recall_allKeys = []
    # store start times for Start_Recall
    Start_Recall.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    Start_Recall.tStart = globalClock.getTime(format='float')
    Start_Recall.status = STARTED
    thisExp.addData('Start_Recall.started', Start_Recall.tStart)
    Start_Recall.maxDuration = None
    # keep track of which components have finished
    Start_RecallComponents = Start_Recall.components
    for thisComponent in Start_Recall.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "Start_Recall" ---
    Start_Recall.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_start_recall* updates
        
        # if text_start_recall is starting this frame...
        if text_start_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_start_recall.frameNStart = frameN  # exact frame index
            text_start_recall.tStart = t  # local t and not account for scr refresh
            text_start_recall.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_start_recall, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_start_recall.started')
            # update status
            text_start_recall.status = STARTED
            text_start_recall.setAutoDraw(True)
        
        # if text_start_recall is active this frame...
        if text_start_recall.status == STARTED:
            # update params
            pass
        
        # *key_start_recall* updates
        waitOnFlip = False
        
        # if key_start_recall is starting this frame...
        if key_start_recall.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_start_recall.frameNStart = frameN  # exact frame index
            key_start_recall.tStart = t  # local t and not account for scr refresh
            key_start_recall.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_start_recall, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_start_recall.started')
            # update status
            key_start_recall.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_start_recall.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_start_recall.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_start_recall.status == STARTED and not waitOnFlip:
            theseKeys = key_start_recall.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_start_recall_allKeys.extend(theseKeys)
            if len(_key_start_recall_allKeys):
                key_start_recall.keys = _key_start_recall_allKeys[-1].name  # just the last key pressed
                key_start_recall.rt = _key_start_recall_allKeys[-1].rt
                key_start_recall.duration = _key_start_recall_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            Start_Recall.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in Start_Recall.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "Start_Recall" ---
    for thisComponent in Start_Recall.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for Start_Recall
    Start_Recall.tStop = globalClock.getTime(format='float')
    Start_Recall.tStopRefresh = tThisFlipGlobal
    thisExp.addData('Start_Recall.stopped', Start_Recall.tStop)
    # check responses
    if key_start_recall.keys in ['', [], None]:  # No response was made
        key_start_recall.keys = None
    thisExp.addData('key_start_recall.keys',key_start_recall.keys)
    if key_start_recall.keys != None:  # we had a response
        thisExp.addData('key_start_recall.rt', key_start_recall.rt)
        thisExp.addData('key_start_recall.duration', key_start_recall.duration)
    thisExp.nextEntry()
    # the Routine "Start_Recall" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    trial_asso = data.TrialHandler2(
        name='trial_asso',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(
        'encode_pairs.xlsx', 
        selection='0:60'
    )
    , 
        seed=None, 
    )
    thisExp.addLoop(trial_asso)  # add the loop to the experiment
    thisTrial_asso = trial_asso.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial_asso.rgb)
    if thisTrial_asso != None:
        for paramName in thisTrial_asso:
            globals()[paramName] = thisTrial_asso[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrial_asso in trial_asso:
        currentLoop = trial_asso
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrial_asso.rgb)
        if thisTrial_asso != None:
            for paramName in thisTrial_asso:
                globals()[paramName] = thisTrial_asso[paramName]
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[cross_25, image_blank],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(trial_asso, data.TrialHandler2) and thisTrial_asso.thisN != trial_asso.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cross_25* updates
            
            # if cross_25 is starting this frame...
            if cross_25.status == NOT_STARTED and tThisFlip >= 2.25-frameTolerance:
                # keep track of start time/frame for later
                cross_25.frameNStart = frameN  # exact frame index
                cross_25.tStart = t  # local t and not account for scr refresh
                cross_25.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cross_25, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cross_25.started')
                # update status
                cross_25.status = STARTED
                cross_25.setAutoDraw(True)
            
            # if cross_25 is active this frame...
            if cross_25.status == STARTED:
                # update params
                pass
            
            # if cross_25 is stopping this frame...
            if cross_25.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cross_25.tStartRefresh + 0.25-frameTolerance:
                    # keep track of stop time/frame for later
                    cross_25.tStop = t  # not accounting for scr refresh
                    cross_25.tStopRefresh = tThisFlipGlobal  # on global time
                    cross_25.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cross_25.stopped')
                    # update status
                    cross_25.status = FINISHED
                    cross_25.setAutoDraw(False)
            
            # *image_blank* updates
            
            # if image_blank is starting this frame...
            if image_blank.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_blank.frameNStart = frameN  # exact frame index
                image_blank.tStart = t  # local t and not account for scr refresh
                image_blank.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_blank, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_blank.started')
                # update status
                image_blank.status = STARTED
                image_blank.setAutoDraw(True)
            
            # if image_blank is active this frame...
            if image_blank.status == STARTED:
                # update params
                pass
            
            # if image_blank is stopping this frame...
            if image_blank.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_blank.tStartRefresh + jitter_ITI-frameTolerance:
                    # keep track of stop time/frame for later
                    image_blank.tStop = t  # not accounting for scr refresh
                    image_blank.tStopRefresh = tThisFlipGlobal  # on global time
                    image_blank.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_blank.stopped')
                    # update status
                    image_blank.status = FINISHED
                    image_blank.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # the Routine "blank" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trial_2AFC" ---
        # create an object to store info about Routine trial_2AFC
        trial_2AFC = data.Routine(
            name='trial_2AFC',
            components=[image_scene_2AFC, key_2AFC, image_target, image_lure],
        )
        trial_2AFC.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        image_scene_2AFC.setSize(scene_size)
        image_scene_2AFC.setImage(scene)
        # create starting attributes for key_2AFC
        key_2AFC.keys = []
        key_2AFC.rt = []
        _key_2AFC_allKeys = []
        image_target.setPos(target_pos)
        image_target.setSize(object_size)
        image_target.setImage(object_encode)
        image_lure.setPos(lure_pos)
        image_lure.setSize(lure_size)
        image_lure.setImage(lure)
        # Run 'Begin Routine' code from code_debug_2AFC
        print(f"trial {trial_asso.thisN}, image = {image_scene_2AFC}")
        print(f"trial {trial_asso.thisN}, image = {image_target}")
        print(f"trial {trial_asso.thisN}, image = {image_lure}")
        # store start times for trial_2AFC
        trial_2AFC.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial_2AFC.tStart = globalClock.getTime(format='float')
        trial_2AFC.status = STARTED
        thisExp.addData('trial_2AFC.started', trial_2AFC.tStart)
        trial_2AFC.maxDuration = None
        # keep track of which components have finished
        trial_2AFCComponents = trial_2AFC.components
        for thisComponent in trial_2AFC.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trial_2AFC" ---
        # if trial has changed, end Routine now
        if isinstance(trial_asso, data.TrialHandler2) and thisTrial_asso.thisN != trial_asso.thisTrial.thisN:
            continueRoutine = False
        trial_2AFC.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 3.0:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_scene_2AFC* updates
            
            # if image_scene_2AFC is starting this frame...
            if image_scene_2AFC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_scene_2AFC.frameNStart = frameN  # exact frame index
                image_scene_2AFC.tStart = t  # local t and not account for scr refresh
                image_scene_2AFC.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_scene_2AFC, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_scene_2AFC.started')
                # update status
                image_scene_2AFC.status = STARTED
                image_scene_2AFC.setAutoDraw(True)
            
            # if image_scene_2AFC is active this frame...
            if image_scene_2AFC.status == STARTED:
                # update params
                pass
            
            # if image_scene_2AFC is stopping this frame...
            if image_scene_2AFC.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_scene_2AFC.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_scene_2AFC.tStop = t  # not accounting for scr refresh
                    image_scene_2AFC.tStopRefresh = tThisFlipGlobal  # on global time
                    image_scene_2AFC.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_scene_2AFC.stopped')
                    # update status
                    image_scene_2AFC.status = FINISHED
                    image_scene_2AFC.setAutoDraw(False)
            
            # *key_2AFC* updates
            waitOnFlip = False
            
            # if key_2AFC is starting this frame...
            if key_2AFC.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_2AFC.frameNStart = frameN  # exact frame index
                key_2AFC.tStart = t  # local t and not account for scr refresh
                key_2AFC.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_2AFC, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_2AFC.started')
                # update status
                key_2AFC.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_2AFC.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_2AFC.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_2AFC is stopping this frame...
            if key_2AFC.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_2AFC.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    key_2AFC.tStop = t  # not accounting for scr refresh
                    key_2AFC.tStopRefresh = tThisFlipGlobal  # on global time
                    key_2AFC.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_2AFC.stopped')
                    # update status
                    key_2AFC.status = FINISHED
                    key_2AFC.status = FINISHED
            if key_2AFC.status == STARTED and not waitOnFlip:
                theseKeys = key_2AFC.getKeys(keyList=['up','down'], ignoreKeys=["escape"], waitRelease=False)
                _key_2AFC_allKeys.extend(theseKeys)
                if len(_key_2AFC_allKeys):
                    key_2AFC.keys = [key.name for key in _key_2AFC_allKeys]  # storing all keys
                    key_2AFC.rt = [key.rt for key in _key_2AFC_allKeys]
                    key_2AFC.duration = [key.duration for key in _key_2AFC_allKeys]
                    # was this correct?
                    if (key_2AFC.keys == str(correct_key_AFC)) or (key_2AFC.keys == correct_key_AFC):
                        key_2AFC.corr = 1
                    else:
                        key_2AFC.corr = 0
            
            # *image_target* updates
            
            # if image_target is starting this frame...
            if image_target.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_target.frameNStart = frameN  # exact frame index
                image_target.tStart = t  # local t and not account for scr refresh
                image_target.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_target, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_target.started')
                # update status
                image_target.status = STARTED
                image_target.setAutoDraw(True)
            
            # if image_target is active this frame...
            if image_target.status == STARTED:
                # update params
                pass
            
            # if image_target is stopping this frame...
            if image_target.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_target.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_target.tStop = t  # not accounting for scr refresh
                    image_target.tStopRefresh = tThisFlipGlobal  # on global time
                    image_target.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_target.stopped')
                    # update status
                    image_target.status = FINISHED
                    image_target.setAutoDraw(False)
            
            # *image_lure* updates
            
            # if image_lure is starting this frame...
            if image_lure.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_lure.frameNStart = frameN  # exact frame index
                image_lure.tStart = t  # local t and not account for scr refresh
                image_lure.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_lure, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_lure.started')
                # update status
                image_lure.status = STARTED
                image_lure.setAutoDraw(True)
            
            # if image_lure is active this frame...
            if image_lure.status == STARTED:
                # update params
                pass
            
            # if image_lure is stopping this frame...
            if image_lure.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_lure.tStartRefresh + 3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_lure.tStop = t  # not accounting for scr refresh
                    image_lure.tStopRefresh = tThisFlipGlobal  # on global time
                    image_lure.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_lure.stopped')
                    # update status
                    image_lure.status = FINISHED
                    image_lure.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trial_2AFC.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial_2AFC.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial_2AFC" ---
        for thisComponent in trial_2AFC.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial_2AFC
        trial_2AFC.tStop = globalClock.getTime(format='float')
        trial_2AFC.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial_2AFC.stopped', trial_2AFC.tStop)
        # check responses
        if key_2AFC.keys in ['', [], None]:  # No response was made
            key_2AFC.keys = None
            # was no response the correct answer?!
            if str(correct_key_AFC).lower() == 'none':
               key_2AFC.corr = 1;  # correct non-response
            else:
               key_2AFC.corr = 0;  # failed to respond (incorrectly)
        # store data for trial_asso (TrialHandler)
        trial_asso.addData('key_2AFC.keys',key_2AFC.keys)
        trial_asso.addData('key_2AFC.corr', key_2AFC.corr)
        if key_2AFC.keys != None:  # we had a response
            trial_asso.addData('key_2AFC.rt', key_2AFC.rt)
            trial_asso.addData('key_2AFC.duration', key_2AFC.duration)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if trial_2AFC.maxDurationReached:
            routineTimer.addTime(-trial_2AFC.maxDuration)
        elif trial_2AFC.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-3.000000)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trial_asso'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[text_end],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # if text_end is stopping this frame...
        if text_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end.tStartRefresh + 10.0-frameTolerance:
                # keep track of stop time/frame for later
                text_end.tStop = t  # not accounting for scr refresh
                text_end.tStopRefresh = tThisFlipGlobal  # on global time
                text_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_end.stopped')
                # update status
                text_end.status = FINISHED
                text_end.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if end.maxDurationReached:
        routineTimer.addTime(-end.maxDuration)
    elif end.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
