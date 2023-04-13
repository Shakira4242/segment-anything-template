import runpod
import os
import time
import app as user_src

## load your model(s) into vram here
user_src.init()

def handler(event):
    return event.input

runpod.serverless.start({
    "handler": handler
})