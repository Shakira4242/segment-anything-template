FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git wget libgl1-mesa-glx libglib2.0-0

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# We add the banana boilerplate here
ADD server.py .

#Download checkpoint
RUN wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Add your custom app code, init() and inference()
ADD app.py .

# Add handler.py
ADD handler.py .

EXPOSE 8000

CMD python3 -u handler.py