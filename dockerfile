FROM ghcr.io/gmu-geosciences/repast4py-container:latest
# Taken from: https://github.com/Repast/repast4py/blob/master/Dockerfile
 
# Install the python requirements
COPY ./requirements.txt ./requirements.txt
RUN pip install --break-system-packages -r ./requirements.txt
  
# Set the PYTHONPATH to include the /repast4py folder which contains the core folder
ENV PYTHONPATH=/repast4py/src
