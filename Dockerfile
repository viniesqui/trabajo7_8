# Use an official Python runtime as a parent image
FROM python:3.12-rc-slim-buster

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# puerto que se va a expone por el api 
EXPOSE 8000

#Correr main
CMD ["python", "main.py"]