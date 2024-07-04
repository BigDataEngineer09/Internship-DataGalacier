# Use the Fedora base image for reducing the docker image size
FROM fedora:latest

# Install Python and pip
RUN yum -y install python3 && \
    yum -y install python3-pip && \
    yum -y clean all

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed dependencies specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the port Flask runs on
EXPOSE 5000

#set enivronment variable 
ENV NAME OpentoAll

# Run the app model
CMD ["python3", "app.py"]
