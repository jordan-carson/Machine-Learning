# Use an official Python runtime as a parent image
FROM python:3.6

# set the working directory
WORKDIR /Users/jordancarson/PycharmProjects/AmazonKaggle-MLCapstone

# copy the current directory contents into the container
ADD . /Users/jordancarson/PycharmProjects/AmazonKaggle-MLCapstone

# Install any needed packaged specificied in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# make port 80 available to the world outstide this container
EXPOSE 80

# define environment varaibles
ENV NAME World

# RUN app.py when the container launches
CMD ["python", "local_runner.py"]