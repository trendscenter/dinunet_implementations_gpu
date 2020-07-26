FROM coinstacteam/coinstac-base:cuda-9.2

# Copy the current directory contents into the container
COPY ./requirements.txt /computation/requirements.txt

# Set the working directory
WORKDIR /computation

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . /computation
