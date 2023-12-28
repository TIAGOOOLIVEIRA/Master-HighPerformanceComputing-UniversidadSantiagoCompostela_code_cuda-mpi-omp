# Use the "hpcnube-base-image" as the base image
FROM hpcnube-base-image

# Install the required packages (OpenJDK 8, vim, openssh-server)
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk vim openssh-server

# Add a user named "luser" with UID 2000
RUN useradd -u 2000 -m luser

# Set the password for the "luser" user (change it as needed)
RUN echo 'luser:HPC2024!' | chpasswd

# Create the .bashrc file for the "luser" with Java and Hadoop environment variables
RUN echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> /home/luser/.bashrc
RUN echo 'export HADOOP_HOME=/path/to/hadoop' >> /home/luser/.bashrc

# Add Hadoop binaries to the PATH
RUN echo 'export PATH=${HADOOP_HOME}/bin:$PATH' >> /home/luser/.bashrc

# Expose SSH port
EXPOSE 22

# Start the SSH server
CMD ["/usr/sbin/sshd", "-D"]

