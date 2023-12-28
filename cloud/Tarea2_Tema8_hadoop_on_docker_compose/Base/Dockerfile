FROM ubuntu:latest
MAINTAINER Tomas <tf.pena@usc.es>

# Switch to root user
USER root

ENV HADOOP_VERSION 3.3.6
ENV BASE_DIR /opt/bd
ENV LOG_TAG "[BASE Hadoop_${HADOOP_VERSION}]:"
ENV REPOSITORY https://dlcdn.apache.org/hadoop/common

# STEP 1: Instala el software necesario
RUN echo "$LOG_TAG Actualizando e instalando paquetes básicos" && \
    apt-get update && \
    apt-get install -y --no-install-recommends openjdk-8-jre python3 curl locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists

# Genera locales
RUN echo "$LOG_TAG Creando locales" && \
    locale-gen es_ES.UTF-8 && update-locale LANG=es_ES.UTF-8

# Crea un directorio para la instalación en /opt: 
RUN mkdir -p ${BASE_DIR}
    
# Change to /opt/bd directory
WORKDIR ${BASE_DIR}

# Descarga la última versión 3.1.1 Hadoop en /opt/bd
RUN echo "$LOG_TAG Descargando Hadoop" && \
    curl -fLO "${REPOSITORY}/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz"  && \
    tar xzvf hadoop-${HADOOP_VERSION}.tar.gz && \
    ln -s hadoop-${HADOOP_VERSION} hadoop && \
    rm hadoop-${HADOOP_VERSION}.tar.gz


# Crea un grupo hadoop y un usuario hdadmin para ejecutar los diferentes demonios de Hadoop (HDFS y YARN). Cambia el propietario del directorio de hadoop 
RUN groupadd -r hadoop && \
    useradd -r -g hadoop -d ${BASE_DIR} -s /bin/bash hdadmin

# Crea directorio para los ficheros de log
RUN mkdir ${HADOOP_HOME}/logs

# Establece permisos 
RUN chown -R hdadmin:hadoop ${BASE_DIR}


