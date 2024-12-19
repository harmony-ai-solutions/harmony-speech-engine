# Docker Setup

Harmony Speech Engine can be easily set up and run using Docker. This allows you to run the application without worrying about installing dependencies manually.

## Prerequisites

- **Docker**: Make sure Docker is installed on your system. You can download it from [Docker's official website](https://www.docker.com/get-started).
- **Docker Compose**: Ensure Docker Compose is installed. It usually comes with Docker Desktop installations.

For GPU support:

- **NVIDIA GPU Drivers**: Required if you plan to use GPU acceleration.
- **NVIDIA Container Toolkit**: Install from [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Docker Compose Files

There are two `docker-compose` files provided:

- **`docker-compose.yml`**: Uses CPU only. Suitable if you don't have an NVIDIA GPU.
- **`docker-compose.nvidia.yml`**: Utilizes NVIDIA GPUs for acceleration.

## Running with Docker Compose (CPU Only)

1. Navigate to the project directory:
   ```bash
   cd harmony-speech-engine
   ```
   
2. Start the services:
   ```bash
   docker compose up
   ```
This will start both the Harmony Speech Engine API and the frontend UI, using the CPU specific `compose-file` by default.

## Running with Docker Compose (NVIDIA GPU Support)

1. Ensure your system meets the GPU prerequisites mentioned above.
2. Navigate to the project directory:
   ```bash
   cd harmony-speech-engine
   ```
3. Start the services with GPU support by specifying the `docker-compose` file created for using nvidia:
   ```bash
   docker compose -f docker-compose.nvidia.yml up
   ```
By default, out images provided via docker hub will be downloaded. However you can also build
the docker images yourself locally, by removing the hashtag '#' in front of the build instruction inside the compose files.

If you want to start the docker containers as services in the background, you can use the `-d` flag:
   ```bash
   docker compose -f docker-compose.yml up -d
   ```
   ```bash
   docker compose -f docker-compose.nvidia.yml up -d
   ```

## Stopping the Services
To stop the running containers:
   ```bash
   docker compose down
   ```
or for the NVIDIA GPU setup:
   ```bash
   docker-compose -f docker-compose.nvidia.yml down
   ```

## Configuration
- **Model Configuration**: The models and their settings are defined in `config.yml` for CPU usage and `config.nvidia.yml` for GPU usage.
- **Volumes**: The Docker setup mounts the `config.yml` or `config.nvidia.yml` and the `cache` directory into the container. Ensure these files and directories exist in your project root

## Accessing the Application
- **API**: Once the services are up, the Harmony Speech Engine API is accessible at http://localhost:12080.
- **Frontend UI**: Access the UI at http://localhost:8080.

## Notes
- Ensure that the ports 12080 and 8080 are available on your host machine.
- If you need to customize the configurations, edit the config.yml or config.nvidia.yml files accordingly.