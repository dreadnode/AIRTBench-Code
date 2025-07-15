from pathlib import Path

import docker
import rich
from loguru import logger


def build_container(
    image: str,
    docker_file: str | Path,
    build_path: str | Path = Path(),
    *,
    force_rebuild: bool = False,
) -> str:
    docker_client = docker.DockerClient()

    docker_file = Path(docker_file)
    if not docker_file.exists():
        raise FileNotFoundError(f"Dockerfile not found: {docker_file}")

    build_path = Path(build_path)
    if not build_path.exists():
        raise FileNotFoundError(f"Build path not found: {build_path}")

    full_path = Path(build_path).resolve()
    tag = f"{image}:latest" if ":" not in image else image

    logger.info(f"Building container {tag} from {docker_file}")
    for item in docker_client.api.build(
        path=str(full_path),
        dockerfile=str(docker_file),
        tag=tag,
        nocache=force_rebuild,
        pull=force_rebuild,
        decode=True,
    ):
        if "error" in item:
            rich.print()
            raise RuntimeError(item["error"])
        if "stream" in item:
            rich.print("[dim]" + item["stream"].strip() + "[/]")

    logger.info(f"Container {tag} built successfully")
    return tag
