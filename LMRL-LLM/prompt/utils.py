from typing import Dict

import roslibpy


def append_service(
    client: roslibpy.Ros, name: str, services: Dict[str, roslibpy.Service]
) -> Dict[str, roslibpy.Service]:
    """Update current services with the required one.

    Args:
        client (roslibpy.Ros): ROS client.
        name (str): Required service name.
        services (Dict[str, roslibpy.Service]): Dictionary of current services.

    Returns:
        Dicr[str, roslibpy.Service]: Updated dictionary of services.
    """
    if name not in services:
        services[name] = roslibpy.Service(client, name, client.get_service_type(name))
    return services
