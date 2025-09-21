# %%
import carla
import math

def explore_waypoints(carla_map, step=20.0):
    """
    Systematically traverse the road network using waypoint.next().
    Returns a list of waypoints spaced by 'step'.
    """
    visited = set()
    to_visit = list(carla_map.generate_waypoints(step))  # seed with sparse grid
    result = []

    while to_visit:
        wp = to_visit.pop()
        key = (round(wp.transform.location.x, 1),
               round(wp.transform.location.y, 1),
               round(wp.transform.location.z, 1))

        if key in visited:
            continue

        visited.add(key)
        result.append(wp)

        # expand forward
        for nxt in wp.next(step):
            to_visit.append(nxt)

    return result


def draw_waypoint_network(world, waypoints, length = 2.0, step=20.0, life_time=0.0):
    """
    Draw waypoints as points + connect them with lines to show lane flow.
    """
    for wp in waypoints:
        loc = wp.transform.location + carla.Location(z=0.5)

        # Draw waypoint as green dot
        world.debug.draw_point(
            loc,
            size=0.1,
            color=carla.Color(0, 255, 0),
            life_time=life_time
        )

        yaw = math.radians(wp.transform.rotation.yaw)
        dx = math.cos(yaw)
        dy = math.sin(yaw)

        end = loc + carla.Location(x=dx * length, y=dy * length, z=0.0)

        world.debug.draw_arrow(
            loc,
            end,
            thickness=0.05,
            arrow_size=0.2,
            color=carla.Color(0, 255, 0),
            life_time=life_time
        )

        # Draw forward connections as red lines
        # for nxt in wp.next(step):
        #     world.debug.draw_line(
        #         loc,
        #         nxt.transform.location + carla.Location(z=0.5),
        #         thickness=0.1,
        #         color=carla.Color(255, 0, 0),
        #         life_time=life_time
        #     )


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    carla_map = world.get_map()

    step = 200.0  # meters between waypoints
    waypoints = explore_waypoints(carla_map, step=step)
    print(f"Discovered {len(waypoints)} sparse waypoints")

    draw_waypoint_network(world, waypoints, length=1.0, step=step, life_time=0.0)


if __name__ == "__main__":
    main()

# %%
import carla
import math

def explore_waypoints_from_topology(carla_map, step=5.0):
    """
    Traverse the entire map using topology and .next().
    Returns a list of waypoints spaced by 'step'.
    """
    topology = carla_map.get_topology()
    waypoints = []
    visited = set()

    for segment in topology:
        start_wp, end_wp = segment
        wp = start_wp

        while wp and wp.transform.location.distance(end_wp.transform.location) > step:
            key = (round(wp.transform.location.x, 1),
                   round(wp.transform.location.y, 1),
                   round(wp.transform.location.z, 1))

            if key not in visited:
                visited.add(key)
                waypoints.append(wp)

            next_wps = wp.next(step)
            if not next_wps:
                break
            wp = next_wps[0]  # just follow main lane

    return waypoints


def draw_waypoint_headings(world, waypoints, length=2.0, life_time=0.0):
    """
    Draw waypoints as arrows pointing along their heading (yaw).
    """
    for wp in waypoints:
        loc = wp.transform.location + carla.Location(z=0.5)

        # heading vector from yaw
        yaw = math.radians(wp.transform.rotation.yaw)
        dx, dy = math.cos(yaw), math.sin(yaw)
        end = loc + carla.Location(x=dx * length, y=dy * length)

        world.debug.draw_arrow(
            loc, end,
            thickness=0.05,
            arrow_size=0.2,
            color=carla.Color(0, 255, 0),
            life_time=life_time
        )


def main():
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    carla_map = world.get_map()

    # step controls spacing (meters)
    step = 2
    waypoints = explore_waypoints_from_topology(carla_map, step=step)
    print(f"Discovered {len(waypoints)} waypoints (step={step})")

    # draw arrows in-world
    draw_waypoint_headings(world, waypoints, length=2.0, life_time=0.0)


if __name__ == "__main__":
    main()
