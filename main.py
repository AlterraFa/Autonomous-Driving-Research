import os, sys
import resource
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
sys.path.insert(0, root)
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

import carla
import argparse
import pygame
from src.messages.logger import Logger
from src.control.world import World
from mode import MODE_RUNNERS

class MainArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._subparsers_action = None

    def add_subparsers(self, *args, **kwargs):
        self._subparsers_action = super().add_subparsers(*args, **kwargs)
        return self._subparsers_action

    def format_help(self):
        help_text = super().format_help().rstrip()

        if self._subparsers_action and self._subparsers_action.choices:
            mode_sections = ["\nMode-specific flags:"]

            for mode_name, mode_parser in self._subparsers_action.choices.items():
                mode_sections.append(f"\n  {mode_name}:")
                for action in mode_parser._actions:
                    if action.dest == "help":
                        continue

                    option_text = ", ".join(action.option_strings)
                    if action.nargs in ["+", "*"]:
                        metavar = action.metavar if action.metavar else action.dest.upper()
                        option_text = f"{option_text} {metavar}"
                    elif action.metavar:
                        option_text = f"{option_text} {action.metavar}"
                    elif action.option_strings and action.type is not None and action.nargs != 0:
                        option_text = f"{option_text} {action.dest.upper()}"

                    required_mark = " (required)" if getattr(action, "required", False) else ""
                    help_line = action.help or ""
                    mode_sections.append(f"    {option_text}{required_mark}"
                                         + (f"\n      {help_line}" if help_line else ""))

            help_text += "\n" + "\n".join(mode_sections)

        return help_text + "\n"

    def error(self, message):
        self.print_help(sys.stderr)
        self.exit(2, f"\n{self.prog}: error: {message}\n")


def main(args):
    pygame.init()
    Logger.set_levels("INFO", "WARNING", "ERROR", "CUSTOM", "DEBUG" if args.debug else "INFO")

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    virt_world = World(client, args.traffic_port)
    virt_world.sync           = args.sync
    virt_world.delta          = args.delay
    virt_world.disable_render = False
    virt_world.apply_settings()

    folder  = os.path.dirname(os.path.abspath(__file__))

    viewer_args = {
        "world"  : virt_world,
        "width"  : args.width,
        "height" : args.height,
        "sync"   : args.sync,
        "fps"    : args.fps,
    }

    lp = MODE_RUNNERS[args.mode](args, client, virt_world, folder, viewer_args)

    if lp is not None:
        lp.dump_stats("profile_results.lprof")
    
    
if __name__ == "__main__":
    argparser = MainArgumentParser(description = "CARLA Runner")

    # ====================================================== #
    #                   SHARED ARGUMENT
    # ====================================================== #
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Activate synchronous mode execution')
    argparser.add_argument(
        "--traffic-port",
        metavar = "TMP",
        default = 8000,
        type = int,
        help = "Traffic manager port for actor autopilot function"
    )
    argparser.add_argument(
        "--delay",
        default = 0.05,
        type = float,
        help = "Time step for synchronize server running"
    )
    argparser.add_argument(
        "--timeout",
        default = 10,
        type = float,
        help = "Set timeout for carla client"
    )
    argparser.add_argument(
        "--fps",
        default = 144,
        type = float,
        help = "Max fps for pygame rendering"
    )
    argparser.add_argument(
        "--debug",
        action = "store_true",
        help = "Set logger debugging flag"
    )

    subparser = argparser.add_subparsers(dest = "mode", help = "Execution mode", required = True)

    
    # ====================================================== #
    #                MANUAL CONTROL ARGUMENT
    # ====================================================== #
    manual_parser = subparser.add_parser("manual", help = "Manual driving and recording states")    
    manual_parser.add_argument(
        "--record",
        type = str,
        default = None,
        help = "Specify the relative root log directory and enable the record mode"
    )
    manual_parser.add_argument(
        "--ignore-signs",
        type = float,
        default = 0,
        help = "Ignore traffic sign rules (by percentage)"
    )
    
    
    # ====================================================== #
    #                   REPLAY ARGUMENT
    # ====================================================== #
    replay_parser = subparser.add_parser("replay", help = "Replay the recorded CARLA's states")
    replay_parser.add_argument(
        "--replay-dir",
        type = str,
        nargs = "+",
        help = "Replay Carla recording (.log file path is needed, recording time of .npy must correspond to .log)",   
        required = True
    )
    replay_parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporal (time-based) waypoint generation instead of spatial."
    )
    replay_parser.add_argument(
        "--draw-waypoints",
        action = "store_true",
        help = "Draw debugging waypoints onto the world"
    )
    replay_parser.add_argument(
        "--collect-data",
        type = str,
        default = None,
        help = "Data collection directory for DNN training"
    )
    replay_parser.add_argument(
        "--use-turn",
        action = "store_true",
        help = "Turn on turn classification at junctions"
    )
    replay_parser.add_argument(
        "--headless",
        action = "store_true",
        help = "Enable Pygame headless rendering"
    )
    replay_parser.add_argument(
        "--clear-npcs",
        action = "store_true",
        help = "Clear out all NPCs rendering"
    )
    replay_parser.add_argument(
        "--redo-traj",
        action = "store_true",
        help = "Redo Trajectory logging"
    )

    # ====================================================== #
    #                  INFERENCE ARGUMENT
    # ====================================================== #
    infer_parser = subparser.add_parser("inference", help = "Autonomous inference")
    infer_parser.add_argument(
        "--draw-waypoints",
        action = "store_true",
        help = "Draw debugging waypoints onto the world"
    )
    infer_parser.add_argument(
        "--model-path",
        type = str,
        help = "Path to models file as well as its class reference",
        required = True
    )
    infer_parser.add_argument(
        "--render-map",
        action = "store_true",
        help = "Force render map"
    )
    
    args = argparser.parse_args()
    args.width, args.height = [int(x) for x in args.res.split('x')]

    main(args)