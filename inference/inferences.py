#!/usr/bin/env python3

def get_command():
    command = input(
        "\nEnter command [stand / walk / left / right / stop / quit]: "
    ).strip().lower()

    command_map = {
        'stand': 'stand',
        'walk': 'walk',
        'left': 'left',
        'right': 'right',
        'stop': 'stop',
        'quit': 'quit'
    }

    return command_map.get(command, 'unknown')