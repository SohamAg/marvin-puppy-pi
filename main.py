import time
from inference import get_fake_inference
from actions import handle_command


def main():
    print("Starting Artrium controller")
    tick = 0

    while True:
        command = get_fake_inference(tick)
        print(f"[tick={tick}] inferred command = {command}")
        handle_command(command)
        tick += 1
        time.sleep(1)


if __name__ == "__main__":
    main()