from __future__ import annotations

from pathlib import Path


SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
}


def find_video_files(source_dir: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


def choose_video(videos: list[Path]) -> Path:
    print()
    print("Available videos:")
    print()
    for index, video in enumerate(videos, start=1):
        print(f"{index}. {video.name}")
    print()

    while True:
        selection = input("Choose a video by entering its number: ").strip()
        try:
            selected_index = int(selection)
        except ValueError:
            print("Invalid selection. Please enter a number from the list.")
            continue

        if 1 <= selected_index <= len(videos):
            return videos[selected_index - 1]

        print("Invalid selection. Please enter a number from the list.")
