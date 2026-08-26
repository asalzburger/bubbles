import json
import argparse
import colorsys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from read_pixel_matrix import read_pixel_matrix


@dataclass(frozen=True)
class Seed:
    pixels: tuple[tuple[int, int], ...]

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        columns, rows = zip(*self.pixels)
        return min(columns), min(rows), max(columns), max(rows)


def saturation(pixel: tuple[int, ...]) -> float:
    red, green, blue = (channel / 255 for channel in pixel[:3])
    return colorsys.rgb_to_hsv(red, green, blue)[1] * 255


def find_seeds(
    pixel_matrix: list[list[tuple[int, ...]]],
    slice_height: int = 10,
    min_pixels: int = 5,
    saturation_threshold: float = 150,
) -> list[Seed]:
    """Find saturated 8-connected components within horizontal slices, bottom first."""
    if slice_height < 1 or min_pixels < 1:
        raise ValueError("slice_height and min_pixels must be positive")
    if not 0 <= saturation_threshold <= 255:
        raise ValueError("saturation_threshold must be between 0 and 255")

    height = len(pixel_matrix)
    width = len(pixel_matrix[0]) if height else 0
    seeds: list[Seed] = []

    for slice_end in range(height, 0, -slice_height):
        slice_start = max(0, slice_end - slice_height)
        candidates = {
            (column, row)
            for row in range(slice_start, slice_end)
            for column in range(width)
            if saturation(pixel_matrix[row][column]) >= saturation_threshold
        }

        while candidates:
            component = _pop_component(candidates)
            if len(component) >= min_pixels:
                seeds.append(Seed(tuple(component)))

    return seeds


def _pop_component(candidates: set[tuple[int, int]]) -> set[tuple[int, int]]:
    start = candidates.pop()
    component = {start}
    pending = [start]

    while pending:
        column, row = pending.pop()
        neighbors: Iterable[tuple[int, int]] = (
            (column + column_offset, row + row_offset)
            for row_offset in (-1, 0, 1)
            for column_offset in (-1, 0, 1)
            if column_offset or row_offset
        )
        for neighbor in neighbors:
            if neighbor in candidates:
                candidates.remove(neighbor)
                component.add(neighbor)
                pending.append(neighbor)

    return component


def draw_seeds(image_path: Path, seeds: list[Seed], output_path: Path) -> None:
    with Image.open(image_path) as image:
        annotated_image = image.convert("RGBA")

    overlay = Image.new("RGBA", annotated_image.size)
    overlay_draw = ImageDraw.Draw(overlay)
    for seed in seeds:
        for column, row in seed.pixels:
            overlay_draw.point((column, row), fill=(0, 255, 255, 255))
        overlay_draw.rectangle(seed.bounds, outline=(255, 0, 255, 255), width=1)

    Image.alpha_composite(annotated_image, overlay).save(output_path)


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(
        description="Find saturated 8-connected seed segments in horizontal image slices."
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=project_root / "resources" / "ABCMO_294_detail.png",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "resources" / "ABCMO_294_detail_seeds.png",
    )
    parser.add_argument("--slice-height", type=int, default=10)
    parser.add_argument("--min-pixels", type=int, default=5)
    parser.add_argument("--saturation-threshold", type=float, default=150)
    return parser.parse_args()

def extract_pixel_coordinates(Seed):
    folder_name = "resources"
    file_name = "pixels.json"
    pixel_coordinates = list(Seed.pixels)
    script_dir = Path(__file__).parent.parent
    target_path = script_dir / folder_name / file_name
    with open(target_path, 'w') as f:
            json.dump(pixel_coordinates, f, indent = 4)
    

def find_largest_seed(seeds: list[Seed], i = 0):
    largest_seed = max(seeds, key=lambda seed: len(seed.pixels))
    largest_seed_index = seeds.index(max(seeds, key=lambda seed: len(seed.pixels)))
    #largest_path = arguments.output.with_name(f"{arguments.output.stem}_largest{arguments.output.suffix}")
    #draw_seeds(arguments.image, [largest_seed], largest_path)
    print(f"Found largest seed at index {largest_seed_index}: {largest_seed}")
    match i:
        case 0:
            #print("Returning index of largest seed!", largest_seed_index)
            return(largest_seed_index)
        case 1:
            #print("Returning largest seed!", largest_index)
            return(largest_seed)
        case 2:
            #print("Returning index and largest seed!", largest_seed_index, largest_seed)
            return(largest_seed, largest_seed_index)

def save_specific_seed(seeds: list[Seed], n: int):
    chosen_seed  = seeds[n]
    chosen_path = arguments.output.with_name(f"{arguments.output.stem}_seed_{n}_{arguments.output.suffix}")
    draw_seeds(arguments.image, [chosen_seed], chosen_path)
    print(f"Saved overlay for seed {n} to {chosen_path}")
    extract_pixel_coordinates(chosen_seed)

if __name__ == "__main__":
    arguments = parse_arguments()
    matrix = read_pixel_matrix(arguments.image)
    found_seeds = find_seeds(
        matrix,
        slice_height=arguments.slice_height,
        min_pixels=arguments.min_pixels,
        saturation_threshold=arguments.saturation_threshold,
    )
    save_specific_seed(found_seeds, find_largest_seed(found_seeds))
    draw_seeds(arguments.image, found_seeds, arguments.output)
    print(f"Found {len(found_seeds)} seeds. Saved overlay to {arguments.output}.")