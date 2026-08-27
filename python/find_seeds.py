import json
import argparse
import colorsys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import transform

from PIL import Image, ImageDraw, ImageFont

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


def draw_seeds(image_path: Path, seeds: list[Seed], output_path: Path, number_seeds: bool = False) -> None:
    with Image.open(image_path) as image:
        annotated_image = image.convert("RGBA")

    overlay = Image.new("RGBA", annotated_image.size)
    overlay_draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(15)
    for i, seed in enumerate(seeds):
        for column, row in seed.pixels:
            overlay_draw.point((column, row), fill=(0, 255, 255, 255))
        overlay_draw.rectangle(seed.bounds, outline=(255, 0, 255, 255), width=1)
        if number_seeds == True:
         overlay_draw.text((seed.bounds[0], seed.bounds[1]), str(i), font=font, fill=(16, 255, 0, 255))

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
    parser.add_argument("--slice-height", type=int, default=10, help="The height of the slices the image is divided into")
    parser.add_argument("--min-pixels", type=int, default=5)
    parser.add_argument("--saturation-threshold", type=float, default=150)
    parser.add_argument("--min-slope", type=float, default=0, help="The minimum amount of slope the transform will use")
    parser.add_argument("--max-slope", type=float, default=1, help="The maximum amount of slope the transform will use")
    parser.add_argument("--lines-amount", type=int, default=3, help="TEMPORARY UNTIL DYNAMIC IMPLEMENTATION! The amount of lines the transform will draw onto the seed")
    parser.add_argument("--number-seeds", type=bool, default=False, help="Whether or not the seeds should be numbered in the image")
    parser.add_argument("--seed-index", type=int, default=None, help="The index of the seed you want to analyze. Defaults to the largest seed")
    return parser.parse_args()

def extract_pixel_coordinates(Seed):
    folder_name = "resources"
    file_name = "pixels.json"
    pixel_coordinates = list(Seed.pixels)
    script_dir = Path(__file__).parent.parent
    target_path = script_dir / folder_name / file_name
    with open(target_path, 'w') as f:
            json.dump(pixel_coordinates, f, indent = 4)
    

def find_largest_seed(seeds: list[Seed], retcase = 0):
    largest_seed = max(seeds, key=lambda seed: len(seed.pixels))
    largest_seed_index = seeds.index(max(seeds, key=lambda seed: len(seed.pixels)))
    print(f"Found largest seed at index {largest_seed_index}")
    match retcase:
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
    line_path = arguments.output.with_name(f"{arguments.output.stem}_seed_{n}_transform_lines{arguments.output.suffix}")
    line_pixel_path = arguments.output.with_name(f"{arguments.output.stem}_seed_{n}_line_diagram_{arguments.output.suffix}")
    draw_seeds(arguments.image, [chosen_seed], chosen_path)
    #transform.draw_pixel_lines(arguments.min_slope, arguments.max_slope, chosen_path, line_pixel_path)
    transform.drawLines(chosen_path, arguments.lines_amount, line_path, arguments.min_slope, arguments.max_slope)
    print(f"Saved overlay for seed {n} to {chosen_path}")
    extract_pixel_coordinates(chosen_seed)
    transform.draw_diagram(arguments.min_slope, arguments.max_slope, f"Seed {n} m/t diagram")

if __name__ == "__main__":
    arguments = parse_arguments()
    matrix = read_pixel_matrix(arguments.image)
    found_seeds = find_seeds(
        matrix,
        slice_height=arguments.slice_height,
        min_pixels=arguments.min_pixels,
        saturation_threshold=arguments.saturation_threshold,
    )
    if arguments.seed_index is None:
        arguments.seed_index = find_largest_seed(found_seeds, 0)
    save_specific_seed(found_seeds, arguments.seed_index)
    draw_seeds(arguments.image, found_seeds, arguments.output, arguments.number_seeds)
    print(f"Found {len(found_seeds)} seeds. Saved overlay to {arguments.output}.")